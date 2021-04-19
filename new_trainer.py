import tensorflow as tf
from models.depth_decoder_creater import DepthDecoder_full
from models.encoder_creater import ResNet18_new
from models.posenet_decoder_creator import PoseDecoder

from utils import disp_to_depth, concat_pose_params
from src.trainer_helper import *
# from src.dataset_loader import DataLoader
from src.dataset_loader_kitti_raw import DataLoader_KITTI_Raw
from src.DataPprocessor import DataProcessor

import numpy as np
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import sys
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.models = {}
        self.num_frames_per_batch = len(self.opt.frame_idx)
        self.data_loader = DataLoader_KITTI_Raw(self.opt.num_epochs, self.opt.batch_size, debug_mode=self.opt.debug_mode,
                                                dataset_for='train', split_name='eigen_zhou')
        self.shape_scale0 = [self.opt.batch_size, self.opt.height, self.opt.width, 3]

        self.tgt_image = None
        self.tgt_image_net = None
        self.tgt_image_aug = None
        self.avg_reprojection = False

        self.tgt_image_all = []
        self.src_image_stack_all = []
        self.proj_image_stack_all = []
        self.proj_error_stack_all = []
        self.pred_auto_masks = []

        self.pixel_losses = 0.
        self.smooth_losses = 0.
        # self.total_loss = 0.
        self.losses = {}
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.batch_idx = tf.Variable(0, dtype=tf.int32, trainable=False)

        self.batch_processor = DataProcessor(not self.opt.debug_mode)
        self.epoch_cnt = tf.Variable(0, dtype=tf.int64, trainable=False)
        boundaries = [self.opt.lr_step_size, self.opt.lr_step_size*2]           # [15, 30]
        values = [self.opt.learning_rate * scale for scale in [1, 0.1, 0.01]]   # [1e-4, 1e-5, 1e-6]
        self.lr_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
        learning_rate = self.lr_fn(self.epoch_cnt)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.train_sum_writer = None
        self.weights_idx = 0    # for saved weights
        self.back_project_dict = {}
        self.project_3d_dict = {}

        self.init_app()

    def init_app(self):
        """Initiate Pose, Depth and Auto-Masking models
        self.models['depth_enc'] = ResNet18_new([2, 2, 2, 2]), channel_num=3
        self.models['depth_dec'] = DepthDecoder_full()
        self.models['pose_enc'] = ResNet18_new([2, 2, 2, 2]), channel_num=6
        self.models['pose_dec'] = PoseDecoder(num_ch_enc=[64, 64, 128, 256, 512])
        """

        self.models['depth_enc'] = ResNet18_new([2, 2, 2, 2])
        self.models['depth_dec'] = DepthDecoder_full()
        self.models['pose_enc'] = ResNet18_new([2, 2, 2, 2])
        shapes = [(1, 96, 320, 64), (1, 48, 160, 64), (1, 24, 80, 128), (1, 12, 40, 256), (1, 6, 20, 512)]
        inputs = [tf.random.uniform(shape=(shapes[i])) for i in range(len(shapes))]
        self.models['pose_dec'] = PoseDecoder(num_ch_enc=[64, 64, 128, 256, 512])
        _ = self.models['pose_dec'].predict(inputs)

        build_models(self.models, show_summary=False, check_outputs=False)
        self.load_models()

        # for scale in range(self.opt.num_scales):
        self.back_project_dict[self.opt.src_scale] = BackProjDepth(self.shape_scale0, self.opt.src_scale)
        self.project_3d_dict[self.opt.src_scale] = Project3D(self.shape_scale0, self.opt.src_scale)

    def load_models(self):
        if self.opt.from_scratch:
            print("\tfrom scratch, no weights loaded")
        else:
            print("\tloading trained models from %s..." % self.opt.weights_dir)
            for m_name in self.opt.models_to_load:
                self.models[m_name].load_weights(os.path.join(self.opt.weights_dir, m_name + '.h5'))

    # ----- For process_batch() -----
    def get_smooth_loss(self, disp, img):
        norm_disp = disp / (tf.reduce_mean(disp, [1, 2], keepdims=True) + 1e-7)
        grad_disp_x = tf.math.abs(norm_disp[:, :-1, :, :] - norm_disp[:, 1:, :, :])
        grad_disp_y = tf.math.abs(norm_disp[:, :, :-1, :] - norm_disp[:, :, 1:, :])

        grad_img_x = tf.math.abs(img[:, :-1, :, :] - img[:, 1:, :, :])
        grad_img_y = tf.math.abs(img[:, :, :-1, :] - img[:, :, 1:, :])

        weight_x = tf.math.exp(-tf.reduce_mean(grad_img_x, 3, keepdims=True))
        weight_y = tf.math.exp(-tf.reduce_mean(grad_img_y, 3, keepdims=True))

        smoothness_x = grad_disp_x * weight_x
        smoothness_y = grad_disp_y * weight_y

        return tf.reduce_mean(smoothness_x) + tf.reduce_mean(smoothness_y)

    def compute_reproject_loss(self, proj_image, tgt_image):
        l1_loss = tf.reduce_mean(tf.math.abs(proj_image - tgt_image), axis=3, keepdims=True)
        ssim_loss = tf.reduce_mean(SSIM(proj_image, tgt_image), axis=3, keepdims=True)
        loss = self.opt.ssim_ratio * ssim_loss + (1 - self.opt.ssim_ratio) * l1_loss
        return loss

    def reset_losses(self):
        self.smooth_losses = 0.
        self.pixel_losses = 0.
        # self.total_loss = 0.
        self.losses = {}
        self.proj_image_stack_all = []
        self.proj_error_stack_all = []
        self.pred_auto_masks = []
        self.tgt_image_all = []

    def compute_losses(self, input_imgs, outputs):
        sourec_scale = 0
        tgt_image = input_imgs[('color', 0, sourec_scale)]
        self.reset_losses()
        total_loss = 0.

        for scale in range(self.opt.num_scales):
            # -------------------
            # 1. Reprojection / warping loss
            # -------------------
            reproject_losses = []
            for i, f_i in enumerate(self.opt.frame_idx[1:]):
                proj_image = outputs[('color', f_i, scale)]
                assert proj_image.shape[2] == tgt_image.shape[2] == self.opt.width
                reproject_losses.append(self.compute_reproject_loss(proj_image, tgt_image))

                # for collect
                proj_error = tf.math.abs(proj_image - tgt_image)
                if i == 0:
                    proj_image_stack = proj_image
                    proj_error_stack = proj_error
                else:
                    proj_image_stack = tf.concat([proj_image_stack, proj_image], axis=3)
                    proj_error_stack = tf.concat([proj_error_stack, proj_error], axis=3)

            reproject_losses = tf.concat(reproject_losses, axis=3)

            if self.avg_reprojection:
                reproject_loss = tf.math.reduce_mean(reproject_losses, axis=3, keepdims=True)
            else:
                reproject_loss = reproject_losses

            # -------------------
            # 2. Optional: auto-masking
            # identity error between source vs. current scale
            # -------------------
            if not self.opt.do_automasking:
                combined = reproject_losses
            else:
                identity_reprojection_losses = []
                for f_i in self.opt.frame_idx[1:]:
                    proj_image = input_imgs[('color', f_i, sourec_scale)]
                    identity_reprojection_losses.append(self.compute_reproject_loss(proj_image, tgt_image))

                identity_reprojection_losses = tf.concat(identity_reprojection_losses, axis=3)

                # if use average reprojection loss
                if self.avg_reprojection:
                    identity_reprojection_loss = tf.math.reduce_mean(identity_reprojection_losses, axis=3, keepdims=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

                # add random numbers to break ties
                identity_reprojection_loss += (tf.random.normal(identity_reprojection_loss.shape)
                                               * tf.constant(1e-5, dtype=tf.float32))
                combined = tf.concat([identity_reprojection_loss, reproject_loss], axis=3)

                self.pred_auto_masks.append(
                    tf.expand_dims(
                        tf.cast(tf.math.argmin(combined, axis=3) > 1, tf.float32) * 255, -1))

            # -------------------
            # 3. Final reprojectioni loss -> as pixel loss
            # -------------------
            if combined.shape[-1] != 1:
                to_optimise = tf.reduce_min(combined, axis=3)
            else:
                to_optimise = combined
            reprojection_loss = tf.reduce_mean(to_optimise)
            self.pixel_losses += reprojection_loss

            # -------------------
            # 4. Smoothness loss: Gradient Loss based on image pixels
            # -------------------
            disp_s = outputs[('disp', scale)]
            tgt_image_s = input_imgs['color', 0, scale]
            smooth_loss_raw = self.get_smooth_loss(disp_s, tgt_image_s)
            smooth_loss = self.opt.smoothness_ratio * smooth_loss_raw / 2 ** scale

            # ------------------
            # 5. Overall Loss, accumulate scale-wise
            # ------------------
            total_loss_tmp = reprojection_loss + smooth_loss
            total_loss += total_loss_tmp

            # ------------------
            # Optional: Collect results for summary
            # ------------------
            self.smooth_losses += smooth_loss
            self.losses['loss/%d' % scale] = total_loss_tmp
            self.proj_image_stack_all.append(proj_image_stack)
            self.proj_error_stack_all.append(proj_error_stack)
            if scale == 0:
                src_image_stack_aug = tf.concat([input_imgs[('color_aug', -1, 0)],
                                                 input_imgs[('color_aug', -1, 0)]], axis=3)
                self.src_image_stack_all.append(src_image_stack_aug)
                self.tgt_image_all.append(tgt_image)

        self.pixel_losses /= self.opt.num_scales
        self.smooth_losses /= self.opt.num_scales
        total_loss /= self.opt.num_scales

        self.losses['loss/total'] = total_loss
        if self.opt.debug_mode:
            colormapped_normal = colorize(outputs[('disp', 0)], cmap='plasma')
            print("disp map shape", colormapped_normal.shape)
            plt.imshow(colormapped_normal.numpy()), plt.show()

        return total_loss


    def generate_images_pred(self, input_imgs, input_Ks, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        # -----------
        # with open(r"D:\MA\Recources\monodepth2-original\check_inverse_warp_model.pkl", 'rb') as df:
        #     imported = pickle.load(df)
        # inputs_imp = imported['inputs_imgs_Ks']
        # # outputs_imp = imported['outputs']
        # -----------

        for scale in range(self.opt.num_scales):
            # print("===== scale %d ====="%scale)

            # -----------------------------
            disp_tf = outputs[("disp", scale)]
            # disp_tf = outputs_imp[("disp", scale)]
            # disp_tf = np.transpose(disp_tf, [0, 2, 3, 1])
            # print("disp_tf shape", disp_tf.shape)
            # -----------------------------

            disp = tf.image.resize(disp_tf, (self.opt.height, self.opt.width))
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            # -----------------------------
            # outputs[("depth", 0, scale)] = depth
            # outputs_imp[("depth", 0, scale)] = depth
            # -----------------------------

            for i, f_i in enumerate(self.opt.frame_idx[1:]):
                T = outputs[("cam_T_cam", 0, f_i)]

                # ----------
                cam_points = self.back_project_dict[self.opt.src_scale].run_func(depth, input_Ks[("inv_K", self.opt.src_scale)])
                pix_coords = self.project_3d_dict[self.opt.src_scale].run_func(cam_points, input_Ks[("K", self.opt.src_scale)], T)

                # ----------
                outputs[("sample", f_i, scale)] = pix_coords

                # -----------
                input_tf = input_imgs[("color", f_i, self.opt.src_scale)]
                # input_tf = inputs_imp[("color", f_i, self.opt.src_scale)]
                # input_tf = np.transpose(input_tf, [0,2,3,1])
                # ------------
                res = bilinear_sampler(input_tf, outputs[("sample", f_i, scale)])

                outputs[("color", f_i, scale)] = res

        if self.opt.debug_mode:
            for i in range(self.opt.batch_size):
                print(i)
                fig = plt.figure(figsize=(3, 2))
                fig.add_subplot(3, 2, 1)
                tgt = input_imgs[('color', 0, 0)][i].numpy()
                # tgt = inputs_imp[('color', 0, 0)][i]
                # tgt = np.transpose(tgt, [1,2,0])
                plt.imshow(tgt)

                fig.add_subplot(3, 2, 3)
                src0 = input_imgs[('color_aug', -1, 0)][i].numpy()
                # src0 = inputs_imp[('color_aug', -1, 0)][i]
                # src0 = np.transpose(src0, [1,2,0])
                print(np.max(np.max(src0)), np.min(np.min(src0)))
                plt.imshow(src0)

                fig.add_subplot(3, 2, 4)
                src1 = input_imgs[('color_aug', 1, 0)][i].numpy()
                # src1 = inputs_imp[('color_aug', 1, 0)][i]
                # src1 = np.transpose(src1, [1, 2, 0])
                print(np.max(np.max(src1)), np.min(np.min(src1)))
                plt.imshow(src1)

                fig.add_subplot(3, 2, 5)
                out0 = outputs[("color", -1, 0)][i].numpy()
                print(np.max(np.max(out0)), np.min(np.min(out0)))
                plt.imshow(out0)
                fig.add_subplot(3, 2, 6)
                out1 = outputs[("color", 1, 0)][i].numpy()
                print(np.max(np.max(out1)), np.min(np.min(out1)))
                plt.imshow(out1)
                plt.show()

    def predict_poses(self, input_imgs, outputs):
        """Use pose enc-dec to calculate camera's angles and translations"""
        # ---------------
        # with open("D:\MA\Recources\monodepth2-original\check_inverse_warp_model.pkl", 'rb') as df:
        #     imported = pickle.load(df)
        # outputs_imp = imported['outputs']
        # input_imp = imported['inputs_imgs_Ks']
        # frames_for_pose = {f_i: tf.constant(np.transpose(input_imp[("color_aug", f_i, 0)],[0,2,3,1]))
        #                    for f_i in self.opt.frame_idx}
        frames_for_pose = {f_i: input_imgs[("color_aug", f_i, 0)] for f_i in self.opt.frame_idx}
        # ---------------

        for f_i in self.opt.frame_idx[1:]:
            # To maintain ordering we always pass frames in temporal order
            if f_i < 0:
                pose_inputs = [frames_for_pose[f_i], frames_for_pose[0]]
            else:
                pose_inputs = [frames_for_pose[0], frames_for_pose[f_i]]

            pose_inputs = self.models["pose_enc"](tf.concat(pose_inputs, axis=3), training=self.opt.train_pose)

            pred_pose_raw = self.models["pose_dec"](pose_inputs, training=self.opt.train_pose)
            # ---------
            axisangle = pred_pose_raw['angles']
            translation = pred_pose_raw['translations']
            # axisangle = outputs_imp[("axisangle", 0, f_i)]
            # translation = outputs_imp[("translation", 0, f_i)]
            # ---------
            # Invert the matrix if the frame id is negative
            outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
        return outputs

    def process_batch(self, input_imgs, input_Ks):
        """The whoel pipeline implemented in minibatch (pairwise images)
        1. Use Depth enc-dec, to predict disparity map in mutli-scales
        2. Use Pose enc-dec, to predict poses
        3. Use products from 1.2. to generate image (reprojection) predictions
        4. Compute Losses from 3.
        """
        # ------------
        # with open("D:\MA\Recources\monodepth2-original\check_inverse_warp_model.pkl", 'rb') as df:
        #     imported = pickle.load(df)
        # inputs_imp = imported['inputs_imgs_Ks']
        # tgt_image_aug = tf.constant(np.transpose(inputs_imp[('color_aug', 0, 0)], [0,2,3,1]))
        tgt_image_aug = input_imgs[('color_aug', 0, 0)]
        # -------------

        # Depth Encoder
        feature_raw = self.models['depth_enc'](tgt_image_aug, self.opt.train_depth)
        # Depth Decoder
        pred_disp = self.models['depth_dec'](feature_raw, self.opt.train_depth)
        outputs = {}
        for s in range(self.opt.num_scales):
            # Collect raw disp prediction at each scale
            disp_raw = pred_disp["output_%d"%s]
            outputs[('disp', s)] = disp_raw
            # Collect depth at respective scale, but resized to source_scale
            disp_src_size = tf.image.resize(disp_raw, (self.opt.height, self.opt.width))
            _, outputs[('depth', s)] = disp_to_depth(disp_src_size, self.opt.min_depth, self.opt.max_depth)

        # -------------
        # 2. Pose
        # -------------
        outputs.update(self.predict_poses(input_imgs, outputs))
        # -------------
        # 3. Generate Reprojection from 1, 2
        # -------------
        # self.generate_images_pred(input_imgs, input_K_mulscale, outputs)
        self.generate_images_pred(input_imgs, input_Ks, outputs)

        # -------------
        # 4. Compute Losses from 3.
        # - Conducted in training procecss, here just for better understanding
        # self.compute_losses(input_imgs, outputs)
        # -------------
        return outputs

    def start_training(self, dataset):
        """Training loop
        Tf.keras 与 pytorch 相似，只是将 loss.backward() 换成 tf.GradientTape()
        - loss function 需要自己定义, 返回 total loss即可, 就可通过 GradientTape 跟踪
        """
        dataset_iter = iter(dataset)
        for epoch in range(self.opt.num_epochs):
            self.epoch_cnt.assign(epoch)
            self.optimizer.lr = self.lr_fn(epoch)     # learning rate 15:1e-4; >16:1e-5
            print("\tlearning rate - epoch %d: " % epoch, self.optimizer.get_config()['learning_rate'])

            for i in tqdm(range(self.data_loader.steps_per_epoch),
                          desc='Epoch%d/%d' % (epoch+1, self.opt.num_epochs)):
                self.batch_idx.assign(i)
                # data preparation
                batch = dataset_iter.get_next()
                trainable_weights_all = self.get_trainable_weights()
                input_imgs, input_Ks = self.batch_processor.prepare_batch(batch[..., :3], batch[..., 3:])
                # training
                # grads = self.grad(input_imgs, input_Ks, trainable_weights_all)
                grads, outputs = self.grad(input_imgs, input_Ks, trainable_weights_all)
                self.optimizer.apply_gradients(zip(grads, trainable_weights_all))
                self.global_step.assign_add(1)

                if (self.global_step + 1) % (self.data_loader.steps_per_epoch // 2) == 0:
                    if not self.opt.debug_mode:
                        self.save_models()

    # @tf.function    # turn off to debug, e.g. with plt
    def grad(self, input_imgs, input_Ks, trainables):
        with tf.GradientTape() as tape:
            outputs = self.process_batch(input_imgs, input_Ks)
            total_loss = self.compute_losses(input_imgs, outputs)
            grads = tape.gradient(total_loss, trainables)
            if self.global_step % self.opt.record_freq == 0:
                tf.print("loss: ", self.losses['loss/total'], output_stream=sys.stdout)
            # # collect data
            # early_phase = self.batch_idx % self.summary_freq == 0 and self.global_step < 2000
            # late_phase = self.global_step % 2000 == 0
            # if early_phase or late_phase:
            # print("colleting data in step ", self.global_step, " batch index ", self.batch_idx)
            #     with self.train_sum_writer.as_default():
            #         self.collect_summary(outputs)
            # if self.global_step % self.summary_freq == 0:
            #     print("flush data in step ", self.global_step, " batch index ", self.batch_idx)
            #     self.train_sum_writer.flush()

        return grads, outputs

    def get_trainable_weights(self):
        trainable_weights_all = []
        for m_name, model in self.models.items():
            if self.opt.train_pose and "pose" in m_name:
                # print("Training %s ..."%m_name)
                trainable_weights_all.extend(model.trainable_weights)
            if self.opt.train_depth and "depth" in m_name:
                # print("Training %s ..."%m_name)
                trainable_weights_all.extend(model.trainable_weights)
        return trainable_weights_all

    def train(self):
        if not self.opt.train_pose and not self.opt.train_depth:
            print("specify a model to train")
            return

        dataset = self.data_loader.build_dataset()
        if self.opt.recording:
            train_log_dir = self.opt.record_summary_path + self.opt.model_name + "/" + self.opt.run_mode
            self.train_sum_writer = tf.summary.create_file_writer(train_log_dir)
            print('Summary will be stored in %s ' % train_log_dir)
        # for epoch in range(self.opt.num_epochs):
        print("->Start training...")
        self.start_training(dataset)

    def save_models(self, not_saved=()):
        weights_path = os.path.join(self.opt.save_model_path, 'weights_%d' % self.weights_idx)
        self.weights_idx += 1
        if not os.path.isdir(weights_path):
            os.makedirs(weights_path)
        for m_name, model in self.models.items():
            if m_name not in not_saved:
                m_path = os.path.join(weights_path, m_name + '.h5')
                tf.print("saving {} to:".format(m_name), m_path, output_stream=sys.stdout)
                model.save_weights(m_path)

    def collect_summary(self, outputs):
        # total losses in multiple scales
        # tf.print("saving total losses in different scales...")
        for loss_name, loss_val in self.losses.items():
            tf.summary.scalar(loss_name, loss_val, step=self.epoch_cnt)

        # sub-losses
        tf.print("saving pixel and smoothness loss...")
        tf.summary.scalar("pixel loss", self.pixel_losses, step=self.epoch_cnt)
        tf.summary.scalar("smooth loss", self.smooth_losses, step=self.epoch_cnt)

        # poses
        # axis_names = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
        # for i in range(len(axis_names)):
        #     tf.summary.histogram(axis_names[i], self.pred_poses[:, :, :, i], step=self.epoch_cnt)

        # images
        tf.summary.image('tgt_image', self.tgt_image_all[0], step=self.epoch_cnt)
        for s in range(self.opt.num_scales):
            tf.summary.image('scale{}_disparity_color_image'.format(s),
                             colorize(outputs['disp',s], cmap='plasma'), step=self.epoch_cnt)
            # tf.summary.image('scale{}_disparity_gray_image'.format(s),
            #                  normalize_image(outputs['disp',s]), step=self.epoch_cnt)
            for i in range(2):
                tf.summary.image('scale{}_projected_image_{}'.format(s, i),
                                 self.proj_image_stack_all[s][:, :, :, i * 3:(i + 1) * 3], step=self.epoch_cnt)
                tf.summary.image('scale{}_proj_error_{}'.format(s, i),
                                     self.proj_error_stack_all[s][:, :, :, i * 3:(i + 1) * 3], step=self.epoch_cnt)
            if self.opt.do_automasking:
                tf.summary.image('scale{}_automask_image'.format(s),
                                 self.pred_auto_masks[s], step=self.epoch_cnt)

    def compute_depth_losses(self, input_imgs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training
        -> Only for KITTI-RawData, where velodyne depth map is valid !!!

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = tf.clip_by_value(tf.image.resize(depth_pred, [375, 1242]), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = input_imgs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = tf.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= np.median(depth_gt) / np.median(depth_pred)

        depth_pred = tf.clip_by_value(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)
        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    # def is_time_to_flush(self):
    #     decision = False
    #     if self.global_step < 2000 and self.batch_idx % (self.summary_freq * 2) == 0:
    #         decision = True
    #     elif self.global_step >= 2000 and self.is_time_to_collect():
    #         decision = True
    #     if decision:
    #         tf.print("data is flushed in step ", self.global_step, output_stream=sys.stdout)
    #     return decision
    #
    # def is_time_to_collect(self):
    #     if self.train_sum_writer is None:
    #         return False
    #     early_phase = self.batch_idx % self.summary_freq == 0 and self.global_step < 2000
    #     late_phase = self.global_step % 2000 == 0
    #     decision = early_phase or late_phase
    #     if decision:
    #         tf.print("data is collected in step ", self.global_step, output_stream=sys.stdout)
    #     return decision


if __name__ == '__main__':
    trainer = Trainer(options)
    trainer.train(record=False)
