import tensorflow as tf
from models.depth_decoder_creater import DepthDecoder_full
from models.encoder_creater import ResNet18_new
from models.posenet_decoder_creator import PoseDecoder

from utils import del_files
from src.trainer_helper import *
from datasets.data_loader_kitti import DataLoader as DataLoaderKITTI
from datasets.data_loader_custom import DataLoader as DataLoaderCustom
from datasets.dataset_kitti import KITTIRaw, KITTIOdom, VeloChallenge
from src.DataPreprocessor_exp import DataProcessor as Preprocessor_exp
from src.DataPprocessor import DataProcessor as Preprocessor

import numpy as np
from collections import defaultdict
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
        self.feed_size = list(map(int, self.opt.feed_size))
        self.models = {}
        self.num_frames_per_batch = len(self.opt.frame_idx)
        self.train_loader = None
        self.val_loader = None
        self.mini_val_loader= None
        self.tgt_image = None
        self.tgt_image_net = None
        self.tgt_image_aug = None
        self.avg_reprojection = False
        self.train_loss_tmp = []
        self.losses = {}
        self.early_stopping_losses = defaultdict(list)
        self.global_step = tf.constant(0, dtype=tf.int64)
        self.depth_metric_names = []
        self.val_losses_min = defaultdict(lambda: 10)    # random number as initiation
        self.mini_val_iter = None
        self.train_iter = None
        self.val_iter = None
        self.batch_processor = None
        self.has_depth_gt = False
        self.lr_fn = None
        self.optimizer = None

        self.summary_writer = {}
        self.weights_idx = 0    # for saved weights
        self.back_project_dict = {}
        self.project_3d_dict = {}

        self.init_app()

    def init_app(self):
        """Init dataset Class, init Pose, Depth and Auto-Masking models
        self.models['depth_enc'] = ResNet18_new([2, 2, 2, 2]), channel_num=3
        self.models['depth_dec'] = DepthDecoder_full()
        self.models['pose_enc'] = ResNet18_new([2, 2, 2, 2]), channel_num=6
        self.models['pose_dec'] = PoseDecoder(num_ch_enc=[64, 64, 128, 256, 512])
        """
        view_options(self.opt)
        # Choose dataset
        dataset_choices = {
            'kitti_raw': KITTIRaw,
            'kitti_odom': KITTIOdom,
            'velocity': VeloChallenge,
        }
        split_folder = os.path.join('splits', self.opt.split)
        train_file = 'train_files.txt'
        val_file = 'val_files.txt'

        # Train dataset & loader
        train_dataset = dataset_choices[self.opt.dataset](
            split_folder, train_file, data_path=self.opt.data_path)
        # if 'custom' in self.opt.dataset:
        #     DataLoader = DataLoaderCustom
        # else:
        DataLoader = DataLoaderKITTI
        self.train_loader = DataLoader(train_dataset, self.opt.num_epochs, self.opt.batch_size, self.opt.frame_idx,
                                       inp_size=self.feed_size, sharpen_factor=self.opt.sharpen_factor)
        buffer_size = 1000 if self.opt.feed_size[1] == 192 else 500
        self.train_iter = self.train_loader.build_train_dataset(buffer_size=buffer_size)

        # Validation dataset & loader
        # - mini-val during the epoch
        mini_val_dataset = dataset_choices[self.opt.dataset](
            split_folder, val_file, data_path=self.opt.data_path)
        self.mini_val_loader = DataLoader(mini_val_dataset, self.opt.num_epochs, batch_size=4,
                                          frame_idx=self.opt.frame_idx, inp_size=self.feed_size, sharpen_factor=None)
        self.mini_val_iter = self.mini_val_loader.build_val_dataset(include_depth=self.train_loader.has_depth,
                                                                    buffer_size=100, shuffle=True)
        # - Val after one epoch
        val_dataset = dataset_choices[self.opt.dataset](
            split_folder, val_file, data_path=self.opt.data_path)
        self.val_loader = DataLoader(val_dataset, num_epoch=self.opt.num_epochs, batch_size=2,
                                     frame_idx=self.opt.frame_idx, inp_size=self.feed_size, sharpen_factor=None)
        self.val_iter = self.val_loader.build_val_dataset(include_depth=self.train_loader.has_depth,
                                                          buffer_size=100, shuffle=True)
        if self.train_loader.has_depth:
            # Val metrics only when depth_gt available
            self.has_depth_gt = self.train_loader.has_depth
            self.depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms",
                                       "da/a1", "da/a2", "da/a3"]

        # # Batch data preprocessor
        # if self.opt.random_crop:
        #     DataProcessor = Preprocessor_exp
        # else:
        DataProcessor = Preprocessor_exp
        self.batch_processor = DataProcessor(frame_idx=self.opt.frame_idx, intrinsics=train_dataset.K,
                                             inp_size=self.feed_size, dataset_name=train_dataset.name)

        # Init models
        self.models['depth_enc'] = ResNet18_new(norm_inp=self.opt.depth_norm_inp)
        self.models['depth_dec'] = DepthDecoder_full()
        self.models['pose_enc'] = ResNet18_new()
        self.models['pose_dec'] = PoseDecoder(num_ch_enc=[64, 64, 128, 256, 512])

        build_models(self.models, inp_shape=self.feed_size)
        self.load_models()
        self.unfreeze_partial_models()

        # Set optimizer
        # Originally: [15], [1e-4, 1e-5], decay 10
        # exp: [3,6], [1e-4, /5, /25], decay 5
        boundaries = [self.opt.lr_step_size, self.opt.lr_step_size*2]
        values = [self.opt.learning_rate / scale for scale in [1, self.opt.lr_decay, self.opt.lr_decay**2]]# [1e-4, 5e-5, 1e-5]
        self.lr_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_fn(1))

        # Init inverse-warping helpers
        # shape_scale0 = (self.opt.batch_size, self.opt.height, self.opt.width, 3)
        # self.back_project_dict[self.opt.src_scale] = BackProjDepth(shape_scale0, self.opt.src_scale)
        # self.project_3d_dict[self.opt.src_scale] = Project3D(shape_scale0, self.opt.src_scale)

    def load_models(self):
        if self.opt.from_scratch:
            print('\ttraining completely from scratch, no ImageNet-pretrained weights for encoder...')
        else:
            weights_dir = 'logs/weights/pretrained_resnet18'    # in case no weights folder provided
            for m_name in self.opt.models_to_load:
                if self.opt.weights_dir != '':
                    weights_dir = self.opt.weights_dir
                    self.models[m_name].load_weights(os.path.join(weights_dir, m_name + '.h5'))
                else:
                    if 'enc' in m_name:
                        print('\tloading pretrained weights for encoders')
                        self.models[m_name].load_weights(os.path.join(weights_dir, m_name + '.h5'))
            print('weights loaded from ', weights_dir)

    def unfreeze_partial_models(self):
        """Only unfreeze part of the model layers"""
        unfrz_layers = {
            'depth_enc': self.opt.num_unf_de,
            'depth_dec': self.opt.num_unf_dd,
            'pose_enc': self.opt.num_unf_pe,
            'pose_dec': self.opt.num_unf_pd
        }
        for m_name, model in self.models.items():
            num_unfrozen = unfrz_layers[m_name]
            if num_unfrozen is not None:
                unfreeze_models(model, model_name=m_name, num_unfrozen=num_unfrozen)
            # model.summary()

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
        abs_diff = tf.math.abs(proj_image - tgt_image)
        ssim_diff = SSIM(proj_image, tgt_image)
        l1_loss = tf.reduce_mean(abs_diff, axis=3, keepdims=True)
        ssim_loss = tf.reduce_mean(ssim_diff, axis=3, keepdims=True)
        loss = self.opt.ssim_ratio * ssim_loss + (1 - self.opt.ssim_ratio) * l1_loss
        return loss

    def reset_losses(self):
        self.losses = {'pixel_loss': 0, 'smooth_loss': 0}

    def compute_losses(self, input_imgs, outputs):
        source_scale = 0
        tgt_image = input_imgs[('color', 0, source_scale)]
        self.reset_losses()
        total_loss = 0.

        for scale in range(self.opt.num_scales):
            # -------------------
            # 1. Reprojection / warping loss
            # -------------------
            reproject_losses = []
            for i, f_i in enumerate(self.opt.frame_idx[1:]):
                # sampler_mask = outputs[('sampler_mask', f_i, scale)]
                proj_image = outputs[('color', f_i, scale)]
                assert proj_image.shape[2] == tgt_image.shape[2] == input_imgs[('color',0,0)].shape[2]
                reproject_losses.append(self.compute_reproject_loss(proj_image, tgt_image))

            reproject_losses = tf.concat(reproject_losses, axis=3)  # B,H,W,2

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
                    proj_image = input_imgs[('color', f_i, source_scale)]
                    identity_reprojection_losses.append(self.compute_reproject_loss(proj_image, tgt_image))

                identity_reprojection_losses = tf.concat(identity_reprojection_losses, axis=3)  # B,H,W,2

                # if use average reprojection loss
                if self.avg_reprojection:
                    identity_reprojection_loss = tf.math.reduce_mean(identity_reprojection_losses, axis=3, keepdims=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

                # add random numbers to break ties
                identity_reprojection_loss += (tf.random.normal(identity_reprojection_loss.shape)
                                               * tf.constant(1e-5, dtype=tf.float32))
                combined = tf.concat([identity_reprojection_loss, reproject_loss], axis=3)  # B,H,W,4
                outputs[('automask', 0)] = tf.expand_dims(
                        tf.cast(tf.math.argmin(combined, axis=3) > 1, tf.float32) * 255, -1)

            # -------------------
            # 3. Final reprojection loss -> as pixel loss
            # -------------------
            if combined.shape[-1] != 1:
                to_optimise = tf.reduce_min(combined, axis=3)
            else:
                to_optimise = combined
            reprojection_loss = tf.reduce_mean(to_optimise)
            self.losses['pixel_loss'] += reprojection_loss

            # -------------------
            # 4. Smoothness loss: Gradient Loss based on image pixels
            # -------------------
            disp_s = outputs[('disp', 0, scale)]
            smooth_loss_raw = self.get_smooth_loss(disp_s, tgt_image)
            smooth_loss = self.opt.smoothness_ratio * smooth_loss_raw / (2 ** scale)
            self.losses['smooth_loss'] += smooth_loss   # for summary
            # ------------------
            # 5. accumulate pixel and smooth losses scale-wise
            # ------------------
            total_loss_tmp = reprojection_loss + smooth_loss
            total_loss += total_loss_tmp

        self.losses['pixel_loss'] /= self.opt.num_scales
        self.losses['smooth_loss'] /= self.opt.num_scales
        total_loss /= self.opt.num_scales

        #  ------ New losses below ---------
        if self.opt.add_rot_loss:
            rot_loss = outputs[('rot_loss',)] * self.opt.rot_loss_w
            self.losses['rot_loss'] = rot_loss
            total_loss += self.losses['rot_loss']

        self.losses['loss/total'] = total_loss

        for k, v in self.losses.items():
            print(k, v, ' | ')

        if self.opt.debug_mode and self.opt.show_image_debug:
            colormapped_normal = colorize(outputs[('disp', 0, 0)], cmap='plasma', expand_dim=True)
            arrange_display_images([colormapped_normal[0]])

        return self.losses

    def generate_images_pred(self, input_imgs, input_Ks, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        image_shape = input_imgs[('color', 0, 0)].shape
        K = input_Ks[('K', self.opt.src_scale)]
        K_inv = input_Ks[('inv_K', self.opt.src_scale)]
        sampler_padding = 'zeros' if self.opt.use_sampler_mask else 'border'
        for scale in range(self.opt.num_scales):
            disp = outputs[('disp', 0, scale)]
            # disp = tf.image.resize(disp_tf, self.image_size)
            _, depth = self.disp_to_depth(disp)
            if scale == 0:
                outputs[('depth', 0, 0)] = depth

            # -----------------------------
            for i, f_i in enumerate(self.opt.frame_idx[1:]):
                T = outputs[('cam_T_cam', f_i, 0)][0]
                # cam_points = BackProjDepth(image_shape, 0).run_func(depth, K_inv)
                # pix_coords = Project3D(image_shape, 0).run_func(cam_points, K, T)
                cam_points = back_proj_depth(depth, K_inv, image_shape, 0)
                pix_coords = project_3d(cam_points, K, T, image_shape, 0)

                outputs[('sample', f_i, scale)] = pix_coords
                input_tf = input_imgs[('color', f_i, self.opt.src_scale)]
                res = bilinear_sampler(input_tf, pix_coords, padding=sampler_padding)
                outputs[('color', f_i, scale)] = res
                if self.opt.use_sampler_mask:
                    sampler_mask = tf.cast(res * 255 > 1e-3, tf.float32)
                else:
                    sampler_mask = None
                outputs[('sampler_mask', f_i, scale)] = sampler_mask
        if self.opt.debug_mode and self.opt.show_image_debug:
            show_images(input_imgs, outputs)

    def predict_poses(self, input_imgs, outputs):
        """Use pose enc-dec to calculate camera's angles and translations"""
        frames_for_pose = {f_i: input_imgs[('color_aug', f_i, 0)] for f_i in self.opt.frame_idx}

        if self.opt.random_crop:
            frames_for_pose = {f_i: tf.image.resize(frame, (self.opt.height, self.opt.width))
                               for f_i, frame in frames_for_pose.items()}

        rot_loss = 0.
        for f_i in self.opt.frame_idx[1:]:
            # To maintain ordering we always pass frames in temporal order
            if f_i < 0:
                pose_inputs = [frames_for_pose[f_i], frames_for_pose[0]]
            else:
                pose_inputs = [frames_for_pose[0], frames_for_pose[f_i]]

            pose_features = self.models["pose_enc"](tf.concat(pose_inputs, axis=3), training=True,
                                                    unfreeze_num=self.opt.num_unf_pe)
            pred_pose_raw = self.models["pose_dec"](pose_features, training=True)
            axisangle = pred_pose_raw['angles']
            translation = pred_pose_raw['translations']
            # Invert the matrix if the frame id is negative
            invert = f_i < 0
            M = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=invert)
            outputs[('cam_T_cam', f_i, 0)] = [M]

            if self.opt.add_rot_loss:
                # same procedure, but
                # - swap the frames,
                # - not invert the transformation matrix
                pose_features = self.models["pose_enc"](tf.concat(pose_inputs[::-1], axis=3), training=True,
                                                        unfreeze_num=self.opt.num_unf_pe)
                pred_pose_raw = self.models["pose_dec"](pose_features, training=True)
                axisangle = pred_pose_raw['angles']
                translation = pred_pose_raw['translations']
                M_inv = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=not invert)
                outputs[('cam_T_cam', f_i, 0)].append(M_inv)
                rot_loss += rotation_consistency_loss(outputs[('cam_T_cam', f_i, 0)])

        if self.opt.add_rot_loss:
            outputs[('rot_loss',)] = rot_loss

        return outputs

    def process_batch(self, input_imgs, input_Ks):
        """The whoel pipeline implemented in minibatch (pairwise images)
        1. Use Depth enc-dec, to predict disparity map in mutli-scales
        2. Use Pose enc-dec, to predict poses
        3. Use products from 1.2. to generate image (reprojection) predictions
        4. Compute Losses from 3.
        """
        tgt_image_aug = input_imgs[('color_aug', 0, 0)]
        if self.opt.random_crop:
            tgt_image_aug = tf.image.resize(input_imgs[('color_aug', 0, 0)],
                                            (self.opt.height, self.opt.width))
        # Depth Encoder
        feature_raw = self.models['depth_enc'](tgt_image_aug, self.opt.train_depth,
                                               unfreeze_num=self.opt.num_unf_de)
        # Depth Decoder
        pred_disp = self.models['depth_dec'](feature_raw, self.opt.train_depth)
        outputs = {}
        for s in range(self.opt.num_scales):
            # Collect raw disp prediction at each scale
            disp_raw = pred_disp["output_%d" % s]
            disp_raw = tf.image.resize(disp_raw, input_imgs[('color', 0, 0)].shape[1:3])
            outputs[('disp', 0, s)] = disp_raw
            # Collect depth at respective scale, but resized to source_scale
            # disp_src_size = tf.image.resize(disp_raw, (self.opt.height, self.opt.width))
            # _, outputs[('depth', 0, s)] = self.disp_to_depth(disp_src_size)
        # -------------
        # 2. Pose
        # -------------
        outputs.update(self.predict_poses(input_imgs, outputs))

        # -------------
        # 3. Generate Reprojection from 1, 2
        # -------------
        self.generate_images_pred(input_imgs, input_Ks, outputs)

        return outputs

    def run_epoch(self, epoch):
        for _ in tqdm(range(self.train_loader.steps_per_epoch),
                      desc='Epoch%d/%d' % (epoch + 1, self.opt.num_epochs)):
            # data preparation
            batch = self.train_iter.get_next()
            input_imgs, input_Ks = self.batch_processor.prepare_batch(batch, is_train=True,
                                                                      random_crop=self.opt.random_crop)
            # training
            trainable_weights = self.get_trainable_weights()
            grads, losses = self.grad(input_imgs, input_Ks,
                                      trainable_weights, global_step=self.global_step)
            self.optimizer.apply_gradients(zip(grads, trainable_weights))
            self.train_loss_tmp.append(losses['loss/total'])

            if self.is_time_to('log', self.global_step):
                mean_loss = np.mean(self.train_loss_tmp)
                self.train_loss_tmp = []
                print("loss: ", mean_loss)

            if self.is_time_to('validate', self.global_step):
                print("validate miniset... random_crop?", self.opt.random_crop)
                self.validate_miniset(save_models=True)
                if self.opt.random_crop:
                    print("validate miniset... WITHOUT random_crop")
                    self.validate_miniset(save_models=False)

            self.global_step += 1
            # if self.is_time_to('special_pass', self.global_step):
            #     self.save_models()

    def start_training(self):
        """Custom training loop
        - use @tf.function for self.grad() and self.dataloader.prepare_batch()
            allow larger batch_size GPU utilization.
        """
        for epoch in range(self.opt.num_epochs):
            self.optimizer.lr = self.lr_fn(epoch+1)     # start from 1, learning rate 15:1e-4; >16:1e-5
            print("\tlearning rate - epoch %d: " % epoch, self.optimizer.get_config()['learning_rate'])

            self.run_epoch(epoch)

            print('-> Validating after epoch %d...' % epoch)
            val_losses = self.start_validating(self.global_step, self.val_iter,
                                               num_run=self.val_loader.steps_per_epoch)
            self.save_models(val_losses, del_prev=False)

            # Set new weights_folder for next epoch
            self.opt.save_model_path = self.opt.save_model_path.replace('\\', '/')
            save_root = os.path.join(self.opt.save_model_path.rsplit('/', 1)[0])
            current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            self.opt.save_model_path = os.path.join(save_root, current_time).replace('\\', '/')
            print('-> New weights will be saved in {}...'.format(self.opt.save_model_path))
            if not os.path.isdir(self.opt.save_model_path):
                os.makedirs(self.opt.save_model_path)

    @tf.function    # turn off to debug, e.g. with plt
    def grad(self, input_imgs, input_Ks, trainables, global_step):
        with tf.GradientTape() as tape:
            outputs = self.process_batch(input_imgs, input_Ks)
            losses = self.compute_losses(input_imgs, outputs)
            total_loss = losses['loss/total']
            grads = tape.gradient(total_loss, trainables)
            if self.is_time_to('log', global_step):
                print("colleting data in step ", global_step)
                self.collect_summary('train', losses, input_imgs, outputs, global_step)
        return grads, losses

    def get_trainable_weights(self):
        # ---------------
        # Fine-tune on new dataset
        # ---------------
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
            print("Please specify a model to train")
            return

        if self.opt.recording and not self.opt.debug_mode:
            train_log_dir = os.path.join(self.opt.record_summary_path, self.opt.model_name, 'train')
            self.summary_writer['train'] = tf.summary.create_file_writer(train_log_dir)
            val_log__dir = os.path.join(self.opt.record_summary_path, self.opt.model_name, 'val')
            self.summary_writer['val'] = tf.summary.create_file_writer(val_log__dir)
            print('\tSummary will be stored in %s ' % train_log_dir)

        print("->Start training...")
        self.start_training()

    @tf.function
    def compute_batch_losses(self, input_imgs, input_Ks):
        """@tf.function enables graph computation, allowing larger batch size
        """
        outputs = self.process_batch(input_imgs, input_Ks)
        losses = self.compute_losses(input_imgs, outputs)
        return outputs, losses

    def start_validating(self, global_step, val_iterator, num_run=100, random_crop=None):
        """run mini-set to see if should store current-best weights"""
        losses_all = defaultdict(list)
        if random_crop is None:
            random_crop = self.opt.random_crop

        for i in range(num_run):
            batch = val_iterator.get_next()
            input_imgs, input_Ks = self.batch_processor.prepare_batch(batch, is_train=False,
                                                                      random_crop=random_crop)
            outputs, losses = self.compute_batch_losses(input_imgs, input_Ks)
            # if ('depth_gt', 0, 0) not in input_imgs.keys():
            #     print(input_imgs.keys())
            #     quit()
            if ('depth_gt', 0, 0) in input_imgs.keys():
                losses.update(
                    self.compute_depth_losses(input_imgs, outputs))
            for metric, loss in losses.items():
                losses_all[metric].append(loss)
        # mean of mini val-set
        for metric, loss in losses_all.items():
            med_range = [int(num_run*0.1), int(num_run*0.9)]
            losses_all[metric] = tf.reduce_mean(
                tf.sort(loss)[med_range[0]: med_range[1]]
            )
        # ----------
        # log val losses and delete
        # ----------
        if self.opt.recording:
            print('-> Writing loss/metrics...')
            # only record last batch
            self.collect_summary('val', losses, input_imgs, outputs, global_step=global_step)

        print('-> Validating losses by depth ground-truth: ')
        val_loss_str = ''
        val_loss_min_str = ''
        for k in list(losses_all):
            if k in self.depth_metric_names:
                val_loss_str = ''.join([val_loss_str, '{}: {:.4f} | '.format(k, losses_all[k])])
                if k in self.val_losses_min:
                    val_loss_min_str = ''.join([val_loss_min_str, '{}: {:.4f} | '.format(k, self.val_losses_min[k])])
        val_loss_str = ''.join([val_loss_str, 'val loss: {:.4f}'.format(losses_all['loss/total'])])
        val_loss_min_str = ''.join([val_loss_min_str, 'val loss: {:.4f}'.format(self.val_losses_min['loss/total'])])
        tf.print('\t current val loss: ', val_loss_str, output_stream=sys.stdout)
        tf.print('\t previous min loss:', val_loss_min_str, output_stream=sys.stdout)
        return losses_all

    def validate_miniset(self, save_models):
        random_crop = True
        if not self.opt.random_crop or (self.opt.random_crop and not save_models):
            random_crop = False
        print("save_models? {} | random_crop? {} | really? {} ".format(save_models, self.opt.random_crop, random_crop))
        val_losses = self.start_validating(self.global_step, self.mini_val_iter, random_crop=random_crop)

        # save models if needed
        if save_models:
            is_lowest, self.val_losses_min = is_val_loss_lowest(val_losses, self.val_losses_min,
                                                                self.depth_metric_names, self.has_depth_gt)
            if is_lowest:
                self.save_models(del_prev=True)

        del val_losses  # delete tmp

    def save_models(self, val_losses=None, del_prev=True, not_saved=()):
        """save models in 1) during epoch val_loss hits new low, 2) after one epoch"""
        # ----------------------
        # To save all weights, all layers of the encoder must be turned to trainable
        # ----------------------
        def unfreeze_all_to_save_models():
            for m_name, model in self.models.items():
                if 'enc' in m_name:
                    model.trainable = True

        unfreeze_all_to_save_models()
        # - delete previous weights
        if del_prev:
            del_files(self.opt.save_model_path)

        # - save new weightspy
        if val_losses is None:
            val_losses = self.val_losses_min

        # has_depth = not isinstance(val_losses['da/a1'], (int, list))    # when not initialized, da/a1 == 10 or []
        weights_name = '_'.join(['weights', str(val_losses['loss/total'].numpy())[2:5]])
        if not self.opt.disable_gt and self.has_depth_gt:
            weights_name = '_'.join([weights_name, str(val_losses['da/a1'].numpy())[2:4]])

        print('-> Saving weights with new low loss:\n', self.val_losses_min)
        weights_path = os.path.join(self.opt.save_model_path, weights_name)
        if not os.path.isdir(weights_path) and not self.opt.debug_mode:
            os.makedirs(weights_path)

        for m_name, model in self.models.items():
            if m_name not in not_saved:
                m_path = os.path.join(weights_path, m_name + '.h5')
                tf.print("saving {} to:".format(m_name), m_path, output_stream=sys.stdout)
                model.save_weights(m_path)

        # -------------------
        # Restore unfreeze options for training
        # -------------------
        self.unfreeze_partial_models()

    def collect_summary(self, mode, losses, input_imgs, outputs, global_step):
        """collect summary for train / validation"""
        writer = self.summary_writer[mode]
        num = min(2, input_imgs[('color', 0, 0)].shape[0])
        with writer.as_default():
            for loss_name in list(losses):
                loss_val = losses[loss_name]
                tf.summary.scalar(loss_name, loss_val, step=global_step)

            # images
            tf.summary.image('tgt_image', input_imgs[('color', 0, 0)][:num], step=global_step)
            tf.summary.image('scale0_disp_color',
                             colorize(outputs[('disp', 0, 0)][:num], cmap='plasma', expand_dim=True), step=global_step)

            for f_i in self.opt.frame_idx[1:]:
                tf.summary.image('scale0_proj_{}'.format(f_i),
                                 outputs[('color', f_i, 0)][:num], step=global_step)
                # tf.summary.image('scale0_proj_error_{}'.format(f_i),
                #                  outputs[('proj_error', f_i, 0)][:2], step=global_step)

            if self.opt.do_automasking:
                tf.summary.image('scale0_automask_image',
                                 outputs[('automask', 0)][:num], step=global_step)

        writer.flush()

    def compute_depth_losses(self, input_imgs, outputs):
        """Compute depth metrics, to allow monitoring during training
        -> Only for KITTI-RawData, where velodyne depth map is valid !!!

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        def valid_width_range(range_select, cropped_shape, full_shape):
            """ get a valid range in horizontal axis
            range_select: (44, 1197)
                it's used when depth_gt is in full shape, but if cropped, this might shrink further
            cropped_shape: (B, H, W, C)
                shape after cropping(no resizing, so height is maintained)
            full_shape: (375, 1242), raw data shape for KITTI
            """
            offset_crop = (full_shape[1] - cropped_shape[1]) // 2
            val_range_crop = (0 + offset_crop, full_shape[1] - offset_crop)
            val_range = (max(range_select[0], val_range_crop[0]),
                         min(range_select[1], val_range_crop[1]))
            return val_range

        depth_gt = input_imgs[('depth_gt', 0, 0)]
        gt_shape = depth_gt.shape[1:3]
        assert 1.3 < gt_shape[1]/gt_shape[0] < 3.5, \
            "judge from aspect ratio, deoth_gt shape might be wrong: {}".format(gt_shape)
        mask = depth_gt > 0
        depth_pred = outputs[('depth', 0, 0)]   # to be resized to (375, 1242)
        depth_pred = tf.clip_by_value(
            tf.image.resize(depth_pred, gt_shape),
            1e-3, 80
        )
        assert depth_pred.shape[1] == 375, \
            'shape: {}, should be {}'.format(depth_pred.shape, depth_gt.shape)
        if self.opt.show_image_debug:
            arrange_display_images(inps_col1=(input_imgs[('color',0,0)],
                                              depth_pred[0], depth_gt[0]))
        # -------------------------
        # garg/eigen crop
        # -> if random cropped, it will be further cropped
        # -------------------------
        crop_mask = np.zeros(shape=mask.shape, dtype=np.bool)
        full_shape = KITTIOdom.full_res_shape   # H,W
        range_w = [44, 1197]
        range_h = [153, 371]
        val_range_w = valid_width_range(range_w, gt_shape, full_shape)
        crop_mask[:, range_h[0]: range_h[1], val_range_w[0]:val_range_w[1], :] = True
        mask = mask.numpy() * crop_mask

        # -------------------------
        # calculate valid region in gt, and median-scaling the depth prediction
        # -------------------------
        depth_gt_masked = tf.boolean_mask(depth_gt, mask)
        depth_pred_masked = tf.boolean_mask(depth_pred, mask)
        depth_pred_med = depth_pred_masked * np.median(depth_gt_masked) / np.median(depth_pred_masked)
        depth_pred_final = tf.clip_by_value(depth_pred_med, 1e-3, 80)

        losses = compute_depth_errors(depth_gt_masked, depth_pred_final)

        for metric in losses:
            if metric not in self.depth_metric_names:
                raise ValueError('only metrics {} are supported, got {}'. format(self.depth_metric_names, metric))

        return losses

    def disp_to_depth(self, disp):
        """Convert network's sigmoid output into depth prediction
        The formula for this conversion is given in the 'additional considerations'
        section of the paper.
        """
        min_disp = 1 / self.opt.max_depth
        max_disp = 1 / self.opt.min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1 / scaled_disp
        return scaled_disp, depth

    def is_time_to(self, event, global_step):
        if self.opt.debug_mode:
            return False

        is_time = False
        events = {0: 'log', 1: 'validate', 2: 'special_pass'}
        if event not in events.values():
            raise NotImplementedError

        if event == events[0]:
            early_phase = global_step % self.opt.record_freq == 0 and global_step < 1000
            late_phase = global_step % (self.opt.record_freq*2) == 0
            if early_phase or late_phase:
                is_time = True
        elif event == events[1]:
            special_pass = global_step % 5 == 0 and self.opt.debug_mode
            normal_pass = global_step % (self.train_loader.steps_per_epoch // self.opt.val_num_per_epoch) == 0
            is_time = normal_pass or special_pass
        elif event == events[2]:
            is_time = False
        return is_time

    # def early_stopping(self, losses, patience=3):
    #     """mean val_loss_metrics are good enough to stop early"""
    #     early_stop = False
    #     for metric in self.min_errors_thresh:
    #         self.early_stopping_losses[metric].append(losses[metric])
    #         mean_error = np.mean(self.early_stopping_losses[metric][-patience:])
    #         if losses[metric] > self.min_errors_thresh[metric]:
    #             print('* early stopping ready')
    #         if mean_error > self.min_errors_thresh[metric]:
    #             return early_stop
    #
    #     if np.mean(self.train_loss_tmp) < 0.11:
    #         early_stop = True
    #
    #     return early_stop


if __name__ == '__main__':
    trainer = Trainer(options)
    trainer.train(record=False)
