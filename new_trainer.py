import tensorflow as tf
from models.depth_decoder_creater import DepthDecoder_full
from models.encoder_creater import ResNet18_new
from models.posenet_decoder_creator import PoseDecoder
from utils import process_enc_outputs, disp_to_depth, concat_pose_params
from src.trainer_helper import SSIM, projective_inverse_warp
from src.dataset_loader import build_dataset
import numpy as np
from src.DataPprocessor import DataProcessor


class Trainer:
    def __init__(self, options=None):
        self.opt = options
        self.models = {}
        self.num_scales = 4
        self.src_scale = 0
        self.height = 192
        self.width = 640
        self.frame_idx = [0, -1, 1]
        self.preprocess = True
        self.auto_mask = False
        self.ssim_ratio = 0.85
        self.smoothness_ratio = 1e-3
        self.min_depth = 0.1
        self.max_depth = 100

        self.tgt_image = None
        self.tgt_image_net = None
        self.tgt_image_aug = None

        self.avg_reprojection = False

        self.tgt_image_stack_all = []
        self.src_image_stack_all = []
        self.proj_image_stack_all = []
        self.proj_error_stack_all = []

        self.pixel_losses = 0.
        self.smooth_losses = 0.
        self.total_loss = 0.
        self.losses = {}

        self.num_epochs = 10
        self.batch_size = 4

        self.batch_processor = DataProcessor()
        lr = 1e-3   # todo: learning rate scheduler
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        self.init_app()

    def init_app(self):
        """Initiate Pose, Depth and Auto-Masking models
        self.models['depth_enc'] = ResNet18_new([2, 2, 2, 2])
        self.models['depth_dec'] = DepthDecoder_full()
        self.models['pose_enc'] = ResNet18_new([2, 2, 2, 2])
        self.models['pose_dec'] = PoseDecoder(num_ch_enc=[64, 64, 128, 256, 512])
        """
        self.models['depth_enc'] = ResNet18_new([2, 2, 2, 2])
        self.models['depth_enc'].load_weights("weights/depth_enc/")

        self.models['depth_dec'] = DepthDecoder_full()
        self.models['depth_dec'].load_weights("weights/depth_dec/")

        self.models['pose_enc'] = ResNet18_new([2, 2, 2, 2])
        self.models['pose_enc'].load_weights("weights/pose_enc/")

        shapes = [(1, 96, 320, 64), (1, 48, 160, 64), (1, 24, 80, 128), (1, 12, 40, 256), (1, 6, 20, 512)]
        inputs = [tf.random.uniform(shape=(shapes[i])) for i in range(len(shapes))]
        self.models['pose_dec'] = PoseDecoder(num_ch_enc=[64, 64, 128, 256, 512])
        _ = self.models['pose_dec'].predict(inputs)
        self.models['pose_dec'].load_weights("weights/pose_dec/")

        self.build_models(show_summary=False, check_outputs=False)

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
        loss = self.ssim_ratio * ssim_loss + (1 - self.ssim_ratio) * l1_loss

        return loss
    
    def compute_losses(self, input_imgs, outputs):
        print("compute_losses ...")
        tgt_image = input_imgs[('color', 0, 0)]

        for scale in range(self.num_scales):
            # -------------------
            # 1. Reprojection / warping loss
            # -------------------
            reproject_losses = []
            for i, f_i in enumerate(self.frame_idx[1:]):
                proj_image = outputs[('color', f_i, scale)]
                assert proj_image.shape[2] == tgt_image.shape[2] == 640
                proj_error = tf.math.abs(proj_image - tgt_image)
                reproject_losses.append(self.compute_reproject_loss(proj_image, tgt_image))
                if i == 0:
                    proj_image_stack = proj_image
                    proj_error_stack = proj_error
                else:
                    proj_image_stack = tf.concat([proj_image_stack, proj_image], axis=3)
                    proj_error_stack = tf.concat([proj_error_stack, proj_error], axis=3)

            reproject_losses = tf.concat(reproject_losses, axis=3)

            if self.avg_reprojection:
                reproject_loss = tf.math.reduce_mean(reproject_losses, axis=1, keepdims=True)
            else:
                reproject_loss = reproject_losses

            # -------------------
            # 2. Optional: auto-masking
            # -------------------
            combined = reproject_losses     # incase no auto-masking
            if self.auto_mask:
                pred_auto_masks = []
                identity_reprojection_losses = []
                for f_i in self.frame_idx[1:]:
                    proj_image = input_imgs[('color', f_i, 0)]
                    identity_reprojection_losses.append(self.compute_reproject_loss(proj_image, tgt_image))

                identity_reprojection_losses = tf.concat(identity_reprojection_losses, axis=3)

                # if use average reprojection loss
                if self.avg_reprojection:
                    identity_reprojection_loss = tf.math.reduce_mean(identity_reprojection_losses, axis=3, keepdims=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

                # add random numbers to break ties
                identity_reprojection_loss += (tf.random.normal(identity_reprojection_loss.shape) * 1e-5)
                combined = tf.concat([identity_reprojection_loss, reproject_loss], axis=3)

                if combined.shape[1] == 1:
                    to_optimise = combined
                else:
                    to_optimise, idxs = tf.math.reduce_min(combined, dim=3)
                    if self.auto_mask:
                        outputs["identity_selection/{}".format(scale)] = (
                                idxs > identity_reprojection_loss.shape[1] - 1).float()

                pred_auto_masks.append(
                    tf.expand_dims(
                        tf.cast(tf.math.argmin(combined, axis=3) > 1, tf.float32) * 255, -1)
                )
            # -------------------
            # 3. Final reprojectioni loss -> as pixel loss
            # -------------------
            reprojection_loss = tf.reduce_mean(tf.reduce_min(combined, axis=3))
            self.pixel_losses += reprojection_loss

            # -------------------
            # 4. Smoothness loss: Gradient Loss based on image pixels
            # -------------------
            disp_s, tgt_image_s = outputs[('disp', scale)], input_imgs['color', 0, scale]
            smooth_loss = self.get_smooth_loss(disp_s, tgt_image_s)
            self.smooth_losses += smooth_loss
            smooth_loss /= (2 ** scale)

            # ------------------
            # 5. Overall Loss, accumulate scale-wise
            # ------------------
            scale_total_loss = reprojection_loss + self.smoothness_ratio * smooth_loss
            self.total_loss += scale_total_loss
            self.losses['loss/%d'%scale] = scale_total_loss

            # ------------------
            # Optional: Collect results for summary
            # ------------------
            src_image_stack_aug = tf.concat([input_imgs[('color_aug', -1, 0)],
                                             input_imgs[('color_aug', -1, 0)]], axis=3)
            self.tgt_image_stack_all.append(tgt_image)
            self.src_image_stack_all.append(src_image_stack_aug)
            self.proj_image_stack_all.append(proj_image_stack)
            self.proj_error_stack_all.append(proj_error_stack)

        self.pixel_losses /= self.num_scales
        self.smooth_losses /= self.num_scales

        self.total_loss /= self.num_scales
        self.losses['loss/total'] = self.total_loss
        return self.total_loss

    def generate_images_pred(self, input_imgs, input_K_mulscale, outputs, pred_poses):
        """Generate warped/reprojected images for a minibatch
        produced images are saved in 'outputs' dir
        """
        print("generate_images_pred...")
        for scale in range(self.num_scales):
            for f_i in self.frame_idx[1:]:
                # Calculate reprojected image.
                # When f_i==-1, since pose is current->previous, inverse order,
                # so intrinsics K also needs to be inverted before warping
                do_invert = True if f_i == -1 else False
                print("\tinput_K_mulscale: ", input_K_mulscale.shape)
                curr_proj_image = projective_inverse_warp(input_imgs[('color', f_i, self.src_scale)],
                                                          tf.squeeze(outputs[('depth', scale)], axis=3),
                                                          pred_poses[:, f_i, :],
                                                          intrinsics=input_K_mulscale[:, self.src_scale, :, :],
                                                          invert=do_invert)
                print("\tprojected image shape :", curr_proj_image.shape, " at scale %d"%scale, " f_id %d"%f_i)
                outputs[('color', f_i, scale)] = curr_proj_image
                # not in use for now
                if self.auto_mask:
                    outputs[("color_identity", f_i, scale)] = input_imgs[("color", f_i, self.src_scale)]
        return outputs

    def predict_poses(self, input_imgs):
        """Use pose enc-dec to calculate camera's angles and translations"""
        tgt_image_aug       = input_imgs[('color_aug', self.frame_idx[0], 0)]
        src_image_pre_aug   = input_imgs[('color_aug', self.frame_idx[1], 0)]
        src_image_post_aug  = input_imgs[('color_aug', self.frame_idx[2], 0)]
        print("============ predict_poses ==============")
        print(src_image_pre_aug.shape)
        print(tgt_image_aug.shape)
        # Enocder
        pose_ctp_raw = self.models['pose_enc'](
            tf.concat([src_image_pre_aug, tgt_image_aug], axis=3)
        )
        pose_ctn_raw = self.models['pose_enc'](
            tf.concat([tgt_image_aug, src_image_post_aug], axis=3)
        )
        # Decoder
        # pose_ctp = [pose_ctp_raw[-1]]
        # pose_ctn = [pose_ctn_raw[-1]]
        pred_pose_ctp_raw = self.models['pose_dec'](pose_ctp_raw, True)
        pred_pose_ctn_raw = self.models['pose_dec'](pose_ctn_raw, True)

        # Collect angles and translations
        pred_pose_ctp = concat_pose_params(pred_pose_ctp_raw, curr2prev=True, curr2next=False)
        pred_pose_ctn = concat_pose_params(pred_pose_ctn_raw, curr2prev=True, curr2next=False)
        print("\t check concated pose shape: ", pred_pose_ctn.shape)
        pred_poses = tf.concat([pred_pose_ctp, pred_pose_ctn], axis=1)
        return pred_poses


    def process_batch(self, input_imgs, input_K_mulscale):
        """The whoel pipeline implemented in minibatch (pairwise images)
        1. Use Depth enc-dec, to predict disparity map in mutli-scales
        2. Use Pose enc-dec, to predict poses
        3. Use products from 1.2. to generate image (reprojection) predictions
        4. Compute Losses from 3.
        """
        tgt_image_aug = input_imgs[('color_aug', 0, 0)]

        # Depth Encoder
        feature_raw = self.models['depth_enc'](tgt_image_aug, True)
        # Depth Decoder
        pred_disp = self.models['depth_dec'](feature_raw, True)
        outputs = {}
        for s in range(self.num_scales):
            # Collect raw disp prediction at each scale
            disp_raw = pred_disp["output_%d"%s]
            outputs[('disp', s)] = disp_raw
            # Collect depth at respective scale, but resized to source_scale
            disp_src_size = tf.image.resize(disp_raw, (self.height, self.width))
            outputs[('depth', s)], _ = disp_to_depth(disp_src_size, self.min_depth, self.max_depth)

        # -------------
        # 2. Pose
        # -------------
        pred_poses = self.predict_poses(input_imgs)

        # -------------
        # 3. Generate Reprojection from 1, 2
        # -------------
        self.generate_images_pred(input_imgs, input_K_mulscale, outputs, pred_poses)

        # -------------
        # 4. Compute Losses from 3.
        # - Conducted in training procecss, here just for better understanding
        # self.compute_losses(input_imgs, outputs)
        # -------------
        return outputs

    def run_epoch(self, dataset):
        """Training loop
        Tf.keras 与 pytorch 相似，只是将 loss.backward() 换成 tf.GradientTape()
        - loss function 需要自己定义, 返回 total loss即可, 就可通过 GradientTape 跟踪
        - 然后 compile(loss=compute_loss)
        """
        for i, batch in enumerate(dataset):
            input_imgs, input_K_mulscale = self.batch_processor.prepare_batch(batch[0], batch[1])
            trainable_weights_all = []
            for m_name, model in self.models.items():
                print("%s training... "%m_name)
                trainable_weights_all.extend(model.trainable_weights)

            # Method-1, calculate in steps
            total_loss, grads = self.grad(input_imgs, input_K_mulscale, trainable_weights_all)
            self.optimizer.apply_gradients(zip(grads, trainable_weights_all))
            print(" **** Batch %d****" % i)
            # ----- Method-2, call .minimize() directly -----
            # outputs = self.process_batch(input_imgs, input_K_mulscale)
            # loss_fn = lambda: self.compute_losses(input_imgs, outputs)
            # var_list_fn = lambda: trainable_weights
            # self.optimizer.minimize(loss_fn, var_list_fn)

    def grad(self, input_imgs, input_K_mulscale, trainables):
        with tf.GradientTape() as tape:
            outputs = self.process_batch(input_imgs, input_K_mulscale)
            total_loss = self.compute_losses(input_imgs, outputs)
            grads = tape.gradient(total_loss, trainables)
        return total_loss, grads

    def train(self):
        dataset = build_dataset(self.num_epochs, self.batch_size)
        for epoch in range(self.num_epochs):
            self.run_epoch(dataset)
            print("------ Epoch %d / %d ------" % (epoch, self.num_epochs))

    def set_train(self):
        pass

    def set_eval(self):
        pass

    def compute_depth_losses(self):
        pass

    # for debugging
    def build_models(self, check_outputs=False, show_summary=False):
        for k, m in self.models.items():
            print(k)
            if "depth_enc" == k:
                inputs = tf.random.uniform(shape=(1, 192, 640, 3))
                outputs = m(inputs)
            elif "depth_dec" == k:
                shapes = [(1, 96, 320, 64), (1, 48, 160, 64), (1, 24, 80, 128), (1, 12, 40, 256), (1, 6, 20, 512)]
                inputs = [tf.random.uniform(shape=(shapes[i])) for i in range(len(shapes))]
                outputs = m(inputs)
            elif "pose_enc" == k:
                shape = (1, 192, 640, 3)
                inputs = tf.concat([tf.random.uniform(shape=shape),
                                    tf.random.uniform(shape=shape)], axis=3)
                outputs = m(inputs)
            elif "pose_dec" == k:
                shapes = [(1, 96, 320, 64), (1, 48, 160, 64), (1, 24, 80, 128), (1, 12, 40, 256), (1, 6, 20, 512)]
                inputs = [tf.random.uniform(shape=(shapes[i])) for i in range(len(shapes))]
                outputs = m(inputs)
            else:
                raise NotImplementedError

            if check_outputs:
                print(type(outputs))
                if isinstance(outputs, dict):
                    for k,v in outputs.items():
                        print(k,"\t",v.shape)
                else:
                    for elem in outputs:
                        print(elem.shape)

            if show_summary:
                m.summary()


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()