import tensorflow as tf
from models.depth_decoder_creater import DepthDecoder_full
from models.encoder_creater import ResNet18_new
from models.posenet_decoder_creator import PoseDecoder_exp
from models.motion_filed_net import IntrinsicsHead
from utils import disp_to_depth, del_files, make_hom_intrinsics
from src.trainer_helper import *
from datasets.data_loader_kitti import DataLoader
from datasets.dataset_kitti import KITTIRaw, KITTIOdom
from src.DataPprocessor import DataProcessor

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
        self.models = {}
        self.num_frames_per_batch = len(self.opt.frame_idx)
        self.train_loader: DataLoader = None
        self.mini_val_loader: DataLoader = None
        self.shape_scale0 = [self.opt.batch_size, self.opt.height, self.opt.width, 3]

        self.tgt_image = None
        self.tgt_image_net = None
        self.tgt_image_aug = None
        self.avg_reprojection = False

        self.train_loss_tmp = []
        self.losses = {}
        self.early_stopping_losses = defaultdict(list)
        self.global_step = tf.constant(0, dtype=tf.int64)
        self.depth_metric_names = []
        self.min_errors_thresh = {}
        self.val_losses_min = defaultdict(lambda: 10)  # random number as initiation
        self.mini_val_iter = None
        self.train_iter = None
        self.batch_processor: DataProcessor = None
        self.lr_fn = None
        self.optimizer = None

        self.summary_writer = {}
        self.weights_idx = 0  # for saved weights
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
        # Choose dataset
        dataset_choices = {
            'kitti_raw': KITTIRaw,
            'kitti_odom': KITTIOdom
        }
        split_folder = os.path.join('splits', self.opt.split)
        train_file = 'train_files.txt'
        val_file = 'val_files.txt'

        # Train dataset & loader
        train_dataset = dataset_choices[self.opt.dataset](
            split_folder, train_file, data_path=self.opt.data_path)
        self.train_loader = DataLoader(train_dataset, self.opt.num_epochs,
                                       self.opt.batch_size, self.opt.frame_idx)
        self.train_iter = self.train_loader.build_train_dataset()

        # Validation dataset & loader
        # - mini-val during the epoch
        mini_val_dataset = dataset_choices[self.opt.dataset](
            split_folder, val_file, data_path=self.opt.data_path)
        self.mini_val_loader = DataLoader(mini_val_dataset, num_epoch=self.opt.num_epochs,
                                          batch_size=2, frame_idx=self.opt.frame_idx)
        self.mini_val_iter = self.mini_val_loader.build_val_dataset(include_depth=self.train_loader.has_depth,
                                                                    buffer_size=300)
        # - Val after one epoch
        val_dataset = dataset_choices[self.opt.dataset](
            split_folder, val_file, data_path=self.opt.data_path)
        self.val_loader = DataLoader(val_dataset, num_epoch=self.opt.num_epochs,
                                     batch_size=4, frame_idx=self.opt.frame_idx)
        self.val_iter = self.val_loader.build_val_dataset(include_depth=self.train_loader.has_depth,
                                                          buffer_size=self.opt.batch_size)
        if self.train_loader.has_depth:
            # Val metrics only when depth_gt available
            self.min_errors_thresh = {'de/abs_rel': 0.14, 'de/sq_rel': 1.,
                                      'de/rms': 4.8, 'da/a1': 0.83}
            self.depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms",
                                       "da/a1", "da/a2", "da/a3"]

        # Batch data preprocessor
        K = None if self.opt.learn_intrinsics else train_dataset.K
        self.batch_processor = DataProcessor(frame_idx=self.opt.frame_idx, intrinsics=K)

        # Init models
        self.models['depth_enc'] = ResNet18_new(norm_inp=self.opt.norm_input)
        self.models['depth_dec'] = DepthDecoder_full()
        self.models['pose_enc'] = ResNet18_new(norm_inp=self.opt.norm_input)
        self.models['pose_dec'] = PoseDecoder_exp(num_frames_to_predict_for=self.opt.pose_num)
        self.models['intrinsics_head'] = IntrinsicsHead(self.shape_scale0[1:3])

        print('* concat depth_pred?', self.opt.concat_depth_pred)
        build_models(self.models, rgb_cat_depth=self.opt.concat_depth_pred)
        self.load_models()

        # Set optimizer
        boundaries = [self.opt.lr_step_size, self.opt.lr_step_size * 2]  # [15, 30]
        values = [self.opt.learning_rate * scale for scale in [1, 0.1, 0.01]]  # [1e-4, 1e-5, 1e-6]
        self.lr_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_fn(0))

        # Init inverse-warping helpers
        self.back_project_dict[self.opt.src_scale] = BackProjDepth(self.shape_scale0, self.opt.src_scale)
        self.project_3d_dict[self.opt.src_scale] = Project3D(self.shape_scale0, self.opt.src_scale)

    def load_models(self):

        def get_weights_name():
            weights_name = m_name
            if m_name == 'pose_enc' and self.opt.concat_depth_pred:
                weights_name = m_name + '_concat'
            elif m_name == 'pose_dec' and self.opt.pose_num == 1:
                weights_name = m_name + '_one_out'
            return weights_name

        if self.opt.from_scratch:
            print('\ttraining completely from scratch, no ImageNet-pretrained weights for encoder...')
        else:
            weights_dir = 'logs/weights/pretrained_resnet18'  # in case no weights folder provided
            for m_name in self.opt.models_to_load:
                print('\tloading', m_name)
                if self.opt.weights_dir != '':
                    weights_dir = self.opt.weights_dir
                    weights_name = get_weights_name()
                    print('weights_name:', weights_name)
                    self.models[m_name].load_weights(os.path.join(weights_dir, weights_name + '.h5'))
                else:
                    if 'enc' in m_name:
                        weights_name = get_weights_name()
                        print('\t\tusing pretrained encoders: ', weights_name)
                        self.models[m_name].load_weights(os.path.join(weights_dir, weights_name + '.h5'))
            print('weights loaded from ', weights_dir)

    # ----- For process_batch() -----
    def get_smooth_loss(self, disp, img):
        norm_disp = disp / (tf.reduce_mean(disp, [1, 2], keepdims=True) + 1e-7)
        # grad_disp_x = tf.math.abs(norm_disp[:, :-1, :, :] - norm_disp[:, 1:, :, :])
        # grad_disp_y = tf.math.abs(norm_disp[:, :, :-1, :] - norm_disp[:, :, 1:, :])
        grad_disp_x = tf.math.abs(norm_disp - tf.roll(norm_disp, shift=1, axis=1))
        grad_disp_y = tf.math.abs(norm_disp - tf.roll(norm_disp, shift=1, axis=2))

        # grad_img_x = tf.math.abs(img[:, :-1, :, :] - img[:, 1:, :, :])
        # grad_img_y = tf.math.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
        grad_img_x = tf.math.abs(img - tf.roll(img, shift=1, axis=1))
        grad_img_y = tf.math.abs(img - tf.roll(img, shift=1, axis=2))

        weight_x = tf.math.exp(-tf.reduce_mean(grad_img_x, 3, keepdims=True))
        weight_y = tf.math.exp(-tf.reduce_mean(grad_img_y, 3, keepdims=True))

        smoothness_x = grad_disp_x * weight_x
        smoothness_y = grad_disp_y * weight_y

        return tf.reduce_mean(smoothness_x) + tf.reduce_mean(smoothness_y)

    def compute_reproject_loss(self, proj_data, tgt_data, sampler_mask=None):
        abs_diff = tf.math.abs(proj_data - tgt_data)
        ssim_diff = SSIM(proj_data, tgt_data)
        # print('ssim before:', tf.reduce_mean(ssim_diff))
        if sampler_mask is not None:
            sampler_mask = tf.concat([sampler_mask] * tgt_data.shape[-1], axis=-1)  # B,H,W,C
            abs_diff = abs_diff * sampler_mask
            ssim_diff = ssim_diff * sampler_mask
            # print('ssim after:', tf.reduce_mean(ssim_diff))

        l1_loss = tf.reduce_mean(abs_diff, axis=3, keepdims=True)
        ssim_loss = tf.reduce_mean(ssim_diff, axis=3, keepdims=True)

        loss = self.opt.ssim_ratio * ssim_loss + (1 - self.opt.ssim_ratio) * l1_loss
        return loss

    def reset_losses(self):
        self.losses = {'pixel_loss': 0, 'smooth_loss': 0}

    def compute_losses(self, inputs, outputs):
        scale0 = 0
        tgt_data = inputs[('color', 0, scale0)]
        if self.opt.use_RGBD:
            tgt_data = tf.concat([
                tgt_data, outputs[('disp', 0, scale0)]
            ], axis=-1)
        self.reset_losses()
        total_loss = 0.

        if self.opt.use_cycle_consistency:
            idx_map = {-1: 1, 1: 0}     # for frame=-1, use the second(reversed) data.
            cycle_loss = 0.
            for f_i in self.opt.frame_idx[1:]:
                mask = outputs[('occlu_aware_mask', f_i, 0)]
                idx = idx_map[f_i]
                data_warped = outputs[('warped_multi_s', f_i, 0)][idx][..., :3]  # doesn't use depth for now
                data_orig = inputs[('color', f_i, 0)]
                if self.opt.use_RGBD:
                    data_warped = outputs[('warped_multi_s', f_i, 0)][idx]
                    data_orig = tf.concat([
                        data_orig, outputs[('disp', f_i, 0)]
                    ])
                mask = tf.concat([mask] * data_orig.shape[-1], axis=-1)
                cycle_loss += tf.reduce_mean(tf.boolean_mask(
                    tf.math.abs(data_warped - data_orig), mask)) * self.opt.cycle_loss_weight

            self.losses['cycle_loss'] = cycle_loss / len(self.opt.frame_idx[1:])
            print("cycle_loss:", self.losses['cycle_loss'])

        for scale in range(self.opt.num_scales):
            # -------------------
            # 1. Reprojection / warping loss
            # -------------------
            reproject_losses = []
            for i, f_i in enumerate(self.opt.frame_idx[1:]):
                sampler_mask = outputs[('sampler_mask', f_i, 0)]
                proj_data = outputs[('warped_multi_s', f_i, scale)][0]  # 0: src->tgt; 1(optional): tgt->src
                assert proj_data.shape[2] == tgt_data.shape[2] == self.opt.width
                reproject_losses.append(
                    self.compute_reproject_loss(proj_data, tgt_data, sampler_mask)
                )
                # if scale == 0:
                    # proj_error = tf.math.abs(proj_data - tgt_data)    # to collect
                    # outputs[('proj_error', f_i, scale)] = proj_error

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
                    sampler_mask = outputs[('sampler_mask', f_i, 0)]
                    src_image = inputs[('color', f_i, scale0)]
                    tgt_image = tgt_data[..., :3]
                    identity_reprojection_losses.append(
                        self.compute_reproject_loss(src_image, tgt_image, sampler_mask)
                    )
                identity_reprojection_losses = tf.concat(identity_reprojection_losses, axis=3)

                # if use average reprojection loss
                if self.avg_reprojection:
                    identity_reprojection_loss = tf.math.reduce_mean(identity_reprojection_losses, axis=3,
                                                                     keepdims=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

                # add random numbers to break ties
                identity_reprojection_loss += (tf.random.normal(identity_reprojection_loss.shape)
                                               * tf.constant(1e-5, dtype=tf.float32))
                combined = tf.concat([identity_reprojection_loss, reproject_loss], axis=3)
                # outputs[('automask', 0)] = tf.expand_dims(
                #     tf.cast(tf.math.argmin(combined, axis=3) > 1, tf.float32) * 255, -1)

            # -------------------
            # 3. Final reprojection loss -> as pixel loss
            if combined.shape[-1] != 1:
                to_optimise = tf.reduce_min(combined, axis=3)
            else:
                to_optimise = combined
            reprojection_loss = tf.reduce_mean(to_optimise) * self.opt.reproj_loss_weight
            print("reprojection loss:", reprojection_loss)
            # exit('xx')
            self.losses['pixel_loss'] += reprojection_loss

            # -------------------
            # 4. Smoothness loss: Gradient Loss based on image pixels
            disp_s = outputs[('disp', 0, scale)]
            tgt_image_s = inputs['color', 0, scale]
            smooth_loss_raw = self.get_smooth_loss(disp_s, tgt_image_s)
            smooth_loss = self.opt.smoothness_ratio * smooth_loss_raw / (2 ** scale)

            # ------------------
            # 5. Overall Loss, accumulate scale-wise
            total_loss_tmp = reprojection_loss + smooth_loss
            total_loss += total_loss_tmp

            # ------------------
            # Optional: Collect results for summary
            self.losses['smooth_loss'] += smooth_loss
            # self.losses['loss/%d' % scale] = total_loss_tmp

        self.losses['pixel_loss'] /= self.opt.num_scales
        self.losses['smooth_loss'] /= self.opt.num_scales
        total_loss /= self.opt.num_scales

        if self.opt.add_pose_loss:
            total_loss += outputs[('pose_loss',)]  # must use `tuple`, otherwise fail in @tf.function
            self.losses['pose_loss'] = outputs[('pose_loss',)]
            print("pose loss:",self.losses['pose_loss'])
        if self.opt.use_cycle_consistency:
            total_loss += self.losses['cycle_loss']

        self.losses['loss/total'] = total_loss

        if self.opt.debug_mode:
            colormapped_normal = colorize(outputs[('disp', 0, 0)], cmap='plasma')
            if colormapped_normal.shape[0] > 1:
                colormapped_normal = colormapped_normal[0]
            plt.imshow(colormapped_normal.numpy()), plt.show()

        return self.losses

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        Keys for `outputs` explained:
        - ['disp', <frame_id>, <scale>]: Tensor
            -> label and content just like `inputs`, it's default setting if not specified
            -> shape (B,H,W,C), channel `C` can be 3 for RGB or 4 for an additional depth map

        - ['cam_T_cam', <frame_id>, 0]: List:
            motion transformation for two cameras / standpoints
                -> #0: from source to target, frame -1->0 or 1->0, regardless of temporal order
                -> #1: from target to source

        - [('warped_multi_s', <frame_id>, <scale>)] : List
            -> #0: warped from source to target, like the `cam_T_cam`
            -> #1: warped from target to source
            -> Examples
                [('warped_multi_s', -1, 0)][0]: data at scale 0, warped from previous to current frame
                [('warped_multi_s', -1, 1)][1]: data at scale 1, warped from current to previous frame
                [('warped_multi_s', 1, 0)][1]:  data at scale 0, warped from current to next frame
        - [('disp_warp', frame_id, 0)]: List
            same as the above, but doesn't include multi-scale, it's a temp use only for occlusion-aware masks
        """

        for scale in range(self.opt.num_scales):
            disp_tgt_s = tf.image.resize(outputs[('disp', 0, scale)], (self.opt.height, self.opt.width))
            # _, depth_tgt_s = disp_to_depth(disp_tgt_s, self.opt.min_depth, self.opt.max_depth)
            _, depth_tgt_s = self.disp_to_depth(disp_tgt_s)

            if scale == 0:
                outputs[('depth', 0, 0)] = depth_tgt_s  # for depth_gt supervision

            for i, f_i in enumerate(self.opt.frame_idx[1:]):
                T = outputs[('cam_T_cam', f_i, 0)][0]  # 0: src->tgt, 1(optional): tgt->src

                if self.opt.use_cycle_consistency and scale == 0:
                    disp_src_s0 = outputs[('disp', f_i, 0)]
                    # _, depth_src_s0 = self.disp_to_depth(disp_src_s0)
                    warped_data, _ = self.inverse_warp(depth_tgt_s, disp_src_s0, inputs, T)
                    # difference between `depth_warp` and `warped_multi_s`: it only includes scale==0
                    outputs[('disp_warp', f_i, 0)] = [warped_data]

                src_data = inputs[('color', f_i, self.opt.src_scale)]

                if self.opt.use_RGBD:
                    disp_src = outputs[('disp', f_i, 0)]
                    src_data = tf.concat([src_data, disp_src], axis=-1)     # B,H,W,4
                data_resamp, pix_coords = self.inverse_warp(depth_tgt_s, src_data, inputs, T)

                outputs[('warped_multi_s', f_i, scale)] = [data_resamp]     # B,H,W,3or4
                # outputs[('sample', f_i, scale)] = pix_coords

                if not self.opt.mask_border:
                    outputs[('sampler_mask', f_i, 0)] = None
                elif scale == 0:
                    outputs[('sampler_mask', f_i, 0)] = self.get_sampler_mask(data_resamp)

        if self.opt.use_cycle_consistency:
            # cycle_consistency is applied only to th frame that is closer to the camera, namely frame -1, 0
            # -> -1 VS. warped -1 from 0
            # -> 0  VS. warped  0 from 1, which has been calculated
            # So now just calculate "warped 0 from -1"
            f_i = -1
            depth_src = self.disp_to_depth(outputs[('disp', f_i, 0)])[1]
            tgt_image = inputs[('color', f_i, 0)]
            T = outputs[('cam_T_cam', f_i, 0)][1]
            outputs[('warped_multi_s', f_i, 0)].append(
                self.inverse_warp(depth_src, tgt_image, inputs, T)[0]  # for reprojection
            )
            outputs.update(self.get_occlu_aware_mask(outputs, inputs))

        if self.opt.debug_mode:
            # ---
            src_data = inputs[('color', 0, 0)].numpy()[0]
            warped_n = outputs[('warped_multi_s', 1, 0)][0].numpy()[0]
            warped_p = outputs[('warped_multi_s', -1, 0)][0].numpy()[0]
            sampler_mask = outputs[('sampler_mask', 1, 0)][1]
            # ---
            disp_orig = outputs[('disp', self.opt.frame_idx[0], 0)][0]
            # disp_resamp1 = tf.expand_dims(outputs[('rgbd', self.opt.frame_idx[1], 0)][0, ..., -1], 2)
            # disp_resamp2 = tf.expand_dims(outputs[('rgbd', self.opt.frame_idx[2], 0)][0, ..., -1], 2)
            # ---
            # mask1 = tf.cast(outputs[('occlu_aware_mask', -1, 0)], dtype=tf.float32)[0]
            # mask2 = tf.cast(outputs[('occlu_aware_mask', 1, 0)], dtype=tf.float32)[0]
            print('mask perc', tf.reduce_mean(sampler_mask))
            inps = [
                warped_p,
                src_data,
                warped_n,
            ]
            num_inps = len(inps)
            fig = plt.figure(figsize=(num_inps, 1))
            for i in range(num_inps):
                print(i)
                fig.add_subplot(num_inps, 1, i + 1)
                plt.imshow(inps[i])
            plt.show()
            # show_images(self.opt.batch_size, inputs, outputs)

    def get_sampler_mask(self, data_resamp):
        """use the first channel to 2-D produce mask
        But before using it, it needs to be replicated to the correct channel.
        Args:
            data_resamp: Tensor, shape (B,H,W,C)
        Returns:
            sampler_mask: Tensor, shape (B,H,W, 1)
        """
        sampler_mask = tf.expand_dims(tf.cast(data_resamp[..., 0] * 255 > 1., tf.float32), axis=3)  # B,H,W
        return sampler_mask

    def get_occlu_aware_mask(self, outputs, input_Ks):
        """Produce occlusion-arware between 2 frames
        This will be use along with "sampler_mask", to only apply on non-border region
        Note that "minimal_projection" put constraint of occlusion between 3 frames
        """

        def delete_tmps(f_i):
            """Free memory, only `outputs[('occlu_aware_mask', f_i, 0)]` is needed """
            del outputs[('disp_warp', f_i, 0)]

        disp_tgt = outputs[('disp', 0, 0)]
        for f_i in self.opt.frame_idx[1:]:
            # inverse-warp the target_depth -> source_depth
            T = outputs[('cam_T_cam', f_i, 0)][1]
            disp_src = tf.image.resize(outputs[('disp', f_i, 0)], (self.opt.height, self.opt.width))
            _, depth_src = self.disp_to_depth(disp_src)
            outputs[('disp_warp', f_i, 0)].append(
                self.inverse_warp(depth_src, disp_tgt, input_Ks, T)[0]
            )
            # ---------------------------
            # Compare: real source VS. from-tgt-transformed source depth map
            # ---------------------------
            disp_src = outputs[('disp', f_i, 0)]
            disp_src_warped, disp_tgt_warped = outputs[('disp_warp', f_i, 0)]
            if f_i == -1:
                # Zr' <= Zl for normal order, where camera closer at tgt-frame / frame==0
                mask_1 = disp_tgt_warped >= disp_src
                mask_2 = disp_src_warped >= disp_tgt
            else:
                # vice versa, where camera is closer at src-frame / frame==1
                mask_1 = disp_tgt_warped <= disp_src
                mask_2 = disp_src_warped <= disp_tgt
            outputs[('occlu_aware_mask', f_i, 0)] = tf.logical_and(mask_1, mask_2)
            delete_tmps(f_i)
        return outputs

    def inverse_warp(self, depth, src_data, input_Ks, T):
        """inverse-warping: transfer src_data -> resamp_data
        Args:
            depth: depth of target (represented by depth & T together)
            src_data: source data to be resampled, can be image or depth map
            T: transformation `source -> target` data
            input_Ks: intrinsics pyramid dict with keys like [('K'or'inv_K, scale)]
        Returns:
            resamp_data: resampled src_data (in target position)
            pix_coords: target grid coordinates for source data
        """
        cam_points = self.back_project_dict[self.opt.src_scale].run_func(
            depth, input_Ks[("inv_K", self.opt.src_scale)]
        )
        pix_coords = self.project_3d_dict[self.opt.src_scale].run_func(
            cam_points, input_Ks[("K", self.opt.src_scale)], T
        )
        resamp_data = bilinear_sampler(src_data, pix_coords, padding=self.opt.padding_mode)

        return resamp_data, pix_coords

    def _generate_poses_and_Ks(self, pose_inputs, axisangles, translations, input_Ks):
        pose_features = self.models['pose_enc'](tf.concat(pose_inputs, axis=-1), training=self.opt.train_pose)
        pred_pose = self.models['pose_dec'](pose_features, training=self.opt.train_pose)
        # for experiments, some modes outputs have shape (B,2,1,3), sp it's a workaround to adapt to different shape
        angle = tf.expand_dims(pred_pose['angles'][:, 0, ...], 1)
        translation = tf.expand_dims(pred_pose['translations'][:, 0, ...], 1)
        axisangles.append(angle)  # B,1,1,3
        translations.append(translation)

        if 'intrinsics_head' in self.models:
            Ks = self.models['intrinsics_head'](pose_features[-1])
            Ks = make_hom_intrinsics(Ks, same_video=True)
            input_Ks.append(self.batch_processor.make_K_pyramid(Ks))

        return axisangles, translations, input_Ks

    def predict_poses(self, inputs, outputs):
        """Use pose enc-dec to calculate camera's angles and translations"""
        pose_inps = {f_i: inputs[("color_aug", f_i, 0)] for f_i in self.opt.frame_idx}
        if self.opt.concat_depth_pred:
            for f_i in self.opt.frame_idx:
                # Get RGB-D, by concat rgb-image and disp-pred for pose prediction
                pose_inps[f_i] = tf.concat(
                    [pose_inps[f_i], outputs[('disp', f_i, 0)]], axis=-1
                )
        pose_loss = 0
        axisangles, translations = [], []
        for f_i in self.opt.frame_idx[1:]:
            # To maintain ordering we always pass frames in temporal order
            if f_i < 0:
                pose_inputs = [pose_inps[f_i], pose_inps[0]]
            else:
                pose_inputs = [pose_inps[0], pose_inps[f_i]]

            input_Ks = []
            axisangles, translations, input_Ks = self._generate_poses_and_Ks(
                pose_inputs, axisangles, translations, input_Ks
            )
            if self.opt.calc_reverse_transform:
                pose_inputs = pose_inputs[::-1]
                axisangles, translations, input_Ks = self._generate_poses_and_Ks(
                    pose_inputs, axisangles,translations, input_Ks
                )

            # Intrinsics for fwd and bwd frames should be identical
            # todo: impose a loss to learn identical intrinsics for swapped frames
            if 'intrinsics_head' in self.models:
                for k in input_Ks[0].keys():
                    if self.opt.calc_reverse_transform:
                        inputs[k] = (input_Ks[0][k] + input_Ks[1][k]) * 0.5
                    else:
                        inputs[k] = input_Ks[0][k]

            # Invert the matrix if the frame id is negative
            loss, M, M_inv = transformation_loss(axisangles, translations,
                                                 invert=(f_i < 0), calc_reverse=self.opt.calc_reverse_transform)
            outputs[('cam_T_cam', f_i, 0)] = [M]
            if self.opt.calc_reverse_transform:
                outputs[('cam_T_cam', f_i, 0)].append(M_inv)
            if self.opt.add_pose_loss:
                pose_loss += loss

        if self.opt.add_pose_loss:
            # store 'pose_loss' in `outputs` instead of `self.loseses` maybe confusing
            # but it's an easy workaround to utilize @tf.function
            outputs[('pose_loss',)] = pose_loss / len(self.opt.frame_idx[1:]) * self.opt.pose_loss_weight

        return inputs, outputs

    def process_batch(self, inputs):
        """The whoel pipeline implemented in minibatch (pairwise images)
        1. Use Depth enc-dec, to predict disparity map in mutli-scales
        2. Use Pose enc-dec, to predict poses
        3. Use products from 1.2. to generate image (reprojection) predictions
        4. Compute Losses from 3.
        """
        # tgt_image_aug = inputs[('color_aug', 0, 0)]
        outputs = {}

        # ------ collecting disp and depth ------
        # - for f_i==0, collect disp in all scales, to calculate smooth loss
        # - for f_i!=0, only collect source-scale disp, for pose inputs
        # ---------------------------------------
        for f_i in self.opt.frame_idx:
            feature_raw = self.models['depth_enc'](inputs[('color_aug', f_i, 0)], self.opt.train_depth)
            pred_disp = self.models['depth_dec'](feature_raw, self.opt.train_depth)
            if f_i == 0:
                for s in range(self.opt.num_scales):
                    disp_raw = pred_disp["output_%d" % s]
                    outputs[('disp', 0, s)] = disp_raw
                    # Collect depth at respective scale, but resized to source_scale
                    disp_src_size = tf.image.resize(disp_raw, (self.opt.height, self.opt.width))
                    _, outputs[('depth', 0, s)] = disp_to_depth(disp_src_size, self.opt.min_depth, self.opt.max_depth)
            else:
                outputs[('disp', f_i, 0)] = pred_disp["output_0"]

        # -------------
        # 2. Pose
        # -------------
        inputs, outputs = self.predict_poses(inputs, outputs)

        # -------------
        # 3. Generate Reprojection from 1, 2
        # -------------
        self.generate_images_pred(inputs, outputs)

        return outputs

    def run_epoch(self, epoch):
        for _ in tqdm(range(self.train_loader.steps_per_epoch),
                      desc='Epoch%d/%d' % (epoch + 1, self.opt.num_epochs)):
            # data preparation
            batch = self.train_iter.get_next()
            inputs = self.batch_processor.prepare_batch(batch, is_train=True)

            # training
            trainable_weights = self.get_trainable_weights()
            grads, losses = self.grad(inputs, trainable_weights, global_step=self.global_step)
            self.optimizer.apply_gradients(zip(grads, trainable_weights))
            self.train_loss_tmp.append(losses['loss/total'])

            if self.is_time_to('log', self.global_step):
                mean_loss = np.mean(self.train_loss_tmp)
                self.train_loss_tmp = []
                print("loss: ", mean_loss)

            if self.is_time_to('validate', self.global_step):
                self.validate_miniset()

            self.global_step += 1

    def start_training(self):
        """Custom training loop
        - use @tf.function for self.grad() and self.dataloader.prepare_batch()
            allow larger batch_size GPU utilization.
        """
        for epoch in range(self.opt.num_epochs):
            self.optimizer.lr = self.lr_fn(epoch)  # learning rate 15:1e-4; >16:1e-5
            print("\tlearning rate - epoch %d: " % epoch, self.optimizer.get_config()['learning_rate'])

            self.run_epoch(epoch)

            print('-> Validating after epoch %d...' % epoch)
            val_losses = self.start_validating(self.global_step, num_run=self.val_loader.steps_per_epoch)
            self.save_models(val_losses, del_prev=False)

            # Set new weights_folder for next epoch
            print("save_model path:", self.opt.save_model_path)
            save_root = os.path.join(self.opt.save_model_path.rsplit('\\')[0])
            current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            self.opt.save_model_path = os.path.join(save_root, current_time)
            print('-> New weights will be saved in %s...' % self.opt.save_model_path)
            if os.path.isdir(self.opt.save_model_path):
                os.makedirs(self.opt.save_model_path)

    # @tf.function  # turn off to debug, e.g. with plt
    def grad(self, inputs, trainables, global_step):
        with tf.GradientTape() as tape:
            outputs = self.process_batch(inputs)
            losses = self.compute_losses(inputs, outputs)
            total_loss = losses['loss/total']
            grads = tape.gradient(total_loss, trainables)
            if self.is_time_to('log', global_step):
                print("colleting data in step ", global_step)
                self.collect_summary('train', losses, inputs, outputs, global_step)
        return grads, losses

    def get_trainable_weights(self):
        trainable_weights_all = []
        for m_name, model in self.models.items():
            if self.opt.train_pose and "pose" in m_name:
                # print("Training %s ..."%m_name)
                trainable_weights_all.extend(model.trainable_weights)
            if self.opt.train_depth and 'depth' in m_name:
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

    # @tf.function
    def compute_batch_losses(self, inputs):
        """@tf.function enables graph computation, allowing larger batch size
        """
        outputs = self.process_batch(inputs)
        losses = self.compute_losses(inputs, outputs)
        return outputs, losses

    def start_validating(self, global_step, num_run=30):
        """run mini-set to see if should store current-best weights"""
        losses_all = defaultdict(list)
        for i in range(num_run):
            batch = self.mini_val_iter.get_next()
            inputs = self.batch_processor.prepare_batch(batch, is_train=False)
            outputs, losses = self.compute_batch_losses(inputs)
            if ('depth_gt', 0) in inputs.keys():
                losses.update(
                    self.compute_depth_losses(inputs, outputs))
            for metric, loss in losses.items():
                losses_all[metric].append(loss)

        # mean of mini val-set
        for metric, loss in losses_all.items():
            med_range = [int(num_run * 0.1), int(num_run * 0.9)]
            losses_all[metric] = tf.reduce_mean(
                tf.sort(loss)[med_range[0]: med_range[1]]
            )
        # ----------
        # log val losses and delete
        # ----------
        if self.opt.recording:
            print('-> Writing loss/metrics...')
            # only record last batch
            self.collect_summary('val', losses, inputs, outputs, global_step=global_step)

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

    def early_stopping(self, losses, patience=3):
        """mean val_loss_metrics are good enough to stop early"""
        early_stop = False
        for metric in self.min_errors_thresh:
            self.early_stopping_losses[metric].append(losses[metric])
            mean_error = np.mean(self.early_stopping_losses[metric][-patience:])
            if mean_error > self.min_errors_thresh[metric]:
                return early_stop

        if np.mean(self.train_loss_tmp) < 0.9:
            early_stop = True

        return early_stop

    def validate_miniset(self):
        val_losses = self.start_validating(self.global_step)

        # save models if needed
        is_lowest, self.val_losses_min = is_val_loss_lowest(val_losses,
                                                            self.val_losses_min, self.min_errors_thresh)
        if is_lowest:
            self.save_models(del_prev=True)
        if self.early_stopping(val_losses):
            self.save_models(val_losses, del_prev=False)

        del val_losses  # delete tmp

    def save_models(self, val_losses=None, del_prev=True):
        """save models in 1) during epoch val_loss hits new low, 2) after one epoch"""
        # - delete previous weights
        if del_prev:
            del_files(self.opt.save_model_path)

        # - save new weightspy
        if val_losses is None:
            val_losses = self.val_losses_min
        if self.train_loader.has_depth:
            weights_name = '_'.join(['weights', str(val_losses['loss/total'].numpy())[2:5],
                                     str(val_losses['da/a1'].numpy())[2:4]])
        else:
            weights_name = '_'.join(['weights', str(val_losses['loss/total'].numpy())[2:5]])
        print('-> Saving weights with new low loss:\n', self.val_losses_min)
        weights_path = os.path.join(self.opt.save_model_path, weights_name)
        if not os.path.isdir(weights_path) and not self.opt.debug_mode:
            os.makedirs(weights_path)

        white_list = []
        if self.opt.train_depth: white_list.extend(['depth_enc', 'depth_dec'])
        if self.opt.train_pose: white_list.extend(['pose_enc', 'pose_dec'])
        if self.opt.train_intriniscs: white_list.extend(['intrinsics_head'])
        for m_name, model in self.models.items():
            if m_name in white_list:
                m_path = os.path.join(weights_path, m_name + '.h5')
                tf.print("saving {} to:".format(m_name), m_path, output_stream=sys.stdout)
                model.save_weights(m_path)

    def collect_summary(self, mode, losses, inputs, outputs, global_step):
        """collect summary for train / validation"""
        writer = self.summary_writer[mode]
        with writer.as_default():
            for loss_name in list(losses):
                loss_val = losses[loss_name]
                tf.summary.scalar(loss_name, loss_val, step=global_step)

            # images
            tf.summary.image('tgt_image', inputs[('color', 0, 0)][:2], step=global_step)
            tf.summary.image('scale0_disp_color', colorize(outputs[('disp', 0, 0)][:2], cmap='plasma'),
                             step=global_step)
            # tf.summary.image('scale0_disp_gray',
            #                  normalize_image(outputs[('disp', 0, 0)][:2]), step=global_step)

            for f_i in self.opt.frame_idx[1:]:
                tf.summary.image('scale0_disp_color', colorize(outputs[('disp', f_i, 0)][:2], cmap='plasma'),
                                 step=global_step)
                tf.summary.image('scale0_proj_{}'.format(f_i),
                                 outputs[('warped_multi_s', f_i, 0)][0][:2,...,:3], step=global_step)
                # tf.summary.image('scale0_proj_error_{}'.format(f_i),
                #                  outputs[('proj_error', f_i, 0)][:2], step=global_step)

            if self.opt.learn_intrinsics:
                tf.summary.scalar('fx', inputs[('K', 0)][0, 0, 0])
                tf.summary.scalar('fy', inputs[('K', 0)][0, 1, 1])
                tf.summary.scalar('x0', inputs[('K', 0)][0, 0, 2])
                tf.summary.scalar('y0', inputs[('K', 0)][0, 1, 2])

            # if self.opt.do_automasking:
            # tf.summary.image('scale0_automask_image',
            #                  outputs[('automask', 0)][:2], step=global_step)

        writer.flush()

    def compute_depth_losses(self, inputs, outputs):
        """Compute depth metrics, to allow monitoring during training
        -> Only for KITTI-RawData, where velodyne depth map is valid !!!

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        losses = {}
        depth_gt = inputs[('depth_gt', 0)]
        mask = depth_gt > 0
        depth_pred = outputs[('depth', 0, 0)]  # to be resized to (375, 1242)
        depth_pred = tf.clip_by_value(
            tf.image.resize(depth_pred, depth_gt.shape[1:-1]),
            1e-3, 80
        )
        assert depth_pred.shape[1] == 375, \
            'shape: {}, should be {}'.format(depth_pred.shape, depth_gt.shape[1:])

        # garg/eigen crop
        crop_mask = np.zeros(shape=mask.shape, dtype=np.bool)
        crop_mask[:, 153:371, 44:1197, :] = True
        mask = mask.numpy() * crop_mask

        depth_gt_masked = tf.boolean_mask(depth_gt, mask)
        depth_pred_masked = tf.boolean_mask(depth_pred, mask)
        depth_pred_med = depth_pred_masked * np.median(depth_gt_masked) / np.median(depth_pred_masked)

        depth_pred_final = tf.clip_by_value(depth_pred_med, 1e-3, 80)

        depth_errors = compute_depth_errors(depth_gt_masked, depth_pred_final)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = depth_errors[i]

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
        events = {0: 'log', 1: 'validate'}
        if event not in events.values():
            raise NotImplementedError

        if event == events[0]:
            early_phase = global_step % self.opt.record_freq == 0 and global_step < 1000
            late_phase = global_step % (self.opt.record_freq*2) == 0
            if early_phase or late_phase:
                is_time = True
        elif event == events[1] and global_step % (
                self.train_loader.steps_per_epoch // self.opt.val_num_per_epoch) == 0:
            is_time = True

        return is_time


if __name__ == '__main__':
    trainer = Trainer(options)
    trainer.train(record=False)
