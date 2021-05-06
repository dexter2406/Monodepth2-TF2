from models.depth_decoder_creater import DepthDecoder_full
from models.encoder_creater import ResNet18_new
from models.posenet_decoder_creator import PoseDecoder_exp

from utils import disp_to_depth, del_files, make_hom_intrinsics
from src.trainer_helper import *
from src.test_inverse_warp import *
from src import regularizers

from datasets.data_loader_kitti import DataLoader
from datasets.dataset_kitti import KITTIRaw, KITTIOdom
from src.DataPprocessor import DataProcessor
import numpy as np
from collections import defaultdict
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
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
        self.num_scales = len(self.opt.scales)
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

        view_options(self.opt)
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
        self.batch_processor = DataProcessor(frame_idx=self.opt.frame_idx, intrinsics=train_dataset.K)
        # if not self.opt.learn_intrinsics:
        #     self.batch_processor.get_intrinsics(train_dataset.K)
        if self.opt.disable_gt:
            self.batch_processor.disable_groundtruth()

        # Init models
        self.models['depth_enc'] = ResNet18_new(norm_inp=self.opt.norm_input)
        self.models['depth_dec'] = DepthDecoder_full()
        self.models['pose_enc'] = ResNet18_new(norm_inp=self.opt.norm_input)
        self.models['pose_dec'] = PoseDecoder_exp(pose_num=self.opt.pose_num)

        build_models(self.models, rgb_cat_depth=self.opt.concat_depth)
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
            if m_name == 'pose_enc' and self.opt.concat_depth:
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
        grad_disp_x = tf.math.abs(norm_disp - tf.roll(norm_disp, shift=1, axis=1))
        grad_disp_y = tf.math.abs(norm_disp - tf.roll(norm_disp, shift=1, axis=2))

        grad_img_x = tf.math.abs(img - tf.roll(img, shift=1, axis=1))
        grad_img_y = tf.math.abs(img - tf.roll(img, shift=1, axis=2))

        weight_x = tf.math.exp(-tf.reduce_mean(grad_img_x, 3, keepdims=True))
        weight_y = tf.math.exp(-tf.reduce_mean(grad_img_y, 3, keepdims=True))

        smoothness_x = grad_disp_x * weight_x
        smoothness_y = grad_disp_y * weight_y

        return tf.reduce_mean(smoothness_x) + tf.reduce_mean(smoothness_y)

    def compute_reproject_loss(self, proj_data, tgt_data, validity_mask=None):
        # todo: weighted SSIM?
        abs_diff = tf.math.abs(proj_data - tgt_data)
        ssim_diff = SSIM(proj_data, tgt_data)
        # print('ssim before:', tf.reduce_mean(ssim_diff))
        if validity_mask is not None:
            abs_diff = abs_diff * validity_mask
            ssim_diff = ssim_diff * validity_mask
            # print('ssim after:', tf.reduce_mean(ssim_diff))

        l1_loss = tf.reduce_mean(abs_diff, axis=3, keepdims=True)
        ssim_loss = tf.reduce_mean(ssim_diff, axis=3, keepdims=True)

        loss = self.opt.ssim_ratio * ssim_loss + (1 - self.opt.ssim_ratio) * l1_loss
        return loss

    def reset_losses(self):
        self.losses = {'pixel_loss': 0, 'smooth_loss': 0}

    def regularize_mask(self, sampler_mask):
        """Regularizer for sampler mask"""
        # for frame_next resampled to frame_prev, there's padding region, which would be masked out
        # when calculating losses. To prevent masked area take dominance, regularizer is applied.
        # Padded area usually below a certain percentage depending on its velocity.
        # if self.opt.mask_loss_w != 0.0:
        mask_loss = 0.
        cover_perc = tf.reduce_mean(sampler_mask)
        print("coverage:", cover_perc)
        if cover_perc < self.opt.mask_cover_min:
            mask_loss = self.opt.mask_cover_min - cover_perc
        return mask_loss

    def compute_losses(self, inputs, outputs):
        scale0 = 0
        image_tgt = inputs[('color', 0, scale0)]
        self.reset_losses()
        total_loss = 0.
        mask_loss = 0.
        pixel_loss = 0.
        smooth_loss = 0.

        # Scales are for smoothness (prevent texture-copy artifacts), and reprojection losses,
        # other losses are not calculated in different scales
        for scale in self.opt.scales:
            scale_wt = 1. / 2 ** scale
            reproject_losses = []

            # -------------------
            # Loss-1: Reprojection / warping loss
            # -------------------
            for i, f_i in enumerate(self.opt.frame_idx[1:]):
                validity_mask = outputs[('validity_mask', f_i, 0)]
                image_src2tgt = outputs[('warped_multi_s', f_i, scale)]
                reproject_losses.append(
                    self.compute_reproject_loss(image_src2tgt, image_tgt, validity_mask)
                )
            reproj_loss = tf.concat(reproject_losses, axis=3)
            if self.avg_reprojection:
                reproj_loss = tf.math.reduce_mean(reproject_losses, axis=3, keepdims=True)

            # -------------------
            # Optional: auto-masking
            # identity error between source vs. current scale
            # -------------------
            if not self.opt.do_automasking:
                combined = reproject_losses
            else:
                identity_reproj_loss = self.apply_automasking_on_reproj_loss(inputs, outputs)
                combined = tf.concat([identity_reproj_loss, reproj_loss], axis=3)
                # outputs[('automask', 0)] = tf.expand_dims(
                #     tf.cast(tf.math.argmin(combined, axis=3) > 1, tf.float32) * 255, -1)

            # ---------------------
            # Final reprojection loss -> named as pixel loss
            # Reprojection Loss with Minimal Projection, monodepth2
            # is not normalize by scale
            # ---------------------
            if self.opt.use_min_proj and combined.shape[-1] != 1:
                to_optimise = tf.reduce_min(combined, axis=3)
            else:
                to_optimise = combined
            pixel_loss += tf.reduce_mean(to_optimise)

            # -------------------
            #  Loss-2: Smoothness loss: Gradient Loss based on image pixels
            # -------------------
            disp_tgt_s = outputs[('disp', 0, scale)]
            image_tgt_s = inputs['color', 0, scale]
            smooth_loss_raw = self.get_smooth_loss(disp_tgt_s, image_tgt_s)
            smooth_loss += smooth_loss_raw * scale_wt

        # ----------------
        # pixel_loss and smooth_loss
        # ----------------
        self.losses['pixel_loss'] = pixel_loss * self.opt.reproj_loss_w / self.num_scales
        self.losses['smooth_loss'] = smooth_loss * self.opt.smoothness_ratio / self.num_scales
        total_loss += self.losses['pixel_loss'] + self.losses['smooth_loss']

        # ---------------
        # Loss-3: Depth Consistency loss, from struct2depth
        # ---------------
        if self.opt.add_depth_loss:
            depth_loss = outputs[('depth_loss',)]
            self.losses['depth_loss'] = depth_loss * self.opt.depth_loss_w
            total_loss += self.losses['depth_loss']

        # ---------------
        # Loss-4: Pose Consistency Losses: rotation should be identical when in reverse order
        # ---------------
        if self.opt.add_rot_loss:
            rot_loss = outputs[('rot_loss',)] * self.opt.pose_loss_w
            self.losses['rot_loss'] = rot_loss
            total_loss += rot_loss
            print("rot loss:", self.losses['rot_loss'])

        # ---------------
        # Loss-5: Mask regularizer on sampler_mask, prevent the mask to shrink to zero, should stay zero
        # ---------------
        if self.opt.add_mask_loss:
            mask_loss = self.regularize_mask(outputs[('sampler_mask', -1, 0)]) * self.opt.mask_loss_w
            total_loss += mask_loss
            self.losses['mask_loss'] = mask_loss

        self.losses['loss/total'] = total_loss

        for k, v in self.losses.items():
            print(k, '\t', v)
        if self.opt.debug_mode:
            colormapped_normal = colorize(outputs[('disp', 0, 0)], cmap='plasma')
            if colormapped_normal.shape[0] > 1:
                colormapped_normal = colormapped_normal[0]
            plt.imshow(colormapped_normal.numpy()), plt.show()

        return self.losses

    def apply_automasking_on_reproj_loss(self, inputs, outputs):
        identity_reproj_losses = []
        for f_i in self.opt.frame_idx[1:]:
            validity_mask = outputs[('validity_mask', f_i, 0)]
            image_src = inputs[('color', f_i, 0)]
            image_tgt = inputs[('color', 0, 0)]
            identity_reproj_losses.append(
                self.compute_reproject_loss(image_src, image_tgt, validity_mask)
            )
        identity_reproj_loss = tf.concat(identity_reproj_losses, axis=3)
        if self.avg_reprojection:
            identity_reproj_loss = tf.math.reduce_mean(identity_reproj_loss, axis=3, keepdims=True)
        # add random numbers to break ties
        identity_reproj_loss += (tf.random.normal(identity_reproj_loss.shape)
                                 * tf.constant(1e-5, dtype=tf.float32))
        return identity_reproj_loss

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        Keys for `outputs` explained:
        - ['disp', <frame_id>, <scale>]: Tensor
            -> label and content just like `inputs`, it's default setting if not specified
            -> shape (B,H,W,C), channel `C` can be 3 for RGB or 4 for an additional depth map
        - [('warped_multi_s', <frame_id>, <scale>)] : Tensor
            source frame inverse warped to target frame, e.g. -1->0, 1->0
        """
        rot_losses = 0.
        depth_losses = 0.
        for scale in self.opt.scales:
            scale_wt = 1. / 2 ** scale
            disp_tgt_s = tf.image.resize(outputs[('disp', 0, scale)], (self.opt.height, self.opt.width))
            # _, depth_tgt_s = disp_to_depth(disp_tgt_s, self.opt.min_depth, self.opt.max_depth)
            _, depth_tgt_s = self.disp_to_depth(disp_tgt_s)

            if scale == 0:
                outputs[('depth', 0, 0)] = depth_tgt_s  # for depth_gt supervision

            for i, f_i in enumerate(self.opt.frame_idx[1:]):
                # ----------------
                # Inverse warp source -> target view, namely prev->curr, next->curr
                # ----------------
                data_src = inputs[('color', f_i, scale)]
                if scale in self.opt.scales[1:]:
                    data_src2tgt, transformation_map = self.inverse_warping(depth_tgt_s, data_src, outputs, f_i)
                    outputs[('warped_multi_s', f_i, scale)] = data_src2tgt[..., :3]

                elif scale == self.opt.scales[0]:
                    # Many calculation only happens in source scale (scale==0)
                    if self.opt.add_depth_loss:
                        depth_src = self.disp_to_depth(outputs[('disp', f_i, 0)])[1]
                        data_src = tf.concat([data_src, depth_src], axis=3)
                    data_src2tgt, transformation_map = self.inverse_warping(depth_tgt_s, data_src, outputs, f_i)
                    outputs[('warped_multi_s', f_i, 0)] = data_src2tgt[..., :3]
                    outputs[('sampler_mask', f_i, 0)] = transformation_map.mask
                    outputs['transformed_map', f_i, 0] = transformation_map.depth
                    if scale == 0 and self.opt.use_occlu_mask:
                        occlu_mask = self.get_occlusion_aware_mask(outputs, f_i, transformation_map, 0)
                        outputs[('validity_mask', f_i, 0)] = transformation_map.mask * occlu_mask
                    else:
                        outputs[('validity_mask', f_i, 0)] = outputs[('sampler_mask', f_i, 0)]

                    if self.opt.add_depth_loss:
                        depth_src2tgt = data_src2tgt[..., 3:4]
                        depth_l1_diff = tf.abs(transformation_map.depth - depth_src2tgt)
                        depth_scaling = tf.reduce_mean(transformation_map.depth, axis=[1, 2], keepdims=True) + \
                                        tf.reduce_mean(depth_src2tgt, axis=[1, 2], keepdims=True)
                        depth_losses += tf.reduce_mean(
                            depth_l1_diff / depth_scaling * outputs[('validity_mask', f_i, 0)]
                        ) * scale_wt
                    # ----------------
                    # Rotation Consistency loss
                    # ----------------
                    if self.opt.add_rot_loss:
                        # todo: rot_loss needs be normalized by 2**scale?
                        rot_loss = self.motion_consistency_losses(outputs, f_i)
                        rot_losses += rot_loss * scale_wt

        if self.opt.add_rot_loss:
            # store 'rot_loss' in `outputs` instead of `self.loseses` maybe confusing
            # but it's an easy workaround to utilize @tf.function
            outputs[('rot_loss',)] = rot_losses
        if self.opt.add_depth_loss:
            outputs[('depth_loss',)] = depth_losses

        if self.opt.debug_mode:
            show_images(inputs, outputs)
        return outputs

    def get_occlusion_aware_mask(self, outputs, f_i, transformed_map, scale):
        depth_src2tgt = self.disp_to_depth(outputs[('disp', f_i, scale)])[1]
        occlu_mask = tf.cast(
            tf.greater_equal(transformed_map.depth, depth_src2tgt),
            tf.float32)
        return occlu_mask

    def motion_consistency_losses(self, outputs, f_i):
        M, M_inv = make_transform_mats(
            outputs[('axisangles', f_i, 0)],
            outputs[('translations', f_i, 0)],
            invert=(f_i < 0)
        )
        # calculate pose loss (actually it's consistency loss.. in struct2depth)
        rot_loss = regularizers.rot_consis_loss(M, M_inv)
        return rot_loss

    def inverse_warping(self, depth_goal, frame_to_resamp, outputs, f_i, invert=None):
        """inverse-warping: transfer src_data -> resamp_data
        Args:
            depth_goal: depth of desired frame
            frame_to_resamp: data to be resampled, can be image or depth map
            outputs: contains K, rotation angles, translations
        Returns:
            frame_resamp: resampled data
            transformation_map: TransformationMap object
                containing target 1)grid coordinates, 2)transformed depthmap, 3) validity mask
        """
        transformation_map = self.create_transforamtion_map(depth_goal, outputs, f_i, invert)
        frame_resamp = resampler(
            frame_to_resamp,
            transformation_map.pixel_x,
            transformation_map.pixel_y,
            safe=False)
        return frame_resamp, transformation_map

    def create_transforamtion_map(self, depth_goal, outputs, f_i, invert=None):
        if invert is None:
            reverse_order = f_i < 0
        else:
            reverse_order = invert  # Override
        idx = 1 if reverse_order else 0
        intrinsics = outputs[('K', 0)]
        rot_angles = outputs[('axisangles', f_i, 0)][idx]
        translations = outputs[('translations', f_i, 0)][idx]
        px, py, z = get_transformed_depth_coords(depth_goal, translations, rot_angles, intrinsics)
        pixel_x, pixel_y, mask = clamp_and_filter_result(px, py, z)
        transformation_map = TransformationMap(pixel_x, pixel_y, z, mask)
        return transformation_map

    def _generate_poses_and_Ks(self, pose_inputs, axisangles, translations):
        pose_inputs = tf.concat(pose_inputs, axis=-1)
        pose_features = self.models['pose_enc'](pose_inputs, training=self.opt.train_pose)
        pred_pose = self.models['pose_dec'](pose_features, training=self.opt.train_pose)

        # for experiments, some modes outputs have shape (B,2,1,3), sp it's a workaround to adapt to different shape
        angle = tf.expand_dims(pred_pose['angles'][:, 0, ...], 1)
        translation = tf.expand_dims(pred_pose['translations'][:, 0, ...], 1)
        axisangles.append(angle)  # B,1,1,3
        translations.append(translation)

        return axisangles, translations

    def predict_poses(self, inputs, outputs):
        """Use pose enc-dec to calculate camera's angles and translations"""
        pose_inps = {f_i: inputs[("color_aug", f_i, 0)] for f_i in self.opt.frame_idx}
        if self.opt.concat_depth:
            for f_i in self.opt.frame_idx:
                # Get RGB-D, by concat rgb-image and disp-pred for pose prediction
                pose_inps[f_i] = tf.concat(
                    [pose_inps[f_i], outputs[('disp', f_i, 0)]], axis=-1)

        for f_i in self.opt.frame_idx[1:]:
            # To maintain ordering we always pass frames in temporal order
            axisangles, translations = [], []
            if f_i < 0:
                pose_inputs = [pose_inps[f_i], pose_inps[0]]
            else:
                pose_inputs = [pose_inps[0], pose_inps[f_i]]

            axisangles, translations = self._generate_poses_and_Ks(
                pose_inputs, axisangles, translations
            )

            if self.opt.include_revers:
                pose_inputs = pose_inputs[::-1]
                axisangles, translations = self._generate_poses_and_Ks(
                    pose_inputs, axisangles, translations
                )

            outputs[('axisangles', f_i, 0)] = axisangles
            outputs[('translations', f_i, 0)] = translations

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
        for k in inputs:
            for scale in self.opt.scales:
                if ('K', scale) == k or ('K_inv', scale):
                    outputs[k] = inputs[k]

        # ------ collecting disp and depth ------
        # - for f_i==0, collect disp in all scales, to calculate smooth loss
        # - for f_i!=0, only collect source-scale disp, for pose inputs
        # ---------------------------------------
        for f_i in self.opt.frame_idx:
            feature_raw = self.models['depth_enc'](inputs[('color_aug', f_i, 0)], self.opt.train_depth)
            pred_disp = self.models['depth_dec'](feature_raw, self.opt.train_depth)
            if f_i == 0:
                for s in self.opt.scales:
                    disp_raw = pred_disp["output_%d" % s]
                    outputs[('disp', f_i, s)] = disp_raw
                    # Collect depth at respective scale, but resized to source_scale
                    disp_src_size = tf.image.resize(disp_raw, (self.opt.height, self.opt.width))
                    _, outputs[('depth', f_i, s)] = disp_to_depth(disp_src_size, self.opt.min_depth, self.opt.max_depth)
            else:
                outputs[('disp', f_i, 0)] = pred_disp["output_0"]
                # outputs.update(make_depth_pyramids(outputs, f_i, self.opt.pyramid_scales))

        # -------------
        # 2. Pose
        # -------------
        inputs, outputs = self.predict_poses(inputs, outputs)

        # -------------
        # 3. Generate Reprojection from 1, 2
        # -------------
        outputs.update(self.generate_images_pred(inputs, outputs))
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
            self.opt.save_model_path = self.opt.save_model_path.replace('\\', '/')
            save_root = os.path.join(self.opt.save_model_path.rsplit('/', 1)[0])
            current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            self.opt.save_model_path = os.path.join(save_root, current_time).replace('\\', '/')
            print('-> New weights will be saved in {}...'.format(self.opt.save_model_path))
            if not os.path.isdir(self.opt.save_model_path):
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
                # print("Training %s ..." % m_name)
                trainable_weights_all.extend(model.trainable_weights)
            else:
                model.trainable = False
            if self.opt.train_depth and 'depth' in m_name:
                # print("Training %s ..." % m_name)
                trainable_weights_all.extend(model.trainable_weights)
            else:
                model.trainable = False
        return trainable_weights_all

    def train(self):
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
        # losses_all = defaultdict(list)
        losses_all = {}
        for i in range(num_run):
            batch = self.mini_val_iter.get_next()
            inputs = self.batch_processor.prepare_batch(batch, is_train=False)
            outputs, losses = self.compute_batch_losses(inputs)

            # if ('depth_gt', 0) in inputs.keys():
            if not self.opt.disable_gt:
                losses.update(
                    self.compute_depth_losses(inputs, outputs)
                )
            for metric, loss in losses.items():
                if metric not in losses_all.keys():
                    losses_all[metric] = []
                losses_all[metric].append(loss)

        # mean of mini val-set
        for metric, loss in losses_all.items():
            med_range = [min(0, int(num_run * 0.1)), max(1, int(num_run * 0.9))]
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

    def validate_miniset(self):
        val_losses = self.start_validating(self.global_step)

        # save models if needed
        is_lowest, self.val_losses_min = is_val_loss_lowest(val_losses,
                                                            self.val_losses_min,
                                                            self.min_errors_thresh,
                                                            self.opt.disable_gt)
        if is_lowest:
            self.save_models(del_prev=True)
        # if self.early_stopping(val_losses):
        #     self.save_models(val_losses, del_prev=False)

        del val_losses  # delete tmp

    def save_models(self, val_losses=None, del_prev=True):
        """save models in 1) during epoch val_loss hits new low, 2) after one epoch"""
        # - delete previous weights
        if del_prev:
            del_files(self.opt.save_model_path)

        # - save new weightspy
        if val_losses is None:
            val_losses = self.val_losses_min

        weights_name = '_'.join(['weights', str(val_losses['loss/total'].numpy())[2:5]])
        if not self.opt.disable_gt:
            weights_name = '_'.join([weights_name, str(val_losses['da/a1'].numpy())[2:4]])

        print('-> Saving weights with new low loss:\n', self.val_losses_min)
        weights_path = os.path.join(self.opt.save_model_path, weights_name)
        if not os.path.isdir(weights_path) and not self.opt.debug_mode:
            os.makedirs(weights_path)

        white_list = []
        if self.opt.train_depth: white_list.extend(['depth_enc', 'depth_dec'])
        if self.opt.train_pose: white_list.extend(['pose_enc', 'pose_dec'])
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
                                 outputs[('warped_multi_s', f_i, 0)][:2, ..., :3], step=global_step)
                # tf.summary.image('scale0_proj_error_{}'.format(f_i),
                #                  outputs[('proj_error', f_i, 0)][:2], step=global_step)
            if self.opt.learn_intrinsics:
                tf.summary.scalar('fx', outputs[('K', 0)][0, 0, 0], step=global_step)
                tf.summary.scalar('fy', outputs[('K', 0)][0, 1, 1], step=global_step)
                tf.summary.scalar('x0', outputs[('K', 0)][0, 0, 2], step=global_step)
                tf.summary.scalar('y0', outputs[('K', 0)][0, 1, 2], step=global_step)

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
            special_pass = False
            early_phase = global_step % self.opt.record_freq == 0 and global_step < 1000
            late_phase = global_step % (self.opt.record_freq * 2) == 0
            if early_phase or late_phase or special_pass:
                is_time = True
        elif event == events[1]:
            special_pass = False
            is_time = global_step % (self.train_loader.steps_per_epoch // self.opt.val_num_per_epoch) == 0
            is_time = is_time or special_pass

        return is_time
