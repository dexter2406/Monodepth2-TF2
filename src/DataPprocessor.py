import tensorflow as tf
import os
import time
import math
import numpy as np
import sys
from utils import assert_valid_hom_intrinsics


class DataProcessor(object):
    def __init__(self, frame_idx, intrinsics, num_scales=4, disable_gt=True):
        self.disable_gt = disable_gt
        self.num_scales = num_scales     # for evaluation, 1
        self.height = 192
        self.width = 640
        self.frame_idx = frame_idx
        self.batch_size = -1
        self.K = intrinsics
        self.brightness = 0.2
        self.sature_contrast = 0.2
        self.hue = 0.05

    def get_intrinsics(self, K):
        self.K = K

    def disable_groundtruth(self):
        self.disable_gt = True

    @tf.function
    def prepare_batch(self, batch, is_train):
        """Apply augmentation
        tgt_batch: ndarray
            - [<Batch>, 192, 640, 3]
        src_batch: ndarray:
            - [<Batch>, 192, 640, 6]

        Outputs: raw and augmented data, stored in dictionary
        Dictionary keys:
        ("color", <frame_id>, <scale>)          for raw colour images,
        ("color_aug", <frame_id>, <scale>)      for augmented colour images,
        ("K", scale) or ("inv_K", scale)        for camera intrinsics,
        ------- Examples -------
        tgt_image = inputs['color', 0, 0]
        src_image_stack_aug = inputs['color_aug', 1:, :]
        tgt_image_pyramid = inputs['color, 0, :]
        intrinsics_scale_0 = inputs[('K', 0)]
        """
        do_color_aug = is_train and np.random.random() > 0.5
        inputs, input_Ks = self.process_batch_main(batch, do_color_aug)
        return inputs, input_Ks

    def prepare_batch_val(self, batch):
        """duplicate of prepare_batch(), no @tf.function decorator
        For validation OR evaluation
        """
        # tf.print('val shape')
        # tf.print(batch[0].shape, batch[1].shape, output_stream=sys.stdout)
        inputs, input_Ks = self.process_batch_main(batch, do_color_aug=False)
        return inputs, input_Ks

    def process_batch_main(self, batch, do_color_aug=False):
        inputs = {}
        if type(batch) == tuple:
            batch_imgs = batch[0]
            inputs[('depth_gt', 0)] = tf.expand_dims(batch[1], 3)
        else:
            batch_imgs = batch
        tgt_batch, src_batch = batch_imgs[..., :3], batch_imgs[..., 3:]
        self.batch_size = tgt_batch.shape[0]

        # in training / eval_depth / eval_pose
        # frame_idx == [0,-1,1] / [0] / [0,1] respectively
        inputs[('color', self.frame_idx[0], -1)] = tgt_batch
        if len(self.frame_idx) > 1:
            for i, f_i in enumerate(self.frame_idx[1:]):
                inputs[('color', f_i, -1)] = src_batch[..., i*3: (i+1)*3]

        inputs, input_Ks = self.preprocess(inputs, do_color_aug)
        self.delete_raw_images(inputs)
        return inputs, input_Ks

    def generate_aug_params(self):
        brightness = np.random.uniform(0, self.brightness)
        hue = np.random.uniform(0, self.hue)
        saturation, contrast = np.random.uniform(1 - self.sature_contrast, 1 + self.sature_contrast, size=2)
        return brightness, saturation, contrast, hue

    def color_aug(self, image, aug_params):
        """Apply augmentation (fixed seed needed)
        Same aug for each batch (and scales), Note that all images input to the pose network receive the
        same augmentation.
        """
        if aug_params is None:
            return image
        else:
            image = tf.image.adjust_brightness(image, aug_params[0])
            image = tf.image.adjust_saturation(image, aug_params[1])
            image = tf.image.adjust_contrast(image, aug_params[2])
            image = tf.image.adjust_hue(image, aug_params[3])
        return image

    def preprocess(self, inputs, do_color_aug):
        """Make pyramids and augmentations
        - pyramid: use the raw (scale=-=1) to produce scale==[0:4] images
        - augment: correspond to the pyramid source
        """
        aug_params = None
        if do_color_aug is not None:
            # generate one set of random aug factor for one batch, making that same pair has same aug effects
            aug_params = self.generate_aug_params()
        for k in list(inputs):
            if 'depth_gt' in k:
                continue
            img_type, f_i, scale = k    # key components
            if img_type == 'color':
                for scale in range(self.num_scales):
                    src_image = inputs[(img_type, f_i, scale - 1)]
                    resized_image = tf.image.resize(src_image,
                                                    (self.height//(2**scale), self.width//(2**scale)),
                                                    antialias=True)
                    inputs[(img_type, f_i, scale)] = resized_image
                    inputs[(img_type + '_aug', f_i, scale)] = self.color_aug(resized_image, aug_params)

            input_Ks = self.make_K_pyramid()

        return inputs, input_Ks

    def make_K_pyramid(self):
        """genearing intrinsics pyramid
        Args:
            K: partial intrinsics, shape best to be [B,3,3], but [B,4,4], [4,4] or [B,4,4] are also OK
        Returns:
            input_Ks: dict
                a pyramid of homogenous intrinsics and its inverse, each has shape [B,4,4]
        """
        input_Ks = {}
        for scale in range(self.num_scales):
            K_tmp = self.K.copy()
            K_tmp[0, :] *= self.width // (2 ** scale)
            K_tmp[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K_tmp)

            K_tmp = tf.reshape(tf.tile(K_tmp, [self.batch_size, 1]), (self.batch_size, 4, 4))
            inv_K = tf.reshape(tf.tile(inv_K, [self.batch_size, 1]), (self.batch_size, 4, 4))
            assert K_tmp.shape[-1] == 4

            input_Ks[("K", scale)] = K_tmp
            input_Ks[("inv_K", scale)] = inv_K
        assert_valid_hom_intrinsics(input_Ks[("K", 0)])
        return input_Ks

    def delete_raw_images(self, inputs, scale_to_del=-1):
        for idx in self.frame_idx:
            del inputs[('color', self.frame_idx[idx], scale_to_del)]
