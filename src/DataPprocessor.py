import tensorflow as tf
import os
import time
import math
import numpy as np


class DataProcessor(object):
    def __init__(self, is_train=False):
        self.is_train = is_train
        self.num_scales = 4
        self.height = 192
        self.width = 640
        self.frame_idx = [0, -1, 1]
        self.batch_size = -1
        # For KITTI. Must *normalize* when using on different scale
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

    def set_train(self):
        self.is_train = True

    def prepare_batch(self, tgt_batch, src_batch):
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
        tgt_image = input_imgs['color', 0, 0]
        src_image_stack_aug = input_imgs['color_aug', 1:, :]
        tgt_image_pyramid = input_imgs['color, 0, :]
        """
        do_color_aug = self.is_train and np.random.random() > 0.5
        do_flip = False
        # print("\t check input batch: ", tgt_batch.shape," and ", src_batch.shape)
        self.batch_size = tgt_batch.shape[0]

        input_imgs, input_Ks = {}, {}
        input_imgs[('color', self.frame_idx[0], -1)] = tgt_batch
        input_imgs[('color', self.frame_idx[1], -1)] = src_batch[..., :3]
        input_imgs[('color', self.frame_idx[2], -1)] = src_batch[..., 3:]

        input_imgs, input_Ks = self.preprocess(input_imgs, input_Ks)

        return input_imgs, input_Ks

    def delete_raw_images(self, input_imgs, scale_to_del=-1):
        for idx in self.frame_idx:
            del input_imgs[('color', self.frame_idx[idx], scale_to_del)]

    def preprocess(self, input_imgs, input_Ks):
        """Make pyramids and augmentations
        - pyramid: use the raw (scale=-=1) to produce scale==[0:4] images
        - augment: correspond to the pyramid source
        """
        # print("\t preprocessing batch...")
        intrinsics_mscale = []
        for k in list(input_imgs):
            img_type, f_i, scale = k    # key components
            assert img_type == 'color'
            for scale in range(self.num_scales):
                src_image = input_imgs[(img_type, f_i, scale - 1)]
                resized_image = tf.image.resize(src_image,
                                                (self.height//(2**scale), self.width//(2**scale)),
                                                antialias=True)
                input_imgs[(img_type, f_i, scale)] = resized_image
                input_imgs[(img_type + '_aug', f_i, scale)] = self.color_aug(resized_image)

        for scale in range(self.num_scales):
            K = self.K.copy()
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            K = tf.reshape(tf.tile(K, [self.batch_size,1]), (self.batch_size, 4, 4))
            inv_K = tf.reshape(tf.tile(inv_K, [self.batch_size,1]), (self.batch_size, 4, 4))
            assert K.shape[-1] == 4

            input_Ks[("K", scale)] = tf.constant(K)
            input_Ks[("inv_K", scale)] = tf.constant(inv_K)

        return input_imgs, input_Ks

    def make_intrinsics_matrix(self, fx, fy, cx, cy):
        # Assumes batch input
        batch_size = fx.shape[0]
        zeros = tf.zeros_like(fx)
        r1 = tf.stack([fx, zeros, cx], axis=1)
        r2 = tf.stack([zeros, fy, cy], axis=1)
        r3 = tf.constant([0., 0., 1.], shape=[1, 3])
        r3 = tf.tile(r3, [batch_size, 1])
        intrinsics = tf.stack([r1, r2, r3], axis=1)
        return intrinsics

    def color_aug(self, image):
        # todo: implement color augmentations
        """Apply augmentation (fixed seed needed)
        Same aug for each batch (and scales), Note that all images input to the pose network receive the
        same augmentation.
        """
        return image

