import tensorflow as tf
import os
import time
import math
import numpy as np
import sys


class DataProcessor(object):
    def __init__(self, frame_idx, intrinsics=None, is_train=False, num_scales=4):
        self.is_train = is_train
        self.num_scales = num_scales     # for evaluation, 1
        self.height = 192
        self.width = 640
        self.frame_idx = frame_idx
        self.batch_size = -1
        self.K = intrinsics

    @tf.function
    def prepare_batch(self, batch):
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
        # do_color_aug = self.is_train and np.random.random() > 0.5
        # do_flip = False
        # tf.print('train shape', batch[0].shape, batch[1].shape)
        input_imgs, input_Ks = self.process_batch_main(batch)
        return input_imgs, input_Ks

    @tf.function
    def prepare_batch_val(self, batch, depth_batch):
        """duplicate of prepare_batch(), no @tf.function decorator
        For validation OR evaluation
        """
        # tf.print('val shape')
        # tf.print(batch[0].shape, batch[1].shape, output_stream=sys.stdout)
        input_imgs, input_Ks = self.process_batch_main(batch, depth_batch)
        return input_imgs, input_Ks

    def process_batch_main(self, batch):
        input_imgs, input_Ks = {}, {}
        if type(batch) == tuple:
            batch_imgs = batch[0]
            input_imgs[('depth_gt', 0)] = tf.expand_dims(batch[1], 3)
        else:
            batch_imgs = batch
        tgt_batch, src_batch = batch_imgs[..., :3], batch_imgs[..., 3:]
        self.batch_size = tgt_batch.shape[0]

        # in training / eval_depth / eval_pose
        # frame_idx == [0,-1,1] / [0] / [0,1] respectively
        input_imgs[('color', self.frame_idx[0], -1)] = tgt_batch
        if len(self.frame_idx) > 1:
            for i, f_i in enumerate(self.frame_idx[1:]):
                input_imgs[('color', f_i, -1)] = src_batch[..., i*3: (i+1)*3]

        input_imgs, input_Ks = self.preprocess(input_imgs, input_Ks)
        self.delete_raw_images(input_imgs)
        return input_imgs, input_Ks

    def preprocess(self, input_imgs, input_Ks):
        """Make pyramids and augmentations
        - pyramid: use the raw (scale=-=1) to produce scale==[0:4] images
        - augment: correspond to the pyramid source
        """
        # print("\t preprocessing batch...")
        for k in list(input_imgs):
            if 'depth_gt' in k:
                continue
            img_type, f_i, scale = k    # key components
            if img_type == 'color':
                for scale in range(self.num_scales):
                    src_image = input_imgs[(img_type, f_i, scale - 1)]
                    resized_image = tf.image.resize(src_image,
                                                    (self.height//(2**scale), self.width//(2**scale)),
                                                    antialias=True)
                    input_imgs[(img_type, f_i, scale)] = resized_image
                    input_imgs[(img_type + '_aug', f_i, scale)] = self.color_aug(resized_image)

        for scale in range(self.num_scales):
            # For KITTI. Must *normalize* when using on different scale
            K = self.K.copy()
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            K = tf.reshape(tf.tile(K, [self.batch_size,1]), (self.batch_size, 4, 4))
            inv_K = tf.reshape(tf.tile(inv_K, [self.batch_size,1]), (self.batch_size, 4, 4))
            assert K.shape[-1] == 4

            input_Ks[("K", scale)] = K
            input_Ks[("inv_K", scale)] = inv_K

        return input_imgs, input_Ks

    def delete_raw_images(self, input_imgs, scale_to_del=-1):
        for idx in self.frame_idx:
            del input_imgs[('color', self.frame_idx[idx], scale_to_del)]

    def color_aug(self, image):
        # todo: implement color augmentations
        """Apply augmentation (fixed seed needed)
        Same aug for each batch (and scales), Note that all images input to the pose network receive the
        same augmentation.
        """
        return image

