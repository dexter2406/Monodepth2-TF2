from abc import ABC

import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model
from collections import defaultdict
import numpy as np
import time

# tf.keras.backend.set_floatx('float16')


class MotionFieldNet(Model):
    def __init__(self, do_automassk=False):
        """Predict object-motion vectors from a stack of frames or embeddings.
        Args:
          images: Input tensor with shape [B, h, w, 2c], containing two
            depth-concatenated images.
          weight_reg: A float scalar, the amount of weight regularization.
          align_corners: align_corners in resize_bilinear. Only used in version 2.
          auto_mask: True to automatically masking out the residual translations
            by thresholding on their mean values.

        Returns:
          A tuple of 3 tf.Tensors:
          rotation: [B, 3], global rotation angles.
          background_translation: [B, 1, 1, 3], global translation vectors.
          residual_translation: [B, h, w, 3], residual translation vector field. The
            overall translation field is "background_translation + residual_translation".
        """
        super(MotionFieldNet, self).__init__()
        conv_prop_0 = {'kernel_size': 3, 'strides': 2, 'activation': 'relu', 'padding': 'same'}
        conv_prop_1 = {'kernel_size': 1, 'strides': 1, 'use_bias': False, 'padding': 'same'}
        self.do_automask = do_automassk
        self.conv_encoder = []
        self.conv_intrinsics = []
        self.rot_scale, self.trans_scale = self.create_scales()

        num_conv_enc = 7
        channels = [6]  # input stacked image
        for i in range(1, num_conv_enc+1):
            channels.append(2**(i+3))
            self.conv_encoder.append(
                Conv2D(channels[-1], **conv_prop_0, name='Conv%d'%i)
            )
        self.conv_backgrd_motion = Conv2D(6, **conv_prop_1, name='backgrd_motion')
        self.conv_res_motion = Conv2D(3, **conv_prop_1, name='unrefined_res_trans')

        # todo: add Dilation to last few convs
        self.conv_refine_motion = []
        for i, ch in enumerate(channels):
            idx = len(channels) - i - 1     # 7 -> 0
            pref = 'Refine%d' % idx
            conv_prop_2 = {'filters': max(4, ch), 'kernel_size': 3, 'strides': 1,
                           'activation': 'relu', 'padding': 'same'}
            tmp = []
            for j in range(3):
                tmp.append(Conv2D(**conv_prop_2, name=pref+'Conv%d'%(j + 1)))
            tmp.append(Conv2D(3, [1, 1], strides=1,
                       activation=None, use_bias=False, name=pref+'Conv4'))
            self.conv_refine_motion.append(tmp)

        conv_prop = {'filters': 2, 'kernel_size': [1, 1], 'strides': 1, 'padding': 'same'}
        self.conv_intrinsics = [
            Conv2D(**conv_prop, activation=tf.nn.softplus, name='Conv_foci'),
            Conv2D(**conv_prop, use_bias=False, name='Conv_offsets')
        ]

    def call(self, x, training=None):
        conv1 = self.conv_encoder[0](x)
        conv2 = self.conv_encoder[1](conv1)
        conv3 = self.conv_encoder[2](conv2)
        conv4 = self.conv_encoder[3](conv3)
        conv5 = self.conv_encoder[4](conv4)
        conv6 = self.conv_encoder[5](conv5)
        conv7 = self.conv_encoder[6](conv6)

        bottleneck = tf.reduce_mean(conv7, axis=[1, 2], keepdims=True)
        background_motion = self.conv_backgrd_motion(bottleneck)

        t1 = time.perf_counter()
        residual_trans = self.conv_res_motion(background_motion)
        print(residual_trans.shape, conv7.shape, "%.2fms"%((time.perf_counter()-t1)*1000))
        residual_trans = self._refine_motion_field(residual_trans, conv7, idx=7)
        print(residual_trans.shape, conv6.shape, "%.2fms"%((time.perf_counter()-t1)*1000))

        residual_trans = self._refine_motion_field(residual_trans, conv6, idx=6)
        print(residual_trans.shape, conv5.shape, "%.2fms"%((time.perf_counter()-t1)*1000))

        residual_trans = self._refine_motion_field(residual_trans, conv5, idx=5, lite_mode=True)
        print(residual_trans.shape, conv4.shape, "%.2fms"%((time.perf_counter()-t1)*1000))

        residual_trans = self._refine_motion_field(residual_trans, conv4, idx=4, lite_mode=True)
        print(residual_trans.shape, conv3.shape, "%.2fms"%((time.perf_counter()-t1)*1000))

        residual_trans = self._refine_motion_field(residual_trans, conv3, idx=3, lite_mode=True)
        print(residual_trans.shape, conv2.shape, "%.2fms"%((time.perf_counter()-t1)*1000))

        residual_trans = self._refine_motion_field(residual_trans, conv2, idx=2, lite_mode=True)
        print(residual_trans.shape, conv1.shape, "%.2fms"%((time.perf_counter()-t1)*1000))

        residual_trans = self._refine_motion_field(residual_trans, conv1, idx=1, lite_mode=True)
        print(residual_trans.shape, x.shape, "%.2fms"%((time.perf_counter()-t1)*1000))
        residual_trans = self._refine_motion_field(residual_trans, x,  idx=0, lite_mode=True)
        print('================================')

        # residual_trans *= self.trans_scale
        rotation = background_motion[:, 0, 0, :3] * self.rot_scale
        backgrd_trans = background_motion[:, :, :, 3:] * self.trans_scale

        if self.do_automask:
            sq_residual_trans = tf.sqrt(
                tf.reduce_sum(residual_trans ** 2, axis=3, keepdims=True))
            mean_sq_residual_trans = tf.reduce_mean(
                sq_residual_trans, axis=[0, 1, 2])
            # A mask of shape [B, h, w, 1]
            mask_residual_translation = tf.cast(
                sq_residual_trans > mean_sq_residual_trans, residual_trans.dtype
            )
            residual_trans *= mask_residual_translation

        image_height, image_width = x.shape[1:3]
        intrinsic_mat = self.add_intrinsics_head(bottleneck, image_height, image_width)
        return rotation, backgrd_trans, residual_trans, intrinsic_mat
        # return rotation, backgrd_trans, residual_trans
        # return rotation, backgrd_trans

    def _refine_motion_field(self, motion_field, conv, idx, lite_mode=False):
        """Refine residual motion map"""
        upsamp_motion_field = tf.cast(tf.image.resize(motion_field, conv.shape[1:3]), dtype=tf.float32)
        conv_inp = tf.concat([upsamp_motion_field, conv], axis=3)
        i = len(self.conv_refine_motion) - 1 - idx  # backwards index in list
        if not lite_mode:
            output_1 = self.conv_refine_motion[i][0](conv_inp)
            conv_inp = self.conv_refine_motion[i][1](conv_inp)
            output_2 = self.conv_refine_motion[i][2](conv_inp)
            conv_inp = tf.concat([output_1, output_2], axis=-1)
        output = self.conv_refine_motion[i][3](conv_inp)
        output = upsamp_motion_field + output
        return output

    def create_scales(self, constraint_min=0.001):
        """Creates variables representing rotation and translation scaling factors.
        Args:
          constraint_min: minimum value for the variable
        Returns:
          Two scalar variables, rotation and translation scale.
        """
        def constraint(x):
            return tf.nn.relu(x - constraint_min) + constraint_min

        rot_scale = tf.Variable(initial_value=0.01, constraint=constraint, dtype=tf.float32)
        trans_scale = tf.Variable(initial_value=0.01, constraint=constraint, dtype=tf.float32)
        return rot_scale, trans_scale

    def add_intrinsics_head(self, bottleneck, image_height, image_width):
        """Adds a head the preficts camera intrinsics.
        Args:
          bottleneck: A tf.Tensor of shape [B, 1, 1, C], typically the bottlenech
            features of a netrowk.
          image_height: A scalar tf.Tensor or an python scalar, the image height in
            pixels.
          image_width: A scalar tf.Tensor or an python scalar, the image width in
            pixels.

        image_height and image_width are used to provide the right scale for the focal
        length and the offest parameters.

        Returns:
          a tf.Tensor of shape [B, 3, 3], and type float32, where the 3x3 part is the
          intrinsic matrix: (fx, 0, x0), (0, fy, y0), (0, 0, 1).
        """
        image_size = tf.constant([[image_width, image_height]], dtype=tf.float32)
        focal_lens = self.conv_intrinsics[0](bottleneck)
        focal_lens = tf.squeeze(focal_lens, axis=(1, 2)) * image_size

        offsets = self.conv_intrinsics[1](bottleneck)
        offsets = (tf.squeeze(offsets, axis=(1, 2)) + 0.5) * image_size

        foc_inv = tf.linalg.diag(focal_lens)
        intrinsic_mat = tf.concat([foc_inv, tf.expand_dims(offsets, -1)], axis=2)
        last_row = tf.cast(tf.tile([[[0.0, 0.0, 1.0]]], [bottleneck.shape[0], 1, 1]), dtype=tf.float32)
        intrinsic_mat = tf.concat([intrinsic_mat, last_row], axis=1)
        return intrinsic_mat


# @tf.function
def run_model(model, dummy_inp):
    outputs = model(dummy_inp)
    # model.build(dummy_inp.shape)
    # model.summary()
    # for i in outputs:
    #     print(i.shape)
    return outputs


def test_speed():
    dummy_inp = tf.random.uniform((8, 192, 640, 8), dtype=tf.float32)
    input_layer = tf.keras.Input(shape=dummy_inp.shape[1:])
    model = MotionFieldNet(True)
    # model.build(dummy_inp.shape)
    for i in range(100):
        t1 =time.perf_counter()
        outputs = model(dummy_inp)
        print("%.2f"%((time.perf_counter() - t1)*1000))
    # model.summary()
    # outputs = model(input_layer)
    # inp = model.input
    # out = model.outputs
    # new_model = Model(input_layer, outputs)
    # new_model.summary()


def test_upsample():
    x = tf.random.uniform((1, 12, 40, 3))
    t1 = time.perf_counter()
    for i in range(1000):
        if i % 100 == 0: print('-')
        # y = tf.keras.layers.UpSampling2D()(x)
        y = tf.image.resize(x, (24, 80))
    print("%.2f"%((time.perf_counter()-t1)*1000))
    print(y.shape)


if __name__ == '__main__':
    test_speed()
    # test_upsample()



# class MotionFieldNet(Model):
#     def __init__(self, do_automassk=False):
#         """Predict object-motion vectors from a stack of frames or embeddings.
#         Args:
#           images: Input tensor with shape [B, h, w, 2c], containing two
#             depth-concatenated images.
#           weight_reg: A float scalar, the amount of weight regularization.
#           align_corners: align_corners in resize_bilinear. Only used in version 2.
#           auto_mask: True to automatically masking out the residual translations
#             by thresholding on their mean values.
#
#         Returns:
#           A tuple of 3 tf.Tensors:
#           rotation: [B, 3], global rotation angles.
#           background_translation: [B, 1, 1, 3], global translation vectors.
#           residual_translation: [B, h, w, 3], residual translation vector field. The
#             overall translation field is "background_translation + residual_translation".
#         """
#         super(MotionFieldNet, self).__init__()
#         conv_prop_0 = {'kernel_size': 3, 'strides': 2, 'activation': 'relu', 'padding': 'same'}
#         conv_prop_1 = {'kernel_size': 1, 'strides': 1, 'use_bias': False, 'padding': 'same'}
#         self.do_automask = do_automassk
#         self.conv_encoder = []
#         self.conv_intrinsics = []
#         self.rot_scale, self.trans_scale = self.create_scales()
#
#         num_conv_enc = 7
#         channels = [6]  # input stacked image
#         for i in range(1, num_conv_enc+1):
#             channels.append(2**(i+3))
#             self.conv_encoder.append(
#                 Conv2D(channels[-1], **conv_prop_0, name='Conv%d'%i)
#             )
#         self.conv_backgrd_motion = Conv2D(6, **conv_prop_1, name='backgrd_motion')
#         self.conv_res_motion = Conv2D(3, **conv_prop_1, name='unrefined_res_trans')
#
#         self.conv_refine_motion = []
#         for i, ch in enumerate(channels):
#             idx = len(channels) - i - 1     # 7 -> 0
#             pref = 'Refine%d' % idx
#             conv_prop_2 = {'filters': max(4, ch), 'kernel_size': 3, 'strides': 1,
#                            'activation': 'relu', 'padding': 'same'}
#             tmp = []
#             for j in range(3):
#                 tmp.append(Conv2D(**conv_prop_2, name=pref+'Conv%d'%(j + 1)))
#             tmp.append(Conv2D(3, [1, 1], strides=1,
#                        activation=None, use_bias=False, name=pref+'Conv4'))
#             self.conv_refine_motion.append(tmp)
#
#         conv_prop = {'filters': 2, 'kernel_size': [1, 1], 'strides': 1, 'padding': 'same'}
#         self.conv_intrinsics = [
#             Conv2D(**conv_prop, activation=tf.nn.softplus, name='Conv_foci'),
#             Conv2D(**conv_prop, use_bias=False, name='Conv_offsets')
#         ]
#
#     def call(self, x, training=None):
#         t1 = time.perf_counter()
#         conv1 = self.conv_encoder[0](x)
#         conv2 = self.conv_encoder[1](conv1)
#         conv3 = self.conv_encoder[2](conv2)
#         conv4 = self.conv_encoder[3](conv3)
#         conv5 = self.conv_encoder[4](conv4)
#         conv6 = self.conv_encoder[5](conv5)
#         conv7 = self.conv_encoder[6](conv6)
#         print("%.2fms" % ((time.perf_counter() - t1) * 1000))
#
#         t1 = time.perf_counter()
#         bottleneck = tf.reduce_mean(conv7, axis=[1, 2], keepdims=True)
#         background_motion = self.conv_backgrd_motion(bottleneck)
#         print("%.2fms" % ((time.perf_counter() - t1) * 1000))
#
#         t1 = time.perf_counter()
#         residual_trans = self.conv_res_motion(background_motion)
#         print(residual_trans.shape, conv7.shape, "%.2fms"%((time.perf_counter()-t1)*1000))
#
#         t1 = time.perf_counter()
#         # residual_trans = self._refine_motion_field(residual_trans, conv7, idx=7)
#         upsamp_motion_field = tf.image.resize(residual_trans, conv7.shape[1:3])
#         conv_inp = tf.concat([upsamp_motion_field, conv7], axis=3)
#         i = 0  # backwards index in list
#         output_1 = self.conv_refine_motion[i][0](conv_inp)
#         conv_inp = self.conv_refine_motion[i][1](conv_inp)
#         output_2 = self.conv_refine_motion[i][2](conv_inp)
#         output = tf.concat([output_1, output_2], axis=-1)
#         output = self.conv_refine_motion[i][3](output)
#         # output = self.conv_refine_motion[i][3](conv_inp)
#         residual_trans = upsamp_motion_field + output
#         print(residual_trans.shape, conv6.shape, "%.2fms"%((time.perf_counter()-t1)*1000))
#
#         # tmp = np.copy(residual_trans)
#         # for i in range(10):
#         t1 = time.perf_counter()
#         # residual_trans = self._refine_motion_field(residual_trans, conv6, idx=6)
#         upsamp_motion_field = tf.image.resize(residual_trans, conv6.shape[1:3])
#         conv_inp = tf.concat([upsamp_motion_field, conv6], axis=3)
#         i = 1 # backwards index in list
#         output_1 = self.conv_refine_motion[i][0](conv_inp)
#         conv_inp = self.conv_refine_motion[i][1](conv_inp)
#         output_2 = self.conv_refine_motion[i][2](conv_inp)
#         output = tf.concat([output_1, output_2], axis=-1)
#         output = self.conv_refine_motion[i][3](output)
#         # output = self.conv_refine_motion[i][3](conv_inp)
#         residual_trans = upsamp_motion_field + output
#         print("*",residual_trans.shape, conv5.shape, "%.2fms"%((time.perf_counter()-t1)*1000))
#
#         t1 = time.perf_counter()
#         # residual_trans = self._refine_motion_field(residual_trans, conv5, idx=5)
#         upsamp_motion_field = tf.image.resize(residual_trans, conv5.shape[1:3])
#         conv_inp = tf.concat([upsamp_motion_field, conv5], axis=3)
#         i = 2  # backwards index in list
#         output_1 = self.conv_refine_motion[i][0](conv_inp)
#         conv_inp = self.conv_refine_motion[i][1](conv_inp)
#         output_2 = self.conv_refine_motion[i][2](conv_inp)
#         output = tf.concat([output_1, output_2], axis=-1)
#         output = self.conv_refine_motion[i][3](output)
#         # output = self.conv_refine_motion[i][3](conv_inp)
#         residual_trans = upsamp_motion_field + output
#         print("*", residual_trans.shape, conv4.shape, "%.2fms"%((time.perf_counter()-t1)*1000))
#
#         t1 = time.perf_counter()
#         # residual_trans = self._refine_motion_field(residual_trans, conv4, idx=4)
#         upsamp_motion_field = tf.image.resize(residual_trans, conv4.shape[1:3])
#         conv_inp = tf.concat([upsamp_motion_field, conv4], axis=3)
#         i = 3  # backwards index in list
#         output_1 = self.conv_refine_motion[i][0](conv_inp)
#         conv_inp = self.conv_refine_motion[i][1](conv_inp)
#         output_2 = self.conv_refine_motion[i][2](conv_inp)
#         output = tf.concat([output_1, output_2], axis=-1)
#         output = self.conv_refine_motion[i][3](output)
#         # output = self.conv_refine_motion[i][3](conv_inp)
#         residual_trans = upsamp_motion_field + output
#         print("*", residual_trans.shape, conv3.shape, "%.2fms"%((time.perf_counter()-t1)*1000))
#
#         t1 = time.perf_counter()
#         # residual_trans = self._refine_motion_field(residual_trans, conv3, idx=3)
#         upsamp_motion_field = tf.image.resize(residual_trans, conv3.shape[1:3])
#         conv_inp = tf.concat([upsamp_motion_field, conv3], axis=3)
#         i = 4 # backwards index in list
#         # output_1 = self.conv_refine_motion[i][0](conv_inp)
#         # conv_inp = self.conv_refine_motion[i][1](conv_inp)
#         # output_2 = self.conv_refine_motion[i][2](conv_inp)
#         # output = tf.concat([output_1, output_2], axis=-1)
#         # output = self.conv_refine_motion[i][3](output)
#         output = self.conv_refine_motion[i][3](conv_inp)
#         residual_trans = upsamp_motion_field + output
#         print(residual_trans.shape, conv2.shape, "%.2fms"%((time.perf_counter()-t1)*1000))
#
#         t1 = time.perf_counter()
#         # residual_trans = self._refine_motion_field(residual_trans, conv2, idx=2)
#         upsamp_motion_field = tf.image.resize(residual_trans, conv2.shape[1:3])
#         conv_inp = tf.concat([upsamp_motion_field, conv2], axis=3)
#         i = 5  # backwards index in list
#         # output_1 = self.conv_refine_motion[i][0](conv_inp)
#         # conv_inp = self.conv_refine_motion[i][1](conv_inp)
#         # output_2 = self.conv_refine_motion[i][2](conv_inp)
#         # output = tf.concat([output_1, output_2], axis=-1)
#         # output = self.conv_refine_motion[i][3](output)
#         output = self.conv_refine_motion[i][3](conv_inp)
#         residual_trans = upsamp_motion_field + output
#         print(residual_trans.shape, conv1.shape, "%.2fms"%((time.perf_counter()-t1)*1000))
#
#         residual_trans = tf.image.resize(residual_trans, x.shape[1:3])
#
#         t1 = time.perf_counter()
#         # residual_trans = self._refine_motion_field(residual_trans, conv1, idx=1)
#         upsamp_motion_field = tf.image.resize(residual_trans, conv1.shape[1:3])
#         conv_inp = tf.concat([upsamp_motion_field, conv1], axis=3)
#         i = 6  # backwards index in list
#         # output_1 = self.conv_refine_motion[i][0](conv_inp)
#         # conv_inp = self.conv_refine_motion[i][1](conv_inp)
#         # output_2 = self.conv_refine_motion[i][2](conv_inp)
#         # output = tf.concat([output_1, output_2], axis=-1)
#         # output = self.conv_refine_motion[i][3](output)
#         output = self.conv_refine_motion[i][3](conv_inp)
#         residual_trans = upsamp_motion_field + output
#         print(residual_trans.shape, x.shape, "%.2fms"%((time.perf_counter()-t1)*1000))
#
#         t1 = time.perf_counter()
#         # residual_trans = self._refine_motion_field(residual_trans, x,  idx=0)
#         upsamp_motion_field = tf.image.resize(residual_trans, x.shape[1:3])
#         # upsamp_motion_field = tf.image.resize(residual_trans, x.shape[1:3])
#         conv_inp = tf.concat([upsamp_motion_field, x], axis=3)
#         i = 7  # backwards index in list
#         # output_1 = self.conv_refine_motion[i][0](conv_inp)
#         # conv_inp = self.conv_refine_motion[i][1](conv_inp)
#         # output_2 = self.conv_refine_motion[i][2](conv_inp)
#         # output = tf.concat([output_1, output_2], axis=-1)
#         # output = self.conv_refine_motion[i][3](output)
#
#         output = self.conv_refine_motion[i][3](conv_inp)
#         residual_trans = upsamp_motion_field + output
#         print(residual_trans.shape, x.shape, "%.2fms"%((time.perf_counter()-t1)*1000))
#
#         residual_trans *= self.trans_scale
#         rotation = background_motion[:, 0, 0, :3] * self.rot_scale
#         backgrd_trans = background_motion[:, :, :, 3:] * self.trans_scale
#         t1 = time.perf_counter()
#         if self.do_automask:
#             sq_residual_trans = tf.sqrt(
#                 tf.reduce_sum(residual_trans ** 2, axis=3, keepdims=True))
#             mean_sq_residual_trans = tf.reduce_mean(
#                 sq_residual_trans, axis=[0, 1, 2])
#             # A mask of shape [B, h, w, 1]
#             mask_residual_translation = tf.cast(
#                 sq_residual_trans > mean_sq_residual_trans, residual_trans.dtype
#             )
#             residual_trans *= mask_residual_translation
#         print("%.2fms" % ((time.perf_counter() - t1) * 1000))
#         t1 = time.perf_counter()
#         image_height, image_width = x.shape[1:3]
#         # intrinsic_mat = self.add_intrinsics_head(bottleneck, image_height, image_width)
#         print("%.2fms" % ((time.perf_counter() - t1) * 1000))
#         print('================================')
#         # return rotation, backgrd_trans, residual_trans, intrinsic_mat
#         # return rotation, backgrd_trans, residual_trans
#         return rotation, backgrd_trans
#
#     def _refine_motion_field(self, motion_field, conv, idx):
#         """Refine residual motion map"""
#         upsamp_motion_field = tf.image.resize(motion_field, conv.shape[1:3])
#         conv_inp = tf.concat([upsamp_motion_field, conv], axis=3)
#         i = 7 - idx  # backwards index in list
#         output_1 = self.conv_refine_motion[i][0](conv_inp)
#         conv_inp = self.conv_refine_motion[i][1](conv_inp)
#         output_2 = self.conv_refine_motion[i][2](conv_inp)
#
#         output = tf.concat([output_1, output_2], axis=-1)
#         output = self.conv_refine_motion[i][3](output)
#         output = upsamp_motion_field + output
#         return output
#
#     def create_scales(self, constraint_min=0.001):
#         """Creates variables representing rotation and translation scaling factors.
#         Args:
#           constraint_min: minimum value for the variable
#         Returns:
#           Two scalar variables, rotation and translation scale.
#         """
#         def constraint(x):
#             return tf.nn.relu(x - constraint_min) + constraint_min
#
#         rot_scale = tf.Variable(initial_value=0.01, constraint=constraint)
#         trans_scale = tf.Variable(initial_value=0.01, constraint=constraint)
#         return rot_scale, trans_scale
#
#     def add_intrinsics_head(self, bottleneck, image_height, image_width):
#         """Adds a head the preficts camera intrinsics.
#         Args:
#           bottleneck: A tf.Tensor of shape [B, 1, 1, C], typically the bottlenech
#             features of a netrowk.
#           image_height: A scalar tf.Tensor or an python scalar, the image height in
#             pixels.
#           image_width: A scalar tf.Tensor or an python scalar, the image width in
#             pixels.
#
#         image_height and image_width are used to provide the right scale for the focal
#         length and the offest parameters.
#
#         Returns:
#           a tf.Tensor of shape [B, 3, 3], and type float32, where the 3x3 part is the
#           intrinsic matrix: (fx, 0, x0), (0, fy, y0), (0, 0, 1).
#         """
#         image_size = tf.constant([[image_width, image_height]], dtype=tf.float32)
#         focal_lens = self.conv_intrinsics[0](bottleneck)
#         focal_lens = tf.squeeze(focal_lens, axis=(1, 2)) * image_size
#
#         offsets = self.conv_intrinsics[1](bottleneck)
#         offsets = (tf.squeeze(offsets, axis=(1, 2)) + 0.5) * image_size
#
#         foc_inv = tf.linalg.diag(focal_lens)
#         intrinsic_mat = tf.concat([foc_inv, tf.expand_dims(offsets, -1)], axis=2)
#         last_row = tf.tile([[[0.0, 0.0, 1.0]]], [bottleneck.shape[0], 1, 1])
#         intrinsic_mat = tf.concat([intrinsic_mat, last_row], axis=1)
#         return intrinsic_mat



