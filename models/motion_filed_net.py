import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model
from collections import defaultdict
import numpy as np
import time
from utils import make_hom_intrinsics

# tf.keras.backend.set_floatx('float16')


class MotionFieldEncoder(Model):
    def __init__(self, weight_reg=0.0):
        """Predict object-motion vectors from a stack of frames.
        auto_mask: True to automatically masking out the residual translations
            by thresholding on their mean values.
        weight_reg: A float scalar, the amount of weight regularization.
        """
        super(MotionFieldEncoder, self).__init__()
        conv_prop_0 = {'kernel_size': 3, 'strides': 2, 'activation': 'relu', 'padding': 'same',
                       'kernel_regularizer': tf.keras.regularizers.L2(weight_reg)}
        self.conv_encoder = []

        num_conv_enc = 7
        channels = [6]  # input stacked image
        for i in range(1, num_conv_enc+1):
            channels.append(2**(i+3))
            self.conv_encoder.append(
                Conv2D(channels[-1], **conv_prop_0, name='Conv%d'%i)
            )

    def call(self, x, training=None, mask=None):
        """
        Args:
          x: Input tensor with shape [B, h, w, 2c], `c` can be rgb or rgb-d.

        Returns:
          list of features, last one is bottleneck squeezed from Conv7
        """
        features = [x]
        for i in range(len(self.conv_encoder)):
            features.append(self.conv_encoder[i](x))
        bottleneck = tf.reduce_mean(features[-1], axis=[1, 2], keepdims=True)
        features.append(bottleneck)
        return features


class MotionFieldDecoder(Model):
    def __init__(self,do_automassk=True, include_res_trans=True):
        super(MotionFieldDecoder, self).__init__()
        self.include_res_trans = include_res_trans
        self.do_automask = do_automassk
        self.rot_scale = tf.constant(0.01)
        self.trans_scale = tf.constant(0.01)
        conv_prop_1 = {'kernel_size': 1, 'strides': 1, 'use_bias': False, 'padding': 'same'}
        num_conv_enc = 7
        channels = [6]  # should be identical to encoder
        for i in range(1, num_conv_enc + 1):
            channels.append(2 ** (i + 3))
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
                tmp.append(Conv2D(**conv_prop_2, name=pref + 'Conv%d' % (j + 1)))
            tmp.append(Conv2D(3, [1, 1], strides=1,
                              activation=None, use_bias=False, name=pref + 'Conv4'))
            self.conv_refine_motion.append(tmp)

    def call(self, inputs, training=None, mask=None):
        features, bottleneck = inputs[:-1], inputs[-1]
        background_motion = self.conv_backgrd_motion(bottleneck)

        rotation = background_motion[:, :, :, :3] * self.rot_scale          # B,1,1,3
        backgrd_trans = background_motion[:, :, :, 3:] * self.trans_scale   # B,1,1,3

        residual_trans = None
        if self.include_res_trans:
            residual_trans = self.conv_res_motion(background_motion)

            lite_mode_idx = list(range(0, 6))  # 0->5 uses lite_mode to save computation
            for i in range(len(features)-1, -1, -1):  # 7->0
                # t1 = time.perf_counter()
                use_lite_mode = i in lite_mode_idx
                # print('idx=%d uses lite_mode' % i)
                residual_trans = self._refine_motion_field(residual_trans, features[i],
                                                           idx=i, lite_mode=use_lite_mode)
                # print(residual_trans.shape, features[i].shape, "%.2fms" % ((time.perf_counter() - t1) * 1000))

            residual_trans *= self.trans_scale
            if self.do_automask:
                residual_trans = self.apply_automask_to_res_trans(residual_trans)

        outputs = {"rotation": rotation, "backgrd_trans": backgrd_trans, "residual_trans": residual_trans}
        return outputs

    def apply_automask_to_res_trans(self, residual_trans):
        """Masking out the residual translations by thresholding on their mean values."""
        sq_residual_trans = tf.sqrt(
            tf.reduce_sum(residual_trans ** 2, axis=3, keepdims=True))
        mean_sq_residual_trans = tf.reduce_mean(
            sq_residual_trans, axis=[0, 1, 2])
        # A mask of shape [B, h, w, 1]
        mask_residual_translation = tf.cast(
            sq_residual_trans > mean_sq_residual_trans, residual_trans.dtype
        )
        residual_trans *= mask_residual_translation
        return residual_trans

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
        else:
            # use lite mode to save computation
            conv_inp = self.conv_refine_motion[i][0](conv_inp)
        output = self.conv_refine_motion[i][3](conv_inp)
        output = upsamp_motion_field + output
        return output


class IntrinsicsHead(Model):
    def __init__(self, image_size):
        super(IntrinsicsHead, self).__init__()
        self.image_size = list(image_size)    # (H,W)
        conv_prop = {'filters': 2, 'kernel_size': [1, 1], 'strides': 1, 'padding': 'same'}
        self.conv_intrinsics = [
            Conv2D(**conv_prop, activation=tf.nn.softplus, name='Conv_foci'),
            Conv2D(**conv_prop, use_bias=False, name='Conv_offsets')
        ]

    def call(self, inputs, training=None, mask=None):
        """Adds a head the preficts camera intrinsics.
        Args:
          bottleneck: A tf.Tensor of shape [B, H, W, C]
            will be reduce_mean to [B, 1, 1, C] before input

        -> self.image_size are used to provide the right scale for the focal
        length and the offest parameters.
        -> Loss for Intrinsics:
            1) Intrinsics should be identical regardless of the temporal order
            2) optional, could be identical, if only one vidoe/dataset
        -> But for simplicity we could just use the average of
            1) the fwd & bwd matrices,
            2) optional, the whole batch and replicate to the same batch num

        Returns:
          a tf.Tensor of shape [B, 2, 3], and type float32, where the 3x3 part is the
          intrinsic matrix: (fx, 0, x0), (0, fy, y0).
          The full homogenous matrix of shape [B, 4, 4] will be made outside the network
      """
        bottleneck = tf.reduce_mean(inputs, [1, 2], keepdims=True)
        image_size = tf.constant([self.image_size], dtype=tf.float32)
        focal_lens = self.conv_intrinsics[0](bottleneck)
        focal_lens = tf.squeeze(focal_lens, axis=(1, 2)) * image_size
        offsets = self.conv_intrinsics[1](bottleneck)
        offsets = (tf.squeeze(offsets, axis=(1, 2)) + 0.5) * image_size

        foc_diag = tf.linalg.diag(focal_lens)
        intrinsic_mat = tf.concat([foc_diag, tf.expand_dims(offsets, -1)], axis=2)
        return intrinsic_mat


# class EgoMotionNet(Model, MotionFieldEncoder):
#     def __init__(self, add_intrinsics):
#         """A child of MotionField Net only for egomotion
#         - no residual_translation included
#         - optional add_intrinsics_head
#         """
#         super().__init__()
#         self.addintrinscs = add_intrinsics
#
#     def call(self, x):
#         features = [x]
#         for i in range(len(self.conv_encoder)):
#             features.append(self.conv_encoder[i](x))
#
#         bottleneck = tf.reduce_mean(features[-1], axis=[1, 2], keepdims=True)
#         background_motion = self.conv_backgrd_motion(bottleneck)
#         rotation = background_motion[:, 0, 0, :3] * self.rot_scale
#         backgrd_trans = background_motion[:, :, :, 3:] * self.trans_scale
#
#         intrinsic_mat = tf.constant(0)
#         if self.addintrinscs:
#             image_height, image_width = x.shape[1:3]
#             intrinsic_mat = self.add_intrinsics_head(bottleneck, image_height, image_width)
#         return rotation, backgrd_trans, intrinsic_mat


# @tf.function
def run_model(models, dummy_inp):
    features = models['pose_enc'](dummy_inp)
    transformations = models['pose_dec'](features)
    K = models['intrinsics_head'](features[-1])
    return transformations, K


def get_models():
    dummy_inp = tf.random.uniform((3, 192, 640, 8), dtype=tf.float32)
    models = {
        'pose_enc': MotionFieldEncoder(),
        'pose_dec': MotionFieldDecoder(include_res_trans=False),
        'intrinsics_head': IntrinsicsHead(dummy_inp.shape[1:3])
    }
    return models, dummy_inp


def test_speed():
    models, dummy_inp = get_models()
    input_layer = tf.keras.Input(shape=dummy_inp.shape[1:])
    for i in range(100):
        t1 =time.perf_counter()
        run_model(models, dummy_inp)
        print("%.2f"%((time.perf_counter() - t1)*1000))


# def test_upsample():
#     x = tf.random.uniform((1, 12, 40, 3))
#     t1 = time.perf_counter()
#     for i in range(1000):
#         if i % 100 == 0: print('-')
#         # y = tf.keras.layers.UpSampling2D()(x)
#         y = tf.image.resize(x, (24, 80))
#     print("%.2f"%((time.perf_counter()-t1)*1000))
#     print(y.shape)

def check_outputs():
    transforms, K = run_model(*get_models())
    for k, v in transforms.items():
        if v is not None:
            print('{}, shape: {}'.format(k, v.shape))
    print('partial intrinsics:', K.shape)
    print('homogenous K:', make_hom_intrinsics(K, same_video=True).shape)


if __name__ == '__main__':
    # test_speed()
    # test_upsample()
    check_outputs()


"""Encoder-decoder"""
#
# class MotionFieldNet(Model):
#     def __init__(self, do_automassk=False, weight_reg=0.0):
#         """Predict object-motion vectors from a stack of frames.
#         auto_mask: True to automatically masking out the residual translations
#             by thresholding on their mean values.
#         weight_reg: A float scalar, the amount of weight regularization.
#         """
#         super(MotionFieldNet, self).__init__()
#         conv_prop_0 = {'kernel_size': 3, 'strides': 2, 'activation': 'relu', 'padding': 'same',
#                        'kernel_regularizer': tf.keras.regularizers.L2(weight_reg)}
#         conv_prop_1 = {'kernel_size': 1, 'strides': 1, 'use_bias': False, 'padding': 'same'}
#         self.do_automask = do_automassk
#         self.conv_encoder = []
#         self.conv_intrinsics = []
#         self.rot_scale = tf.constant(0.01)
#         self.trans_scale = tf.constant(0.01)
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
#         # todo: add Dilation to last few convs
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
#         """
#         Args:
#           x: Input tensor with shape [B, h, w, 2c], `c` can be rgb or rgb-d.
#
#         Returns:
#           A tuple of 3 tf.Tensors:
#           rotation: [B, 3], global rotation angles.
#           background_translation: [B, 1, 1, 3], global translation vectors.
#           residual_translation: [B, h, w, 3], residual translation vector field. The
#             overall translation field is "background_translation + residual_translation".
#         """
#         features = [x]
#         for i in range(len(self.conv_encoder)):
#             features.append(self.conv_encoder[i](x))
#         bottleneck = tf.reduce_mean(features[-1], axis=[1, 2], keepdims=True)
#         features.append(bottleneck)
#
#         background_motion = self.conv_backgrd_motion(bottleneck)
#         residual_trans = self.conv_res_motion(background_motion)
#
#         lite_mode_idx = list(range(0, 6))  # 0->5 uses lite_mode to save computation
#         for i in range(len(features[:-1]) - 1, -1, -1):  # 7->0
#             t1 = time.perf_counter()
#             use_lite_mode = i in lite_mode_idx
#             print('idx=%d uses lite_mode' % i)
#             residual_trans = self._refine_motion_field(residual_trans, features[i],
#                                                        idx=i, lite_mode=use_lite_mode)
#             print(residual_trans.shape, features[i].shape, "%.2fms" % ((time.perf_counter() - t1) * 1000))
#
#         residual_trans *= self.trans_scale
#         if self.do_automask:
#             residual_trans = self.apply_automask_to_res_trans(residual_trans)
#
#         rotation = background_motion[:, :, :, :3] * self.rot_scale
#         backgrd_trans = background_motion[:, :, :, 3:] * self.trans_scale
#
#         image_height, image_width = x.shape[1:3]
#         intrinsic_mat = self.add_intrinsics_head(bottleneck, image_height, image_width)
#         return rotation, backgrd_trans, residual_trans, intrinsic_mat
#
#     def apply_automask_to_res_trans(self, residual_trans):
#         """Masking out the residual translations by thresholding on their mean values."""
#         sq_residual_trans = tf.sqrt(
#             tf.reduce_sum(residual_trans ** 2, axis=3, keepdims=True))
#         mean_sq_residual_trans = tf.reduce_mean(
#             sq_residual_trans, axis=[0, 1, 2])
#         # A mask of shape [B, h, w, 1]
#         mask_residual_translation = tf.cast(
#             sq_residual_trans > mean_sq_residual_trans, residual_trans.dtype
#         )
#         residual_trans *= mask_residual_translation
#         return residual_trans
#
#     def _refine_motion_field(self, motion_field, conv, idx, lite_mode=False):
#         """Refine residual motion map"""
#         upsamp_motion_field = tf.cast(tf.image.resize(motion_field, conv.shape[1:3]), dtype=tf.float32)
#         conv_inp = tf.concat([upsamp_motion_field, conv], axis=3)
#         i = len(self.conv_refine_motion) - 1 - idx  # backwards index in list
#         if not lite_mode:
#             output_1 = self.conv_refine_motion[i][0](conv_inp)
#             conv_inp = self.conv_refine_motion[i][1](conv_inp)
#             output_2 = self.conv_refine_motion[i][2](conv_inp)
#             conv_inp = tf.concat([output_1, output_2], axis=-1)
#         else:
#             # use lite mode to save computation
#             conv_inp = self.conv_refine_motion[i][0](conv_inp)
#         output = self.conv_refine_motion[i][3](conv_inp)
#         output = upsamp_motion_field + output
#         return output
#
#     def add_intrinsics_head(self, bottleneck, image_height, image_width):
#         """Adds a head the preficts camera intrinsics.
#         Args:
#           bottleneck: A tf.Tensor of shape [B, 1, 1, C]
#           image_height: A scalar tf.Tensor or an python scalar, the image height in pixels.
#           image_width: the image width
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
#         last_row = tf.cast(tf.tile([[[0.0, 0.0, 1.0]]], [bottleneck.shape[0], 1, 1]), dtype=tf.float32)
#         intrinsic_mat = tf.concat([intrinsic_mat, last_row], axis=1)
#         return intrinsic_mat




