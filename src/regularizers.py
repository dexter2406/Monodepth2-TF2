# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regularizers for depth motion fields. Adapted from
https://github.com/google-research/google-research/tree/master/depth_and_motion_learning
"""

import tensorflow as tf


def joint_bilateral_smoothing(smoothed, reference):
    """Computes edge-aware smoothness loss.

    Args:
      smoothed: A tf.Tensor of shape [B, H, W, C1] to be smoothed.
      reference: A tf.Tensor of the shape [B, H, W, C2]. Wherever `reference` has
        more spatial variation, the strength of the smoothing of `smoothed` will
        be weaker.

    Returns:
      A scalar tf.Tensor containing the regularization, to be added to the
      training loss.
    """
    smoothed_dx = _gradient_x(smoothed)
    smoothed_dy = _gradient_y(smoothed)
    ref_dx = _gradient_x(reference)
    ref_dy = _gradient_y(reference)
    weights_x = tf.exp(-tf.reduce_mean(tf.abs(ref_dx), 3, keepdims=True))
    weights_y = tf.exp(-tf.reduce_mean(tf.abs(ref_dy), 3, keepdims=True))
    smoothness_x = smoothed_dx * weights_x
    smoothness_y = smoothed_dy * weights_y
    return tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(abs(smoothness_y))


def normalize_res_trans(res_trans_field, backgrd_trans):
    """Normalizes a residual motion map by the motion map's norm.
    Args:
        res_trans_field: Tensor, [B,H,W,3]
        backgrd_trans: Tensor, [B,...,3]
    """
    def _expand_to_desired_dims(x, diff_dim):
        for i in range(diff_dim):
            x = tf.expand_dims(x, axis=1)
        return x

    _shape = backgrd_trans.shape
    if _shape.ndims in (2, 3) and _shape[-1] == 3:
        backgrd_trans = _expand_to_desired_dims(backgrd_trans, 4 - _shape.ndims)
    norm = tf.reduce_mean(
        tf.square(backgrd_trans), axis=[1, 2, 3], keep_dims=True) * 3.0
    normed_res_trans = res_trans_field / tf.sqrt(norm + 1e-12)
    return normed_res_trans


def l1smoothness(tensor, wrap_around=True):
    """Calculates L1 (total variation) smoothness loss of a tensor.

    Args:
      tensor: A tensor to be smoothed, of shape [B, H, W, C].
      wrap_around: True to wrap around the last pixels to the first.

    Returns:
      A scalar tf.Tensor, The total variation loss.
    """
    with tf.name_scope('l1smoothness'):
        tensor_dx = tensor - tf.roll(tensor, 1, 1)
        tensor_dy = tensor - tf.roll(tensor, 1, 2)
        # We optionally wrap around in order to impose continuity across the
        # boundary. The motivation is that there is some ambiguity between rotation
        # and spatial gradients of translation maps. We would like to discourage
        # spatial gradients of the translation field, and to absorb such gradients
        # into the rotation as much as possible. This is why we impose continuity
        # across the spatial boundary.
        if not wrap_around:
            tensor_dx = tensor_dx[:, 1:, 1:, :]
            tensor_dy = tensor_dy[:, 1:, 1:, :]
        return tf.reduce_mean(
            tf.sqrt(1e-24 + tf.square(tensor_dx) + tf.square(tensor_dy)))


def sqrt_sparsity(motion_map):
    """A regularizer that encourages sparsity.

    This regularizer penalizes nonzero values. Close to zero it behaves like an L1
    regularizer, and far away from zero its strength decreases. The scale that
    distinguishes "close" from "far" is the mean value of the absolute of
    `motion_map`.

    Args:
       motion_map: A tf.Tensor of shape [B, H, W, C]

    Returns:
       A scalar tf.Tensor, the regularizer to be added to the training loss.
    """
    with tf.name_scope('drift'):
        tensor_abs = tf.abs(motion_map)
        mean = tf.stop_gradient(
            tf.reduce_mean(tensor_abs, axis=[1, 2], keep_dims=True))
        # We used L0.5 norm here because it's more sparsity encouraging than L1.
        # The coefficients are designed in a way that the norm asymptotes to L1 in
        # the small value limit.
        return tf.reduce_mean(2 * mean * tf.sqrt(tensor_abs / (mean + 1e-24) + 1))


def rot_consis_loss(M_12, M_21):
    """calculate pose losses
    Total transformation between two frames should be reverse to each other for pure egomtion
    But if there's moving objects, this fails. But not entirely:
    -> Rotation: the objects' doesn't rotate significantly, so it's reasonable to constraint.
    reversed rotations.
    -> Translation: the main motion of objects, so for now we don't treat it as a reliable constraint

    [ R2,  t2 ]    [ R1,  t1 ]     [ R2R1,  R2t1 + t2 ]
    [         ]  . [         ]  =  [                  ]
    [ 000, 1  ]    [ 000,  1 ]     [ 000,       1     ]

    Args:
        M_12 M_21: transformation mat, including rotation and translation
    Returns:
        rot_consis_loss: rotation consistency loss, calculate by rotation matrix
    """
    R_12, R_21 = M_12[:, :3, :3], M_21[:, :3, :3]  # [B,3,3]
    # R_unit, T_zero = combine_rot_trans_mats(R_12, R_21, T_12, T_21)  # translation calculated as [H,W] field
    R_unit = combine_rot_mats(R_12, R_21)  # rotation as 3x3 matrix, save computation

    # Rotation error
    eye = tf.eye(3, batch_shape=R_12.shape[0])
    rot_error = R_unit - eye
    rot_error = tf.reduce_mean(tf.square(rot_error), axis=(1, 2))
    rot_scale_1 = tf.reduce_mean(tf.square(R_12 - eye), axis=(1, 2))
    rot_scale_2 = tf.reduce_mean(tf.square(R_21 - eye), axis=(1, 2))
    rot_error_normed = tf.reduce_mean(
        rot_error / (1e-24 + rot_scale_1 + rot_scale_2)
    )
    return rot_error_normed


def combine_rot_trans_mats(R_12, R_21, T_12, T_21):
    R_unit = combine_rot_mats(R_12, R_21)
    R2T1 = tf.matmul(R_21, tf.expand_dims(T_12, axis=-1))  # (B,3,1)
    T_zero = R2T1 + tf.expand_dims(T_21, axis=-1)  # (B,3,1)
    return R_unit, T_zero


def combine_rot_mats(R_12, R_21):
    R_unit = tf.matmul(R_21, R_12)  # R2R1, shape (B,3,3)
    return R_unit

def _gradient_x(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]


def _gradient_y(img):
    return img[:, :-1, :, :] - img[:, 1:, :, :]
