import tensorflow as tf
from src.trainer_helper import rot_from_axisangle


def get_transformed_depth_coords(depth, translation, rot_angles, intrinsic_mat):
    """
    depth: [B,H,W]
    intrinsic_mat: [B,3,3]
    rot_angles: [B, 3]
    translation: shape [B,1,1,3]
    """
    # print("rot_angles",rot_angles)
    depth = tf.squeeze(depth, axis=-1)
    if translation.shape.ndims not in (2, 4):
        raise ValueError('\'translation\' should have rank 2 or 4, not %d' %
                         translation.shape.ndims)
    if translation.shape[-1] != 3:
        raise ValueError('translation\'s last dimension should be 3, not %d' %
                         translation.shape[1])
    if translation.shape.ndims == 2:
        translation = tf.expand_dims(tf.expand_dims(translation, 1), 1)
    if translation.shape.ndims == 3:
        translation = tf.expand_dims(translation, 1)
    if rot_angles.shape.ndims == 2:
        rot_angles = tf.expand_dims(rot_angles, 1)  # B,1,3
    if rot_angles.shape.ndims == 4:
        rot_angles = tf.squeeze(rot_angles, 1)
    else:
        raise ValueError('rot angles dimension wrong:', rot_angles.shape)
    if intrinsic_mat.shape[1] == intrinsic_mat.shape[2] == 4:
        intrinsic_mat = intrinsic_mat[:, :3, :3]

    _, height, width = tf.unstack(tf.shape(depth))
    grid = tf.squeeze(
        tf.stack(tf.meshgrid(tf.range(width), tf.range(height), (1,))), axis=3)
    grid = tf.cast(grid, tf.float32)
    intrinsic_mat_inv = tf.linalg.inv(intrinsic_mat)

    rot_mat = rot_from_axisangle(rot_angles)[:, :3, :3]
    # We have to treat separately the case of a per-image rotation vector and a
    # per-image rotation field, because the broadcasting capabilities of einsum
    # are limited.
    # IOdentical to the one in inverse_warp. But `einsum` performs
    # the reshaping and invocation of BatchMatMul, instead of doing it manually,
    # as in inverse_warp.
    projected_rotation = tf.einsum('bij,bjk,bkl->bil', intrinsic_mat, rot_mat,
                                   intrinsic_mat_inv)
    pcoords = tf.einsum('bij,jhw,bhw->bihw', projected_rotation, grid, depth)

    projected_translation = tf.einsum('bij,bhwj->bihw', intrinsic_mat,
                                      translation)
    pcoords += projected_translation
    x, y, z = tf.unstack(pcoords, axis=1)
    return x / z, y / z, z


def resampler(data, warp_x, warp_y, safe=True):
    """Resamples input data at user defined coordinates.
    *Second part of the original Bilinear Sampler

    Args:
      data: Tensor of shape `[batch_size, data_height, data_width,
        data_num_channels]` containing 2D data that will be resampled.
      warp_x: Tensor of shape `[batch_size, dim_0, ... , dim_n]` containing the x
        coordinates at which resampling will be performed.
      warp_y: Tensor of the same shape as warp_x containing the y coordinates at
        which resampling will be performed.
      safe: A boolean, if True, warp_x and warp_y will be clamped to their bounds.
        Disable only if you know they are within bounds, otherwise a runtime
        exception will be thrown.
      name: Optional name of the op.

    Returns:
       Tensor of resampled values from `data`. The output tensor shape is
      `[batch_size, dim_0, ... , dim_n, data_num_channels]`.

    Raises:
      ValueError: If warp_x, warp_y and data have incompatible shapes.
    """
    if not warp_x.shape.is_compatible_with(warp_y.shape):
        raise ValueError(
            'warp_x and warp_y are of incompatible shapes: %s vs %s ' %
            (str(warp_x.shape), str(warp_y.shape)))
    warp_shape = tf.shape(warp_x)
    if warp_x.shape[0] != data.shape[0]:
        raise ValueError(
            '\'warp_x\' and \'data\' must have compatible first '
            'dimension (batch size), but their shapes are %s and %s ' %
            (str(warp_x.shape[0]), str(data.shape[0])))

    # Compute the four points closest to warp with integer value.
    warp_floor_x = tf.floor(warp_x)
    warp_floor_y = tf.floor(warp_y)
    # Compute the weight for each point.
    right_warp_weight = warp_x - warp_floor_x
    down_warp_weight = warp_y - warp_floor_y

    warp_floor_x = tf.cast(warp_floor_x, tf.int32)
    warp_floor_y = tf.cast(warp_floor_y, tf.int32)
    warp_ceil_x = tf.cast(tf.math.ceil(warp_x), tf.int32)
    warp_ceil_y = tf.cast(tf.math.ceil(warp_y), tf.int32)

    left_warp_weight = tf.subtract(
        tf.convert_to_tensor(1.0, right_warp_weight.dtype), right_warp_weight)
    up_warp_weight = tf.subtract(
        tf.convert_to_tensor(1.0, down_warp_weight.dtype), down_warp_weight)

    # Extend warps from [batch_size, dim_0, ... , dim_n, 2] to
    # [batch_size, dim_0, ... , dim_n, 3] with the first element in last
    # dimension being the batch index.

    # A shape like warp_shape but with all sizes except the first set to 1:
    warp_batch_shape = tf.concat(
        [warp_shape[0:1], tf.ones_like(warp_shape[1:])], 0)

    warp_batch = tf.reshape(
        tf.range(warp_shape[0], dtype=tf.int32), warp_batch_shape)

    # Broadcast to match shape:
    warp_batch += tf.zeros_like(warp_y, dtype=tf.int32)
    left_warp_weight = tf.expand_dims(left_warp_weight, axis=-1)
    down_warp_weight = tf.expand_dims(down_warp_weight, axis=-1)
    up_warp_weight = tf.expand_dims(up_warp_weight, axis=-1)
    right_warp_weight = tf.expand_dims(right_warp_weight, axis=-1)

    up_left_warp = tf.stack([warp_batch, warp_floor_y, warp_floor_x], axis=-1)
    up_right_warp = tf.stack([warp_batch, warp_floor_y, warp_ceil_x], axis=-1)
    down_left_warp = tf.stack([warp_batch, warp_ceil_y, warp_floor_x], axis=-1)
    down_right_warp = tf.stack([warp_batch, warp_ceil_y, warp_ceil_x], axis=-1)

    def gather_nd(params, indices):
        return (safe_gather_nd if safe else tf.gather_nd)(params, indices)

    # gather data then take weighted average to get resample result.
    result = (
            (gather_nd(data, up_left_warp) * left_warp_weight +
             gather_nd(data, up_right_warp) * right_warp_weight) * up_warp_weight +
            (gather_nd(data, down_left_warp) * left_warp_weight +
             gather_nd(data, down_right_warp) * right_warp_weight) *
            down_warp_weight)
    result_shape = (
            warp_x.get_shape().as_list() + data.get_shape().as_list()[-1:])
    result.set_shape(result_shape)
    return result


def safe_gather_nd(params, indices):
    """Gather slices from params into a Tensor with shape specified by indices.

    Similar functionality to tf.gather_nd with difference: when index is out of
    bound, always return 0.

    Args:
      params: A Tensor. The tensor from which to gather values.
      indices: A Tensor. Must be one of the following types: int32, int64. Index
        tensor.

    Returns:
      A Tensor. Has the same type as params. Values from params gathered from
      specified indices (if they exist) otherwise zeros, with shape
      indices.shape[:-1] + params.shape[indices.shape[-1]:].
    """
    params_shape = tf.shape(params)
    indices_shape = tf.shape(indices)
    slice_dimensions = indices_shape[-1]

    max_index = params_shape[:slice_dimensions] - 1
    min_index = tf.zeros_like(max_index, dtype=tf.int32)

    clipped_indices = tf.clip_by_value(indices, min_index, max_index)

    # Check whether each component of each index is in range [min, max], and
    # allow an index only if all components are in range:
    mask = tf.reduce_all(
        tf.logical_and(indices >= min_index, indices <= max_index), -1)
    mask = tf.expand_dims(mask, -1)

    return (tf.cast(mask, dtype=params.dtype) *
            tf.gather_nd(params, clipped_indices))


def clamp_and_filter_result(pixel_x, pixel_y, z):
    """Clamps and masks out out-of-bounds pixel coordinates.
    *First part of original Bilinear Sampler

    Args:
      pixel_x: a tf.Tensor containing x pixel coordinates in an image.
      pixel_y: a tf.Tensor containing y pixel coordinates in an image.
      z: a tf.Tensor containing the depth ar each (pixel_y, pixel_x)  All shapes
        are [B, H, W].

    Returns:
      pixel_x, pixel_y, mask, where pixel_x and pixel_y are the original ones,
      except:
      - Values that fall out of the image bounds, which are [0, W-1) in x and
        [0, H-1) in y, are clamped to the bounds
      - NaN values in pixel_x, pixel_y are replaced by zeros
      mask is False at allpoints where:
      - Clamping in pixel_x or pixel_y was performed
      - NaNs were replaced by zeros
      - z is non-positive,
      and True everywhere else, that is, where pixel_x, pixel_y are finite and
      fall within the frame.
    """
    _, height, width = tf.unstack(tf.shape(pixel_x))
    height = tf.cast(height, tf.float32)
    width = tf.cast(width, tf.float32)
    x_not_underflow = pixel_x >= 0.0
    y_not_underflow = pixel_y >= 0.0
    x_not_overflow = pixel_x < (width - 1)
    y_not_overflow = pixel_y < (height - 1)
    z_positive = z > 0.0
    x_not_nan = tf.math.logical_not(tf.math.is_nan(pixel_x))
    y_not_nan = tf.math.logical_not(tf.math.is_nan(pixel_y))
    not_nan = tf.logical_and(x_not_nan, y_not_nan)
    not_nan_mask = tf.cast(not_nan, tf.float32)
    pixel_x *= not_nan_mask
    pixel_y *= not_nan_mask
    pixel_x = tf.clip_by_value(pixel_x, 0.0, (width - 1))
    pixel_y = tf.clip_by_value(pixel_y, 0.0, (height - 1))
    mask_stack = tf.stack([
        x_not_underflow, y_not_underflow, x_not_overflow, y_not_overflow,
        z_positive, not_nan
    ],
        axis=0)
    mask = tf.reduce_all(mask_stack, axis=0)
    return pixel_x, pixel_y, mask


class TransformationMap(object):
    """A collection of tensors that described a transformed depth map.

    This class describes the result of a spatial transformation applied on a depth
    map. The initial depth map was defined on a regular pixel grid. Knowing the
    camera intrinsics, each pixel can be mapped to a point in space.

    However once the camera or the scene has moved, when the points are projected
    back onto the camera, they don't fall on a regular pixel grid anymore. To
    obtain a new depth map on a regular pixel grid, one needs to resample, taking
    into account occlusions, and leaving gaps at areas that were occluded before
    the movement.

    This class describes the transformed depth map on an IRREGULAR grid, before
    any resampling. The attributes are 4 tensors of shape [B, H, W]
    (batch, height, width): pixel_x, pixel_y, depth and mask.

    The given a triplet of indices, (b, i, j), the depth at the pixel location
    (pixel_y[b, i, j], pixel_x[b, i, j]) on the depth image is depth[b, i, j].
    As explained above, (pixel_y[b, i, j], pixel_x[b, i, j]) are not regular with
    respect to i and j. They are floating point numbers that generally fall in
    between pixels and can fall out of image boundaries (0, 0), (H - 1, W - 1).
    For all indices b, i, j where 0 <= pixel_y[b, i, j] <= H - 1 and
    0 <= pixel_x[b, i, j] < W - 1, mask[b, i, j] is True, otherwise it's False.

    For convenience, after we mark mask[b, i, j] as false for
    (pixel_y[b, i, j], pixel_x[b, i, j]) that are out of bounds, we clamp
    (pixel_y[b, i, j], pixel_x[b, i, j]) to be within the bounds. So, you're not
    supposed to look at (pixel_y[b, i, j], pixel_x[b, i, j], depth[b, i, j]) where
    mask[b, i, j] is False, but if you do, you'll find that they were clamped
    to be within the bounds. The motivation for this is that if we later use
    pixel_x and pixel_y for warping, the clamping will result in extrapolating
    from the boundary by replicating the boundary value, which is reasonable.
    """

    def __init__(self, pixel_x, pixel_y, depth, mask):
        """Initializes an instance. The arguments is explained above."""
        self._pixel_x = pixel_x
        self._pixel_y = pixel_y
        self._depth = depth
        self._mask = mask
        self.batch_size = depth.shape[0]
        attrs = sorted(self.__dict__.keys())
        # Unlike equality, compatibility is not transitive, so we have to check all
        # pairs.
        for i in range(len(attrs)):
            for j in range(i):
                tensor_i = self.__dict__[attrs[i]]
                tensor_j = self.__dict__[attrs[j]]
                if not tensor_i.shape.is_compatible_with(tensor_j.shape):
                    raise ValueError(
                        'All tensors in TransformedDepthMap\'s constructor must have '
                        'compatible shapes, however \'%s\' and \'%s\' have the '
                        'incompatible shapes %s and %s.' %
                        (attrs[i][1:], attrs[j][1:], tensor_i.shape, tensor_j.shape))
        self._pixel_xy = None

    @property
    def pixel_x(self):
        return self._pixel_x

    @property
    def pixel_y(self):
        return self._pixel_y

    @property
    def depth(self):
        return self._depth

    @property
    def mask(self):
        mask = tf.expand_dims(tf.cast(self._mask, tf.float32), axis=3)
        # mask = tf.concat([mask]*self.batch_size, axis=-1)
        return mask

    @property
    def pixel_xy(self):
        if self._pixel_xy is None:
            name = self._pixel_x.op.name.rsplit('/', 1)[0]
            self._pixel_xy = tf.stack([self._pixel_x, self._pixel_y],
                                      axis=3,
                                      name='%s/pixel_xy' % name)
        return self._pixel_xy


# ------ Experiments ------

def create_transforamtion_map(intrinsics, rot_angles, trans_mat, depth_goal):
    px, py, z = get_transformed_depth_coords(depth_goal, trans_mat, rot_angles, intrinsics)

    pixel_x, pixel_y, mask = clamp_and_filter_result(px, py, z)
    transformation_map = TransformationMap(pixel_x, pixel_y, z, mask)
    return transformation_map


def inverse_warp_test(intrinsics, rot_angles, trans_mat, depth_goal, frame_to_resamp):
    transformation_map = create_transforamtion_map(intrinsics, rot_angles, trans_mat, depth_goal)
    frame_resamp = resampler(
        frame_to_resamp,
        transformation_map.pixel_x,
        transformation_map.pixel_y,
        safe=False)
    return frame_resamp, transformation_map


def make_data():
    batch = 5
    intrinsics = tf.random.uniform(shape=(batch,3,3))
    intrinsic_mat_inv = tf.linalg.inv(intrinsics)
    # rot_mat = tf.random.uniform(shape=(batch,3,3))
    rot_angles = tf.random.uniform(shape=(batch,1,3))
    trans_mat = tf.random.uniform(shape=(batch,1,1,3))
    depth = tf.random.uniform(shape=(batch,192,640))
    frame2rgbd = tf.random.uniform(shape=(batch, 192, 640, 4))
    return intrinsics, intrinsic_mat_inv, rot_angles, trans_mat, depth, frame2rgbd


if __name__ == '__main__':
    intrinsics, intrinsic_mat_inv, rotation_angles, trans_mat, depth, frame2rgbd = make_data()
    inverse_warp_test(intrinsics, rotation_angles, trans_mat, depth, frame2rgbd)
