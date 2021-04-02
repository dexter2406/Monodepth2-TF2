import tensorflow as tf
import numpy as np


def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    y = tf.pad(y, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

    mu_x = tf.nn.avg_pool2d(x, 3, 1, 'VALID')
    mu_y = tf.nn.avg_pool2d(y, 3, 1, 'VALID')

    sigma_x = tf.nn.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
    sigma_y = tf.nn.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
    sigma_xy = tf.nn.avg_pool2d(x * y, 3, 1, 'VALID') - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return tf.clip_by_value((1 - SSIM) / 2, 0, 1)


def projective_inverse_warp(img, depth, pose, intrinsics, invert=False, euler=False):
    """Inverse warp a source image to the target image plane based on projection.
    Args:
      img: the source image [batch, height_s, width_s, 3]
      depth: depth map of the target image [batch, height_t, width_t]
      pose: target to source camera transformation matrix [batch, 6], in the
            order of rx, ry, rz, tx, ty, tz
      intrinsics: camera intrinsics [batch, 3, 3]
    Returns:
      Source image inverse warped to the target image plane [batch, height_t,
      width_t, 3]
    """
    batch, height, width, _ = img.shape
    print("\t projective_inverse_warp: ")
    print("\t input image shape: ", img.shape)
    print("\t intrinsics shape: ", intrinsics.shape)
    # Convert pose vector to matrix
    if euler:
        pose = pose_vec2mat(pose)
    else:
        pose = pose_axis_angle_vec2mat(pose, invert)
    # Construct pixel grid coordinates
    pixel_coords = meshgrid(batch, height, width)
    # Convert pixel coordinates to the camera frame
    cam_coords = pixel2cam(depth, pixel_coords, intrinsics)
    # Construct a 4x4 intrinsic matrix (TODO: can it be 3x4?)
    filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    filler = tf.tile(filler, [batch, 1, 1])
    intrinsics = tf.concat([intrinsics, tf.zeros([batch, 3, 1])], axis=2)
    intrinsics = tf.concat([intrinsics, filler], axis=1)
    # Get a 4x4 transformation matrix from 'target' camera frame to 'source' pixel frame.
    proj_tgt_cam_to_src_pixel = tf.matmul(intrinsics, pose)
    src_pixel_coords = cam2pixel(cam_coords, proj_tgt_cam_to_src_pixel)
    output_img = bilinear_sampler(img, src_pixel_coords)
    return output_img


def pose_axis_angle_vec2mat(vec, invert=False):
    """
    Convert axis angle and translation into 4x4 matrix
    :param vec: [B,1,6] with former 3 vec is axis angle
    :return:
    """
    batch_size, _ = vec.shape

    axisvec = tf.slice(vec, [0, 0], [-1, 3])
    axisvec = tf.reshape(axisvec, [batch_size, 1, 3])

    translation = tf.slice(vec, [0, 3], [-1, 3])
    translation = tf.reshape(translation, [batch_size, 1, 3])

    R = rotfromAxisAngle(axisvec)

    if invert:
        R = tf.compat.v1.transpose(R, [0, 2, 1])
        translation *= -1
    t = getTransMatrix(translation)

    if invert:
        M = tf.matmul(R, t)
    else:
        M = tf.matmul(t, R)
    return M


# added by fangloveskari 2019/09/26
# adapted from monodepth2
def getTransMatrix(trans_vec):
    """
    Convert a translation vector into a 4x4 transformation matrix
    """
    batch_size = trans_vec.shape[0]
    # [B, 1, 1]
    one = tf.ones([batch_size, 1, 1], dtype=tf.float32)
    zero = tf.zeros([batch_size, 1, 1], dtype=tf.float32)

    T = tf.concat([
        one, zero, zero, trans_vec[:, :, :1],
        zero, one, zero, trans_vec[:, :, 1:2],
        zero, zero, one, trans_vec[:, :, 2:3],
        zero, zero, zero, one

    ], axis=2)

    T = tf.reshape(T, [batch_size, 4, 4])
    return T


def rotfromAxisAngle(vec):
    """
    Convert axis angle into rotation matrix
    not euler angle but Axis
    :param vec: [B, 1, 3]
    :return:
    """
    angle = tf.compat.v1.norm(vec, ord=2, axis=2, keepdims=True)
    axis = vec / (angle + 1e-7)

    ca = tf.math.cos(angle)
    sa = tf.sin(angle)

    C = 1 - ca

    x = axis[:, :, :1]
    y = axis[:, :, 1:2]
    z = axis[:, :, 2:3]

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    # [B, 1, 1]
    one = tf.ones_like(zxC, dtype=tf.float32)
    zero = tf.zeros_like(zxC, dtype=tf.float32)

    rot_matrix = tf.concat([
        x * xC + ca, xyC - zs, zxC + ys, zero,
        xyC + zs, y * yC + ca, yzC - xs, zero,
        zxC - ys, yzC + xs, z * zC + ca, zero,
        zero, zero, zero, one
    ], axis=2)

    rot_matrix = tf.reshape(rot_matrix, [-1, 4, 4])

    return rot_matrix


def bilinear_sampler(imgs, coords):
    """Construct a new image by bilinear sampling from the input image.
    Points falling outside the source image boundary have value 0.
    Args:
      imgs: source image to be sampled from [batch, height_s, width_s, channels]
      coords: coordinates of source pixels to sample from [batch, height_t,
        width_t, 2]. height_t/width_t correspond to the dimensions of the output
        image (don't need to be the same as height_s/width_s). The two channels
        correspond to x and y coordinates respectively.
    Returns:
      A new sampled image [batch, height_t, width_t, channels]
    """

    def _repeat(x, n_repeats):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
        rep = tf.cast(rep, 'float32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

    with tf.name_scope('grid_sampling'):
        coords_x, coords_y = tf.compat.v1.split(coords, [1, 1], axis=3)
        inp_size = imgs.get_shape()
        coord_size = coords.get_shape()
        out_size = list(coords.numpy().shape)
        out_size[3] = imgs.numpy().shape[3]

        coords_x = tf.cast(coords_x, 'float32')
        coords_y = tf.cast(coords_y, 'float32')

        # edit at 05/26 by Frank
        # use border pixel [0.5, max-0.5] for out of bounder reprojection
        # pixels
        y_max = tf.cast(tf.compat.v1.shape(imgs)[1] - 1, 'float32')
        x_max = tf.cast(tf.compat.v1.shape(imgs)[2] - 1, 'float32')
        zero = tf.zeros([1], dtype='float32')
        eps = tf.compat.v1.constant([0.5], tf.float32)

        coords_x = tf.clip_by_value(coords_x, eps, x_max - eps)
        coords_y = tf.clip_by_value(coords_y, eps, y_max - eps)

        x0 = tf.compat.v1.floor(coords_x)
        x1 = x0 + 1
        y0 = tf.compat.v1.floor(coords_y)
        y1 = y0 + 1

        x0_safe = tf.clip_by_value(x0, zero, x_max)
        y0_safe = tf.clip_by_value(y0, zero, y_max)
        x1_safe = tf.clip_by_value(x1, zero, x_max)
        y1_safe = tf.clip_by_value(y1, zero, y_max)

        ## bilinear interp weights, with points outside the grid having weight 0
        # wt_x0 = (x1 - coords_x) * tf.cast(tf.compat.v1.equal(x0, x0_safe), 'float32')
        # wt_x1 = (coords_x - x0) * tf.cast(tf.compat.v1.equal(x1, x1_safe), 'float32')
        # wt_y0 = (y1 - coords_y) * tf.cast(tf.compat.v1.equal(y0, y0_safe), 'float32')
        # wt_y1 = (coords_y - y0) * tf.cast(tf.compat.v1.equal(y1, y1_safe), 'float32')

        wt_x0 = x1_safe - coords_x  # 1
        wt_x1 = coords_x - x0_safe  # 0
        wt_y0 = y1_safe - coords_y  # 1
        wt_y1 = coords_y - y0_safe  # 0

        ## indices in the flat image to sample from
        dim2 = tf.cast(inp_size[2], 'float32')
        dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')
        base = _repeat(tf.cast(tf.range(coord_size[0]), 'float32') * dim1,
                       coord_size[1] * coord_size[2])
        print("base size: ", base.shape, "\tbase shape: ", [out_size[0], out_size[1], out_size[2], 1])
        base = tf.reshape(base, [out_size[0], out_size[1], out_size[2], 1])

        base_y0 = base + y0_safe * dim2
        base_y1 = base + y1_safe * dim2
        idx00 = tf.reshape(x0_safe + base_y0, [-1])
        idx01 = x0_safe + base_y1
        idx10 = x1_safe + base_y0
        idx11 = x1_safe + base_y1

        ## sample from imgs
        imgs_flat = tf.reshape(imgs, tf.stack([-1, inp_size[3]]))
        imgs_flat = tf.cast(imgs_flat, 'float32')
        im00 = tf.reshape(tf.compat.v1.gather(imgs_flat, tf.cast(idx00, 'int32')), out_size)
        im01 = tf.reshape(tf.compat.v1.gather(imgs_flat, tf.cast(idx01, 'int32')), out_size)
        im10 = tf.reshape(tf.compat.v1.gather(imgs_flat, tf.cast(idx10, 'int32')), out_size)
        im11 = tf.reshape(tf.compat.v1.gather(imgs_flat, tf.cast(idx11, 'int32')), out_size)

        w00 = wt_x0 * wt_y0
        w01 = wt_x0 * wt_y1
        w10 = wt_x1 * wt_y0
        w11 = wt_x1 * wt_y1

        output = tf.math.add_n([
            w00 * im00, w01 * im01,
            w10 * im10, w11 * im11
        ])
        return output


def pose_vec2mat(vec):
    """Converts 6DoF parameters to transformation matrix
    Args:
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 4, 4]
    """
    batch_size, _ = vec.shape
    translation = tf.slice(vec, [0, 0], [-1, 3])
    translation = tf.expand_dims(translation, -1)
    rx = tf.slice(vec, [0, 3], [-1, 1])
    ry = tf.slice(vec, [0, 4], [-1, 1])
    rz = tf.slice(vec, [0, 5], [-1, 1])
    rot_mat = euler2mat(rz, ry, rx)
    rot_mat = tf.compat.v1.squeeze(rot_mat, axis=[1])
    filler = tf.compat.v1.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    filler = tf.tile(filler, [batch_size, 1, 1])
    transform_mat = tf.concat([rot_mat, translation], axis=2)
    transform_mat = tf.concat([transform_mat, filler], axis=1)
    return transform_mat


def pixel2cam(depth, pixel_coords, intrinsics, is_homogeneous=True):
    """Transforms coordinates in the pixel frame to the camera frame.
    Args:
      depth: [batch, height, width]
      pixel_coords: homogeneous pixel coordinates [batch, 3, height, width]
      intrinsics: camera intrinsics [batch, 3, 3]
      is_homogeneous: return in homogeneous coordinates
    Returns:
      Coords in the camera frame [batch, 3 (4 if homogeneous), height, width]
    """
    print("========================")
    print("\t pixel2cam...")
    print("\t\t depth shape ", depth.shape)
    print("\t\t pixel_coords shape ", pixel_coords.shape)
    print("\t\t intrinsics shape ", intrinsics.shape)

    batch, height, width = depth.shape
    depth = tf.reshape(depth, [batch, 1, -1])
    pixel_coords = tf.reshape(pixel_coords, [batch, 3, -1])
    cam_coords = tf.matmul(tf.compat.v1.matrix_inverse(intrinsics), pixel_coords) * depth
    if is_homogeneous:
        ones = tf.ones([batch, 1, height * width])
        cam_coords = tf.concat([cam_coords, ones], axis=1)
    cam_coords = tf.reshape(cam_coords, [batch, -1, height, width])
    return cam_coords


def cam2pixel(cam_coords, proj):
    """Transforms coordinates in a camera frame to the pixel frame.
    Args:
      cam_coords: [batch, 4, height, width]
      proj: [batch, 4, 4]
    Returns:
      Pixel coordinates projected from the camera frame [batch, height, width, 2]
    """
    batch, _, height, width = cam_coords.shape
    cam_coords = tf.reshape(cam_coords, [batch, 4, -1])
    unnormalized_pixel_coords = tf.matmul(proj, cam_coords)
    x_u = tf.slice(unnormalized_pixel_coords, [0, 0, 0], [-1, 1, -1])
    y_u = tf.slice(unnormalized_pixel_coords, [0, 1, 0], [-1, 1, -1])
    z_u = tf.slice(unnormalized_pixel_coords, [0, 2, 0], [-1, 1, -1])
    x_n = x_u / (z_u + 1e-10)
    y_n = y_u / (z_u + 1e-10)
    pixel_coords = tf.concat([x_n, y_n], axis=1)
    pixel_coords = tf.reshape(pixel_coords, [batch, 2, height, width])
    return tf.compat.v1.transpose(pixel_coords, perm=[0, 2, 3, 1])


def meshgrid(batch, height, width, is_homogeneous=True):
    """Construct a 2D meshgrid.
    Args:
      batch: batch size
      height: height of the grid
      width: width of the grid
      is_homogeneous: whether to return in homogeneous coordinates
    Returns:
      x,y grid coordinates [batch, 2 (3 if homogeneous), height, width]
    """
    x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                    tf.transpose(tf.expand_dims(
                        tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
    y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                    tf.ones(shape=tf.stack([1, width])))
    x_t = (x_t + 1.0) * 0.5 * tf.cast(width - 1, tf.float32)
    y_t = (y_t + 1.0) * 0.5 * tf.cast(height - 1, tf.float32)
    if is_homogeneous:
        ones = tf.ones_like(x_t)
        coords = tf.stack([x_t, y_t, ones], axis=0)
    else:
        coords = tf.stack([x_t, y_t], axis=0)
    coords = tf.tile(tf.expand_dims(coords, 0), [batch, 1, 1, 1])
    return coords

def euler2mat(z, y, x):
    """Converts euler angles to rotation matrix
     TODO: remove the dimension for 'N' (deprecated for converting all source
           poses altogether)
     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        z: rotation angle along z axis (in radians) -- size = [B, N]
        y: rotation angle along y axis (in radians) -- size = [B, N]
        x: rotation angle along x axis (in radians) -- size = [B, N]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, N, 3, 3]
    """
    B = tf.compat.v1.shape(z)[0]
    N = 1
    z = tf.clip_by_value(z, -np.pi, np.pi)
    y = tf.clip_by_value(y, -np.pi, np.pi)
    x = tf.clip_by_value(x, -np.pi, np.pi)

    # Expand to B x N x 1 x 1
    z = tf.expand_dims(tf.expand_dims(z, -1), -1)
    y = tf.expand_dims(tf.expand_dims(y, -1), -1)
    x = tf.expand_dims(tf.expand_dims(x, -1), -1)

    zeros = tf.zeros([B, N, 1, 1])
    ones = tf.ones([B, N, 1, 1])

    cosz = tf.math.cos(z)
    sinz = tf.sin(z)
    rotz_1 = tf.concat([cosz, -sinz, zeros], axis=3)
    rotz_2 = tf.concat([sinz, cosz, zeros], axis=3)
    rotz_3 = tf.concat([zeros, zeros, ones], axis=3)
    zmat = tf.concat([rotz_1, rotz_2, rotz_3], axis=2)

    cosy = tf.math.cos(y)
    siny = tf.sin(y)
    roty_1 = tf.concat([cosy, zeros, siny], axis=3)
    roty_2 = tf.concat([zeros, ones, zeros], axis=3)
    roty_3 = tf.concat([-siny, zeros, cosy], axis=3)
    ymat = tf.concat([roty_1, roty_2, roty_3], axis=2)

    cosx = tf.math.cos(x)
    sinx = tf.sin(x)
    rotx_1 = tf.concat([ones, zeros, zeros], axis=3)
    rotx_2 = tf.concat([zeros, cosx, -sinx], axis=3)
    rotx_3 = tf.concat([zeros, sinx, cosx], axis=3)
    xmat = tf.concat([rotx_1, rotx_2, rotx_3], axis=2)

    rotMat = tf.matmul(tf.matmul(xmat, ymat), zmat)
    return rotMat


def get_multi_scale_intrinsics(intrinsics, num_scales):
    intrinsics_mscale = []
    # Scale the intrinsics accordingly for each scale
    for s in range(num_scales):
        fx = intrinsics[:,0,0] / (2 ** s)
        fy = intrinsics[:,1,1] / (2 ** s)
        cx = intrinsics[:,0,2] / (2 ** s)
        cy = intrinsics[:,1,2] / (2 ** s)
        intrinsics_mscale.append(
            make_intrinsics_matrix(fx, fy, cx, cy))
    intrinsics_mscale = tf.stack(intrinsics_mscale, axis=1)
    return intrinsics_mscale

def make_intrinsics_matrix(fx, fy, cx, cy):
    # Assumes batch input
    batch_size = fx.shape[0]
    zeros = tf.zeros_like(fx)
    r1 = tf.stack([fx, zeros, cx], axis=1)
    r2 = tf.stack([zeros, fy, cy], axis=1)
    r3 = tf.constant([0.,0.,1.], shape=[1, 3])
    r3 = tf.tile(r3, [batch_size, 1])
    intrinsics = tf.stack([r1, r2, r3], axis=1)
    return intrinsics


if __name__ == '__main__':
    import numpy as np
    intrinsics = np.random.rand(4,4).astype(np.float32)

    intrinsics_multiscale = get_multi_scale_intrinsics(intrinsics, 10)
    print(intrinsics_multiscale.shape)
