import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def view_options(opt):
    print('----- Viewing Options -----')
    print('exp_mode     ', opt.exp_mode)
    print('min coverage ', opt.mask_cover_min)
    print('train_pose   ', opt.train_pose)
    print('train_depth  ', opt.train_depth)
    print('add_pose_lose', opt.add_pose_loss)
    print('padding_mode ', opt.padding_mode)
    print('learn K      ', opt.learn_intrinsics)
    print('mask_border, padding_mode:', opt.mask_border, opt.padding_mode)
    print('automasking  ', opt.do_automasking)
    print('use_cycle_ls ', opt.use_cycle_consistency)
    print('cycle weight ', opt.cycle_loss_weight)
    print('poss weight, calc_reverse_transform:', opt.pose_loss_weight, opt.include_revers)
    print('smoothness w ', opt.smoothness_ratio)
    print('reproj w     ', opt.reproj_loss_weight)
    print('use_RGBD     ', opt.use_RGBD)
    print('use MinProj  ', opt.use_min_proj)
    print('concat_depth ', opt.concat_depth)
    print('out_pose_num ', opt.pose_num)
    print('lr           ', opt.learning_rate)


def is_val_loss_lowest(val_losses, val_losses_min, min_errors_thresh, disable_gt=False):
    """save model when val loss hits new low"""
    # just update val_loss_min for the first time, do not save model
    if val_losses_min['loss/total'] == 10:
        print('\tinitialize self.val_loss_min, doesn\'t count')
        skip = True
        val_losses_min['loss/total'] = val_losses['loss/total']
        for metric in min_errors_thresh:
            val_losses_min[metric] = val_losses[metric]
    else:
        # directly skip when loss is not low enough
        # if the loss is the new low, should at least 2 another metrics
        diff = val_losses['loss/total'] - val_losses_min['loss/total']
        tolerance = 0.01
        num_pass_min = 2
        if diff > tolerance:
            skip = True
        else:
            skip = False
            # If has depth_gt, do some additional checks
            if len(min_errors_thresh) != 0 and not disable_gt:
                num_pass = 0
                for metric in min_errors_thresh:
                    if metric == 'da/a1':
                        # for 'da/a1', argmax
                        if val_losses[metric] > val_losses_min[metric]:
                            num_pass += 1
                    else:
                        # for other metric, argmin
                        if val_losses[metric] < val_losses_min[metric]:
                            num_pass += 1
                skip = num_pass < num_pass_min  # if no metric exceeds, decision will be override

            if not skip:
                print('val loss hits new low!')
                val_losses_min['loss/total'] = val_losses['loss/total']
                for metric in min_errors_thresh:
                    val_losses_min[metric] = val_losses[metric]

    return not skip, val_losses_min


def build_models(models_dict, check_outputs=False, show_summary=False, rgb_cat_depth=False):
    print("->Building models")
    for k, m in models_dict.items():
        print("\t%s" % k)
        if "depth_enc" == k:
            inputs = tf.random.uniform(shape=(2, 192, 640, 3))
            outputs = m(inputs)
        elif "depth_dec" == k:
            shapes = [(2, 96, 320, 64), (2, 48, 160, 64), (2, 24, 80, 128), (2, 12, 40, 256), (2, 6, 20, 512)]
            inputs = [tf.random.uniform(shape=(shapes[i])) for i in range(len(shapes))]
            outputs = m(inputs)
        elif "pose_enc" == k:
            if rgb_cat_depth:
                shape = (2, 192, 640, 4)
            else:
                shape = (2, 192, 640, 3)
            inputs = tf.concat([tf.random.uniform(shape=shape),
                                tf.random.uniform(shape=shape)], axis=3)
            outputs = m(inputs)
        elif "pose_enc_concat" == k:
            shape = (2, 192, 640, 4)
            inputs = tf.concat([tf.random.uniform(shape=shape),
                                tf.random.uniform(shape=shape)], axis=3)
            outputs = m(inputs)
        elif "pose_dec" in k:
            shapes = [(2, 96, 320, 64), (2, 48, 160, 64), (2, 24, 80, 128), (2, 12, 40, 256), (2, 6, 20, 512)]
            inputs = [tf.random.uniform(shape=(shapes[i])) for i in range(len(shapes))]
            outputs = m(inputs)
        elif"intrinsics_head" in k:
            inputs = tf.random.uniform([2, 6, 20, 512])
            outputs = m(inputs)
        else:
            print('skipping %s'%k)

        if check_outputs:
            print(type(outputs))
            if isinstance(outputs, dict):
                for k, v in outputs.items():
                    print(k, "\t", v.shape)
            else:
                for elem in outputs:
                    print(elem.shape)

        if show_summary:
            m.summary()

    return models_dict


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
    res = tf.clip_by_value((1 - SSIM) / 2, 0, 1)
    return res


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    Not in use for now
    """
    # 1 - percentile
    thresh = tf.math.maximum((gt / pred), (pred / gt))
    a = [None] * 3
    for i in range(len(a)):
        a[i] = tf.reduce_mean(tf.cast(
            thresh < 1.25**(i+1), dtype=tf.float32)
        )
    # a1 = (thresh < 1.25     ).float().mean()
    # a2 = (thresh < 1.25 ** 2).float().mean()
    # a3 = (thresh < 1.25 ** 3).float().mean()

    # 2 - rooted mean squared error
    rmse = tf.math.sqrt(tf.reduce_mean(
        (gt - pred) ** 2
    ))

    # 3 - log rmse
    # rmse_log = (tf.math.log(gt) - tf.math.log(pred)) ** 2
    # rmse_log = tf.math.sqrt(rmse_log.mean())
    rmse_log = tf.math.sqrt(tf.reduce_mean(
        (tf.math.log(gt) - tf.math.log(pred)) ** 2
    ))

    # 4 - absolute relative error
    abs_rel = tf.reduce_mean(tf.abs(gt - pred) / gt)
    # 5 - squared relative error
    sq_rel = tf.reduce_mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a[0], a[1], a[2]


def colorize(value, vmin=None, vmax=None, cmap=None):
    # normalize
    vmin = tf.reduce_min(value) if vmin is None else vmin
    vmax = tf.reduce_max(value) if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    # squeeze last dim if it exists
    value = tf.squeeze(value)
    # quantize
    indices = tf.cast(tf.math.round(value * 255), tf.int32)
    # gather
    cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'gray')
    colors = tf.constant(cm.colors, dtype=tf.float32)
    value = tf.gather(colors, indices)
    return value


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = tf.reduce_max(x)
    mi = tf.reduce_min(x)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def project_3d(points, K, T, shape, scale):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    # K = tf.cast(K, tf.float32)

    batch_size, height, width, _ = shape
    height, width = height // (2 ** scale), width // (2 ** scale)
    eps = 1e-7

    P = tf.matmul(K, T)[:, :3, :]

    cam_points = tf.matmul(P, points)

    pix_coords = cam_points[:, :2, :] / (tf.expand_dims(cam_points[:, 2, :], axis=1) + eps)
    pix_coords = tf.reshape(pix_coords, (batch_size, 2, height, width))
    pix_coords = tf.transpose(pix_coords, [0, 2, 3, 1])

    # pix_coords = pix_coords.numpy()     # Tensor can't be assigned
    pix_coords_0 = pix_coords[..., 0]
    pix_coords_1 = pix_coords[..., 1]
    tensor_w = tf.ones_like(pix_coords_0) * (width - 1)
    tensor_h = tf.ones_like(tensor_w) * (height - 1)
    pix_coords_0 = tf.expand_dims(pix_coords_0 / tensor_w, axis=-1)
    pix_coords_1 = tf.expand_dims(pix_coords_1 / tensor_h, axis=-1)

    pix_coords = tf.concat([pix_coords_0, pix_coords_1], axis=-1)
    # pix_coords[..., 0] /= width - 1
    # pix_coords[..., 1] /= height - 1
    # pix_coords = tf.convert_to_tensor(pix_coords, dtype=tf.float32)
    pix_coords = (pix_coords - 0.5) * 2
    return pix_coords


class Project3D:
    def __init__(self, shape, scale):
        super(Project3D, self).__init__()
        _, height, width, _ = shape
        self.height, self.width = height // (2 ** scale), width // (2 ** scale)
        self.eps = 1e-7

    def run_func(self, points, K, T):
        P = tf.matmul(K, T)[:, :3, :]

        cam_points = tf.matmul(P, points)
        pix_coords = cam_points[:, :2, :] / (tf.expand_dims(cam_points[:, 2, :], axis=1) + self.eps)
        pix_coords = tf.reshape(pix_coords, (points.shape[0], 2, self.height, self.width))
        pix_coords = tf.transpose(pix_coords, [0, 2, 3, 1])

        pix_coords_0 = pix_coords[..., 0]
        pix_coords_1 = pix_coords[..., 1]
        tensor_w = tf.ones_like(pix_coords_0) * (self.width - 1)
        tensor_h = tf.ones_like(tensor_w) * (self.height - 1)
        pix_coords_0 = tf.expand_dims(pix_coords_0 / tensor_w, axis=-1)
        pix_coords_1 = tf.expand_dims(pix_coords_1 / tensor_h, axis=-1)

        pix_coords = tf.concat([pix_coords_0, pix_coords_1], axis=-1)
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


class BackProjDepth:
    def __init__(self, shape, scale):
        super(BackProjDepth, self).__init__()
        batch_size, height, width, _ = shape
        height, width = height // (2 ** scale), width // (2 ** scale)

        meshgrid = tf.meshgrid(range(width), range(height), indexing='xy')
        id_coords = tf.stack(meshgrid, axis=0)
        ones = tf.ones((batch_size, 1, height * width), dtype=tf.int32)

        pix_coords = tf.expand_dims(
            tf.stack([tf.reshape(id_coords[0], [-1]),
                      tf.reshape(id_coords[1], [-1])], 0), 0)
        # - tile/repeat
        multiples = tf.constant([batch_size, 1, 1])
        pix_coords = tf.tile(pix_coords, multiples)

        pix_coords = tf.concat([pix_coords, ones], 1)
        self.pix_coords = tf.cast(pix_coords, tf.float32)

        self.ones = tf.cast(ones, tf.float32)

    def run_func(self, depth, inv_K):
        batch_size = inv_K.shape[0]
        pix_coords = self.pix_coords
        ones = self.ones
        if batch_size != self.pix_coords.shape[0]:
            pix_coords = tf.slice(pix_coords, [0, 0, 0], [batch_size, -1, -1])
            ones = tf.slice(ones, [0, 0, 0], [batch_size, -1, -1])
        cam_points = tf.matmul(inv_K[:, :3, :3], pix_coords)
        cam_points = tf.reshape(depth, (batch_size, 1, -1)) * cam_points
        cam_points = tf.concat([cam_points, ones], 1)
        return cam_points


def back_proj_depth(depth, inv_K, shape, scale):
    """Layer to transform a depth image into a point cloud
    shape_s: scaled shapes, corresponds to scales = [0,1,2,3]
    """

    batch_size, height, width, _ = shape
    height, width = height // (2**scale), width // (2**scale)

    meshgrid = tf.meshgrid(range(width), range(height), indexing='xy')
    id_coords = tf.stack(meshgrid, axis=0)
    ones = tf.ones((batch_size, 1, height * width), dtype=tf.int32)

    pix_coords = tf.expand_dims(
        tf.stack([tf.reshape(id_coords[0], [-1]),
                  tf.reshape(id_coords[1], [-1])], 0), 0)
    # - tile/repeat
    multiples = tf.constant([batch_size, 1, 1])
    pix_coords = tf.tile(pix_coords, multiples)

    pix_coords = tf.concat([pix_coords, ones], 1)
    pix_coords = tf.cast(pix_coords, tf.float32)

    ones = tf.cast(ones, tf.float32)

    cam_points = tf.matmul(inv_K[:,:3, :3], pix_coords)
    cam_points = tf.reshape(depth, (batch_size, 1, -1)) * cam_points
    cam_points = tf.concat([cam_points, ones], 1)
    return cam_points


def bilinear_sampler(img, coords, padding='border'):
    """ TF-version Bilinear Sampler
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """

    def get_pixel_value(img, x, y):
        """
        Utility function to get pixel value for coordinate
        vectors x and y from a 4D tensor image.
        Input
        -----
        - img: tensor of shape (B, H, W, C)
        - x: flattened tensor of shape (B*H*W,)
        - y: flattened tensor of shape (B*H*W,)
        Returns
        -------
        - output: tensor of shape (B, H, W, C)
        """
        shape = tf.shape(x)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]

        batch_idx = tf.range(0, batch_size)
        batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
        b = tf.tile(batch_idx, (1, height, width))

        indices = tf.stack([b, y, x], 3)
        res = tf.gather_nd(img, indices)
        return res

    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    # rescale x and y to [0, W-1/H-1]
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    x, y = coords[:, ..., 0], coords[:, ..., 1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    x = 0.5 * ((x + 1.0) * tf.cast(max_x - 1, 'float32'))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y - 1, 'float32'))

    if padding == 'zeros':
        zero = tf.zeros([], dtype='int32')
        # x = tf.clip_by_value(x, tf.cast(zero, tf.float32), tf.cast(max_x, tf.float32))
        # y = tf.clip_by_value(y, tf.cast(zero, tf.float32), tf.cast(max_y, tf.float32))

    elif padding == 'border':
        zero = tf.zeros([1], dtype=tf.int32)
        eps = tf.constant([0.5], tf.float32)
        x = tf.clip_by_value(x, eps, tf.cast(max_x, tf.float32) - eps)   # t+
        y = tf.clip_by_value(y, eps, tf.cast(max_y, tf.float32) - eps)   # t+
    else:
        raise NotImplementedError('only zeros / border padding is supported')

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
    return out


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """

    R = rot_from_axisangle(axisangle)
    t = tf.identity(translation)

    if invert:
        R = tf.transpose(R, (0, 2, 1))
        t *= -1

    T = get_translation_matrix(t)
    if invert:
        M = tf.matmul(R, T)
    else:
        M = tf.matmul(T, R)
    return M


def transformation_loss(axisangles, translations, invert=False, calc_reverse=False, include_res_trans_loss=False):
    # todo: add this loss to train Pose Decoder
    """Calculate Difference between two frames
    For poses of frame 1->2 and 2->1,
    one transformation (M) should be equal to the inverse of the other
    Args:
         axisangles: List of Tensors, each with shape (B, 1, 1, 3)
         - if only one Tensor, then no loss will be computed (return None in #0)
         translations: same as axisangle
         invert: calculate cam2cam transform in inverse temporal order
         calc_reverse: whether to calculate reverse transformation (and calculate loss)
     Returns:
         loss: scaler of None. if include_loss, the "inversed cam2cam transform between inverse-order image pair"
            will be compared, which they should be identical.
         M: mean of forward-backward transformation
    """
    # in case pose_loss is disabled, we only need M_1
    M_12 = transformation_from_parameters(axisangles[0][:, 0], translations[0][:, 0], invert)
    M_21 = None
    if not calc_reverse:
        return None, M_12, M_21

    M_21 = transformation_from_parameters(axisangles[1][:, 0], translations[1][:, 0], not invert)

    # calculate pose loss (actually it's consistency loss.. in struct2depth)
    pose_loss, _ = pose_losses(M_12, M_21, include_res_trans_loss)

    # average the error along batch dim, then sum
    # pose_loss = tf.reduce_sum(tf.reduce_mean(
    #     tf.math.abs(M_12 - M_21), axis=0)
    # )

    # average as the final estimate
    # M_mean = (M_1 + M_2) * 0.5
    return pose_loss, M_12, M_21


def pose_losses(M_12, M_21, include_trans_loss=False):
    """calculate pose losses
    Total transformation between two frames should be reverse to each other for pure egomtion
    But if there's moving objects, this fails. But not entirely:
    -> Rotation: the objects' doesn't rotate significantly, so it's reasonable to constraint.
    reversed rotations.
    -> Translation: the main motion of objects, which deviates from background egomotion
        if we put constraint on this part, we need to mask out the region with moving objects
    Note: actually it's consistency loss.. in struct2depth

    [ R2,  t2 ]    [ R1,  t1 ]     [ R2R1,  R2t1 + t2 ]
    [         ]  . [         ]  =  [                  ]
    [ 000, 1  ]    [ 000,  1 ]     [ 000,       1     ]
    """
    def norm(x):
        return tf.reduce_sum(tf.square(x), axis=-1)

    R_12, R_21 = M_12[:, :3, :3], M_21[:, :3, :3]
    T_12, T_21 = M_12[:, :3, -1], M_21[:, :3, -1]
    R_unit = tf.matmul(R_21, R_12)    # R2R1, shape (B,3,3)
    R2T1 = tf.matmul(R_21, tf.expand_dims(T_12, axis=-1))      # (B,3,1)
    T_zero = R2T1 + tf.expand_dims(T_21,axis=-1)             # (B,3,1)

    eye = tf.eye(3, batch_shape=R_12.shape[:1])
    rot_error = R_unit - eye
    rot_error = tf.reduce_mean(tf.square(rot_error), axis=(1, 2))
    rot_scale_1 = tf.reduce_mean(tf.square(R_12 - eye), axis=(1, 2))
    rot_scale_2 = tf.reduce_mean(tf.square(R_21 - eye), axis=(1, 2))
    rot_consis_loss = tf.reduce_mean(
        rot_error / (1e-24 + rot_scale_1 + rot_scale_2)
    )
    trans_consis_loss = None

    # todo: translation loss with "residual_reanslation" map
    if include_trans_loss:
        pass
    return rot_consis_loss, trans_consis_loss


def get_translation_matrix(trans_vec):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    # np impl
    # T = np.zeros((trans_vec.shape[0], 4, 4)).astype(np.float32)
    # t = np.reshape(trans_vec.numpy(), (-1, 3, 1)).astype(np.float32)
    #
    # T[:, 0, 0] = 1
    # T[:, 1, 1] = 1
    # T[:, 2, 2] = 1
    # T[:, 3, 3] = 1
    # T[:, :3, 3, None] = t
    # T = tf.convert_to_tensor(T, dtype=tf.float32)

    batch_size = trans_vec.shape[0]
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


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = tf.norm(vec, 2, 2, keepdims=True)
    axis = vec / (angle + 1e-7)

    ca = tf.math.cos(angle)
    sa = tf.math.sin(angle)
    C = 1-ca

    x = tf.expand_dims(axis[..., 0], 1)
    y = tf.expand_dims(axis[..., 1], 1)
    z = tf.expand_dims(axis[..., 2], 1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    # np impl
    # rot = np.zeros((vec.shape[0], 4, 4))
    # rot[:, 0, 0] = np.squeeze(x * xC + ca)
    # rot[:, 0, 1] = np.squeeze(xyC - zs)
    # rot[:, 0, 2] = np.squeeze(zxC + ys)
    # rot[:, 1, 0] = np.squeeze(xyC + zs)
    # rot[:, 1, 1] = np.squeeze(y * yC + ca)
    # rot[:, 1, 2] = np.squeeze(yzC - xs)
    # rot[:, 2, 0] = np.squeeze(zxC - ys)
    # rot[:, 2, 1] = np.squeeze(yzC + xs)
    # rot[:, 2, 2] = np.squeeze(z * zC + ca)
    # rot[:, 3, 3] = 1
    # rot = tf.convert_to_tensor(rot, dtype=tf.float32)

    # TF impl
    one = tf.ones_like(zxC, dtype=tf.float32)
    zero = tf.zeros_like(zxC, dtype=tf.float32)
    rot = tf.concat([
        x * xC + ca,
        xyC - zs,
        zxC + ys,
        zero,
        xyC + zs,
        y * yC + ca,
        yzC - xs,
        zero,
        zxC - ys,
        yzC + xs,
        z * zC + ca,
        zero, zero, zero, zero, one
    ], axis=2)

    rot = tf.reshape(rot, [-1, 4, 4])
    return rot


""" -------------- Not In Use ---------------"""


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


def show_images(batch_size, input_imgs, outputs, nrow=3, ncol=2):
    for i in range(batch_size):
        print(i)
        fig = plt.figure(figsize=(nrow, ncol))
        fig.add_subplot(nrow, ncol, 1)
        tgt = input_imgs[('color', 0, 0)][i].numpy()
        # tgt = inputs_imp[('color', 0, 0)][i]
        # tgt = np.transpose(tgt, [1,2,0])
        plt.imshow(tgt)

        fig.add_subplot(nrow, ncol, 3)
        src0 = input_imgs[('color_aug', -1, 0)][i].numpy()
        # src0 = inputs_imp[('color_aug', -1, 0)][i]
        # src0 = np.transpose(src0, [1,2,0])
        print(np.max(np.max(src0)), np.min(np.min(src0)))
        plt.imshow(src0)

        fig.add_subplot(nrow, ncol, 4)
        src1 = input_imgs[('color_aug', 1, 0)][i].numpy()
        # src1 = inputs_imp[('color_aug', 1, 0)][i]
        # src1 = np.transpose(src1, [1, 2, 0])
        print(np.max(np.max(src1)), np.min(np.min(src1)))
        plt.imshow(src1)

        fig.add_subplot(nrow, ncol, 5)
        out0 = outputs[('color', -1, 0)][i].numpy()
        print(np.max(np.max(out0)), np.min(np.min(out0)))
        plt.imshow(out0)
        fig.add_subplot(nrow, ncol, 6)
        out1 = outputs[('color', 1, 0)][i].numpy()
        print(np.max(np.max(out1)), np.min(np.min(out1)))
        plt.imshow(out1)
        plt.show()


if __name__ == '__main__':
    import numpy as np
    intrinsics = np.random.rand(4,4).astype(np.float32)
    intrinsics_multiscale = get_multi_scale_intrinsics(intrinsics, 10)
    print(intrinsics_multiscale.shape)
