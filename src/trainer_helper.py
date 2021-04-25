import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def is_val_loss_lowest(val_losses, val_losses_min, min_errors_thresh):
    """save model when val loss hits new low"""
    # just update val_loss_min, but not save model
    if val_losses_min['da/a1'] == 10:
        print('initialize self.val_loss_min, doesn\'t count')
        skip = True
        val_losses_min['loss/total'] = val_losses['loss/total']
        for metric in min_errors_thresh:
            val_losses_min[metric] = val_losses[metric]
    else:
        # directly skip when loss is not low enough
        # if the loss is the new low, should at least 2 another metrics
        if val_losses['loss/total'] < max(0.1, val_losses_min['loss/total']):
            val_losses_min['loss/total'] = val_losses['loss/total']
            num_pass = 0
            for metric in min_errors_thresh:
                if metric == 'da/a1':
                    print('--- a1 ---')
                    if val_losses[metric] > val_losses_min[metric]:
                        val_losses_min[metric] = val_losses[metric]
                        num_pass += 1
                else:
                    if val_losses[metric] < val_losses_min[metric]:
                        val_losses_min[metric] = val_losses[metric]
                        num_pass += 1
            skip = num_pass < 2
            if not skip: print('val loss hits new low!')
        else:
            skip = True
    return not skip, val_losses_min


def build_models(models_dict, check_outputs=False, show_summary=False):
    print("->Building models")
    for k, m in models_dict.items():
        print("\t%s" % k)
        if "depth_enc" == k:
            inputs = tf.random.uniform(shape=(1, 192, 640, 3))
            outputs = m(inputs)
        elif "depth_dec" == k:
            shapes = [(1, 96, 320, 64), (1, 48, 160, 64), (1, 24, 80, 128), (1, 12, 40, 256), (1, 6, 20, 512)]
            inputs = [tf.random.uniform(shape=(shapes[i])) for i in range(len(shapes))]
            outputs = m(inputs)
        elif "pose_enc" == k:
            shape = (1, 192, 640, 3)
            inputs = tf.concat([tf.random.uniform(shape=shape),
                                tf.random.uniform(shape=shape)], axis=3)
            outputs = m(inputs)
        elif "pose_dec" == k:
            shapes = [(1, 96, 320, 64), (1, 48, 160, 64), (1, 24, 80, 128), (1, 12, 40, 256), (1, 6, 20, 512)]
            inputs = [tf.random.uniform(shape=(shapes[i])) for i in range(len(shapes))]
            outputs = m(inputs)
        else:
            raise NotImplementedError

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


def bilinear_sampler(img, coords):
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
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')

    # zero = tf.zeros([], dtype='int32')    #o
    zero = tf.zeros([1], dtype=tf.int32)     #t
    eps = tf.constant([0.5], 'float32')    #t

    # rescale x and y to [0, W-1/H-1]
    x, y = coords[:, ..., 0], coords[:, ..., 1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')

    x = 0.5 * ((x + 1.0) * tf.cast(max_x - 1, 'float32')) #o
    y = 0.5 * ((y + 1.0) * tf.cast(max_y - 1, 'float32')) #o
    x = tf.clip_by_value(x, eps, tf.cast(max_x, tf.float32) - eps)   #t
    y = tf.clip_by_value(y, eps, tf.cast(max_y, tf.float32) - eps)   #t

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


def show_images(batch_size, input_imgs, outputs):
    for i in range(batch_size):
        print(i)
        fig = plt.figure(figsize=(3, 2))
        fig.add_subplot(3, 2, 1)
        tgt = input_imgs[('color', 0, 0)][i].numpy()
        # tgt = inputs_imp[('color', 0, 0)][i]
        # tgt = np.transpose(tgt, [1,2,0])
        plt.imshow(tgt)

        fig.add_subplot(3, 2, 3)
        src0 = input_imgs[('color_aug', -1, 0)][i].numpy()
        # src0 = inputs_imp[('color_aug', -1, 0)][i]
        # src0 = np.transpose(src0, [1,2,0])
        print(np.max(np.max(src0)), np.min(np.min(src0)))
        plt.imshow(src0)

        fig.add_subplot(3, 2, 4)
        src1 = input_imgs[('color_aug', 1, 0)][i].numpy()
        # src1 = inputs_imp[('color_aug', 1, 0)][i]
        # src1 = np.transpose(src1, [1, 2, 0])
        print(np.max(np.max(src1)), np.min(np.min(src1)))
        plt.imshow(src1)

        fig.add_subplot(3, 2, 5)
        out0 = outputs[("color", -1, 0)][i].numpy()
        print(np.max(np.max(out0)), np.min(np.min(out0)))
        plt.imshow(out0)
        fig.add_subplot(3, 2, 6)
        out1 = outputs[("color", 1, 0)][i].numpy()
        print(np.max(np.max(out1)), np.min(np.min(out1)))
        plt.imshow(out1)
        plt.show()


if __name__ == '__main__':
    import numpy as np
    intrinsics = np.random.rand(4,4).astype(np.float32)

    intrinsics_multiscale = get_multi_scale_intrinsics(intrinsics, 10)
    print(intrinsics_multiscale.shape)
