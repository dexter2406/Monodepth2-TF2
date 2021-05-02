import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
from utils import get_models
from src.trainer_helper import transformation_from_parameters, pose_losses, BackProjDepth, Project3D, bilinear_sampler
from utils import disp_to_depth

frame_idx = [0,-1,1]
exp = True


def get_disp_map(models, input_imgs):
    print('-> get_disp_map')
    outputs = {}
    for f_i in frame_idx:
        features = models['depth_enc'](input_imgs[('color', f_i, 0)])
        disp_raw = models['depth_dec'](features)
        disp = disp_raw['output_0']
        # assert disp.shape[-1]==1 and len(disp.shape)==4, "disp shape: {}".format(disp.shape)
        outputs[('disp', f_i, 0)] =  disp
    return outputs


def get_poses_and_transformations(models, input_imgs, outputs):
    print('-> get_poses_and_transformations')
    axisangles, translations = [], []
    if not exp:
        pose_inps = {f_i: input_imgs[("color", f_i, 0)] for f_i in frame_idx}
    else:
        pose_inps = {f_i: tf.concat(
            [input_imgs[("color", f_i, 0)], outputs[('disp', f_i, 0)]], axis=3)
            for f_i in frame_idx
        }
    
    for f_i in frame_idx[1:]:
        print("frame ", f_i)
        if f_i == -1:
            pose_inp = [pose_inps[f_i], pose_inps[0]]
        else:
            pose_inp = [pose_inps[0], pose_inps[f_i]]

        features_fwd = models['pose_enc'](tf.concat(pose_inp, axis=3))
        pred_pose_fwd = models['pose_dec'](features_fwd)
        angle = tf.expand_dims(pred_pose_fwd['angles'][:, 0, ...], 1)
        translation = tf.expand_dims(pred_pose_fwd['translations'][:, 0, ...], 1)
        axisangles.append(angle)  # B,1,1,3
        translations.append(translation)

        features_bwd = models["pose_enc"](tf.concat(pose_inp[::-1], axis=3))
        pred_pose_bwd = models["pose_dec"](features_bwd)
        angle = tf.expand_dims(pred_pose_bwd['angles'][:, 0, ...], 1)
        translation = tf.expand_dims(pred_pose_bwd['translations'][:, 0, ...], 1)
        axisangles.append(angle)
        translations.append(translation)

        loss, M, M_inv = get_transformations(axisangles, translations,
                                             invert=(f_i < 0), calc_reverse=True)
        outputs[('cam_T_cam', f_i, 0)] = [M]
    return outputs


def get_transformations(axisangles, translations, invert, calc_reverse=True):
    # print('-> get_transformations')
    # in case pose_loss is disabled, we only need M_1
    M_12 = transformation_from_parameters(axisangles[0][:, 0], translations[0][:, 0], invert)
    M_21 = None
    if not calc_reverse:
        return None, M_12, M_21

    M_21 = transformation_from_parameters(axisangles[1][:, 0], translations[1][:, 0], not invert)

    # calculate pose loss (actually it's consistency loss.. in struct2depth)
    pose_loss, _ = pose_losses(M_12, M_21, False)

    print(M_12)
    print(M_21)
    # average the error along batch dim, then sum
    # pose_loss = tf.reduce_sum(tf.reduce_mean(
    #     tf.math.abs(M_12 - M_21), axis=0)
    # )

    # average as the final estimate
    # M_mean = (M_1 + M_2) * 0.5
    return pose_loss, M_12, M_21


def get_images(imgs_dir, base_idx):
    print('-> get_images')
    input_imgs = {}
    frame_idx = [0, -1, 1]
    for f_i in frame_idx:
        img_path = os.path.join(imgs_dir, '{:06d}'.format(base_idx+f_i)+'.png')
        image = cv.imread(img_path) / 255.
        input_data = tf.cast(tf.expand_dims(image, axis=0), dtype=tf.float32)
        input_data = tf.image.resize(input_data, (192, 640))
        input_imgs[('color', f_i, 0)] = input_data
        # plt.imshow(input_data[0]), plt.show()
    return input_imgs


def inverse_warp(depth, src_data, T):
    print('-> inverse_warp')
    input_Ks = get_intrinsics()
    back_project = BackProjDepth( [1, 192, 640, 3], 0)
    project_3d = Project3D([1, 192, 640, 3], 0)
    cam_points = back_project.run_func(
        depth, input_Ks[("inv_K", 0)]
    )
    pix_coords = project_3d.run_func(
        cam_points, input_Ks[("K", 0)], T
    )
    resamp_data = bilinear_sampler(src_data, pix_coords, padding='zeros')

    return resamp_data, pix_coords


def get_intrinsics():
    # print('-> get_intrinsics')
    input_Ks = {}
    K = np.array([[0.58, 0, 0.5, 0],
                  [0, 1.92, 0.5, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=np.float32)
    inv_K = np.linalg.pinv(K)
    input_Ks[('K', 0)] = tf.reshape(tf.tile(K, [1, 1]), (1, 4, 4))
    input_Ks[('inv_K', 0)] = tf.reshape(tf.tile(inv_K, [1, 1]), (1, 4, 4))
    return input_Ks


def show_images(inps):
    print('-> show images')
    num_inps = len(inps)
    fig = plt.figure(figsize=(num_inps, 1))
    for i in range(num_inps):
        print(i)
        fig.add_subplot(num_inps, 1, i + 1)
        plt.imshow(inps[i])
    plt.show()


def test_func0():
    weights_dir = 'logs/weights/round_2/epoch_5_lr1e-5_rproW1.5/weights_106_83'
    # weights_dir = 'logs/weights/trained_odom'
    # weights_dir = ''
    imgs_dir = r'D:\MA\Struct2Depth\KITTI_odom_02\image_2\left'

    models = get_models(weights_dir, exp=exp)
    input_imgs = get_images(imgs_dir, base_idx=43)

    outputs = get_disp_map(models, input_imgs)
    outputs.update(get_poses_and_transformations(models, input_imgs, outputs))

    for f_i in frame_idx[1:]:
        T = outputs[('cam_T_cam', f_i, 0)][0]
        _, depth_tgt = disp_to_depth(outputs[('disp', 0, 0)], 0.1, 100)
        src_data = input_imgs[('color', f_i, 0)]
        warped, _ = inverse_warp(depth_tgt, src_data, T)
        outputs[('warped_multi_s', f_i, 0)] = [warped]

    inps = [
        input_imgs[('color', 0, 0)][0],
        outputs[('warped_multi_s', -1, 0)][0][0],
        outputs[('warped_multi_s', 1, 0)][0][0],
        # outputs[('disp', 0, 0)][0],
    ]
    show_images(inps)


if __name__ == '__main__':
    test_func0()
