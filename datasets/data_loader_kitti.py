import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2 as cv
from utils import readlines
import numpy as np
from src.kitti_utils import *
import skimage.transform
import sys
import warnings

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class DataLoader(object):
    def __init__(self, dataset, num_epoch, batch_size, frame_idx, inp_size, sharpen_factor):
        self.dataset = dataset
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.buffer_size = 1000  # todo: how to set good buffer size?
        self.frame_idx = frame_idx
        self.K = self.dataset.K     # 4,4
        self.steps_per_epoch: int = None
        self.filenames: list = None
        self.num_items: int = None
        self.read_filenames()
        self.data_path = self.dataset.data_path
        self.include_depth = None
        self.has_depth = self.has_depth_file()
        self.include_bbox = False
        self.has_bbox = self.has_bbox_file()
        self.inp_size = inp_size
        self.sharpen_factor = sharpen_factor
        self.h_range_crop = self.get_valid_h_range()

    def read_filenames(self):
        # e.g. splits\\eigen_zhou\\train.txt
        split_path = os.path.join(self.dataset.split_folder, self.dataset.split_name).replace('\\', '/')
        self.filenames = readlines(split_path)
        self.num_items = len(self.filenames)
        self.steps_per_epoch = self.num_items // self.batch_size

    def build_train_dataset(self, buffer_size=None, include_bbox=True):
        return self.build_combined_dataset(include_depth=False, num_repeat=self.num_epoch,
                                           buffer_size=buffer_size, include_bbox=include_bbox)

    def build_val_dataset(self, include_depth, buffer_size, shuffle, include_bbox=True):
        return self.build_combined_dataset(include_depth=include_depth, num_repeat=None,
                                           buffer_size=buffer_size, shuffle=shuffle, include_bbox=include_bbox)

    def build_eval_dataset(self):
        # todo: verify file numbers, num_repeat=??
        return self.build_combined_dataset(include_depth=False, num_repeat=1,
                                           shuffle=False, drop_last=False, include_bbox=True)

    def build_combined_dataset(self, include_depth, num_repeat, include_bbox=False,
                               drop_last=True, shuffle=True, buffer_size=None):
        # Data path
        data = self.collect_image_files()
        dataset = tf.data.Dataset.from_tensor_slices(data)
        # ----------
        # Generate dataset by file paths.
        # - if include depth, generate dataset giving (images, depths), other wise only images
        # - can be manually turned off
        self.include_depth = include_depth
        if include_depth and not self.has_depth:
            self.include_depth = False
            warnings.warn('Cannot use depth gt, because no available depth file found')
        # the outputs are implicitly handled by 'out_maps'
        out_maps = [tf.float32, tf.float32] if self.include_depth else [tf.float32]
        if include_bbox and self.has_bbox:
            self.include_bbox = True
            out_maps.append(tf.int32)
        dataset = dataset.map(lambda x: tf.py_function(self.parse_func_combined, [x], Tout=out_maps),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # ----------
        if shuffle:
            buffer_size = self.buffer_size if buffer_size is None else buffer_size  # override if provided
            dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=shuffle)

        dataset = dataset.batch(self.batch_size, drop_remainder=drop_last)

        dataset = dataset.repeat(num_repeat)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        data_iter = iter(dataset)
        return data_iter

    def parse_helper_test_sequence(self, filepath):
        full_path = bytes.decode(filepath.numpy())
        file_root, file_name = os.path.split(full_path)
        return tf.constant(file_name)

    def parse_helper_get_depth(self, tgt_path):
        """Get depth gt, helper for parse_func_combined()"""
        calib_path, depth_path, side = self.derive_depth_path_from_img(tgt_path)
        depth_gt = generate_depth_map(calib_path, depth_path, side)
        depth_gt = skimage.transform.resize(
            depth_gt, self.dataset.full_res_shape, order=0, preserve_range=True, mode='constant')
        depth_gt = tf.constant(depth_gt, dtype=tf.float32)
        return depth_gt

    def parse_helper_get_image(self, filepath):
        full_path = bytes.decode(filepath.numpy())
        file_root, file_name = os.path.split(full_path)
        image_idx, image_ext = file_name.split('.')
        image_stack = [None] * len(self.frame_idx)  # 3 - training; 2 - evaluate pose; 1 - evaluate depth
        ext_bboxes = []
        tgt_path: str = None

        for i, f_i in enumerate(self.frame_idx):
            idx_num = int(image_idx) + f_i
            idx_num = max(0, idx_num)
            if 'raw' in self.dataset.data_path:
                image_name = ''.join(["{:010d}".format(idx_num), '.', image_ext])
            elif 'odom' in self.dataset.data_path:
                image_name = ''.join(["{:06d}".format(idx_num), '.', image_ext])
            elif 'custom' in self.dataset.data_path:
                # Custom dataset
                image_name = ''.join(["{:06d}".format(idx_num), '.', image_ext])
            elif 'Velocity' in self.dataset.data_path:
                image_name = ''.join(["{:03d}".format(idx_num), '.', image_ext])
            else:
                raise NotImplementedError
            image_path = os.path.join(file_root, image_name)
            if f_i == 0:
                tgt_path = image_path.replace('\\', '/')

            # Get images
            image_string = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image_string)
            image = tf.image.convert_image_dtype(image, tf.float32)

            # ---------------------------
            # get bboxes, one for val_mask, one for far_object
            # ---------------------------
            if self.include_bbox:
                ext_bboxes.append(
                    self.derive_bbox_path_from_img(image_path))

            # --------------------
            # Extra modification for VelocityChallenge
            # --------------------
            if 'velocity' in self.dataset.name:
                image = self.extra_mod_for_VelocityChallenge(image)

            image_stack[i] = tf.image.resize(image, self.inp_size)

        output_imgs = tf.concat(image_stack, axis=2)    # B,H,W,9
        return output_imgs, tgt_path, ext_bboxes

    def extra_mod_for_VelocityChallenge(self, image):
        """Extra preprocessing for Velocity Challenge dataset
        The scene is too dim for detector, feature-extractor and depth net
        -> crop_to_aspect_ratio: to fit (192, 640)
        -> adjust gamma: to brighten scene
        -> sharpen: to de-blur, strength of 3 or 5 are proper
        """
        image = self.center_crop_image_to_asp_r(image)
        image = tf.image.adjust_gamma(image, 0.6)
        image = self.sharpen(image, strength=self.sharpen_factor)
        return image

    def sharpen(self, image, strength):
        if self.sharpen_factor is not None:
            scales = {3: -0.25, 5: -0.5, 9: -1}
            kernel = np.ones(shape=(3, 3)) * scales[strength]
            kernel[1, 1] = strength
            image = cv.filter2D(image.numpy(), -1, kernel)
        return image

    def get_valid_h_range(self):
        """Get starting and ending points in height axis, in case there is center cropping
        -> if aspect ratio not aligned with self.inp_size, do center crop
        -> if not, return original range [0, h+1)
        """
        h, w = self.dataset.full_res_shape
        asp_ratio = self.inp_size[1] / self.inp_size[0]   # W/H, usually 640/192
        tolerance = 0.05
        if abs(w / h - asp_ratio) > tolerance:
            h_goal = int(w // asp_ratio)
            offset = int((h - h_goal) / 2)
            h_end = offset + h_goal + 1
        else:
            offset = 0
            h_end = h + 1
        return [offset, h_end]

    def center_crop_image_to_asp_r(self, image):
        offset, h_end = self.h_range_crop
        image = image[offset: h_end, :, :]
        return image

    def adapt_bbox_for_center_crop(self, bboxes):
        """Crop bbox so that is doesn't exceed boundary in center-cropped image
        Args:
            bboxes: list, (B, 4)
                l, t, r, b
                max and min box, for validity_mask and depth_constraint respectively
        Returns:
            modified bbox, adapted to the image boundary after ceter-crop
        """
        offset, h_end = self.h_range_crop
        bboxes = np.asarray(bboxes)
        # bboxes[:, 1] = np.maximum((bboxes[:, 1]), offset)
        # bboxes[:, 3] = np.minimum((bboxes[:, 3]), h_end)
        # bboxes[:, 3] = np.maximum(bboxes[:, 3], 0)
        # print("boxes before", bboxes)
        for i in range(len(bboxes)):
            if np.sum(bboxes[i]) != 0:
                bboxes[i, 1] = np.maximum((bboxes[i, 1]), offset)
                bboxes[i, 3] = np.minimum((bboxes[i, 3]), h_end)
        # print("boxes after", bboxes)
        return bboxes

    def parse_func_combined(self, filepath):
        """Decode filenames to images
        Givn one file, find the neighbors and concat to shape [H,W,3*3]
        """
        # parse images
        output_imgs, tgt_path, ext_bboxes = self.parse_helper_get_image(filepath)
        outs = [output_imgs]

        # include bbox
        if len(ext_bboxes) > 0:
            outs.append(tf.concat(ext_bboxes, axis=0))  # B,2,4

        # parse and generate depth_gt
        if self.include_depth:
            depth_gt = self.parse_helper_get_depth(tgt_path)
            outs.append(depth_gt)
        # for o in outs:
        #     print(o.shape)
        return outs

    def derive_bbox_path_from_img(self, img_str):
        """Derive the path of depth values acoording to the image path. Just replace the extension to 'txt
        E.g. F:\\Dataset\\kitti_raw\\2011_10_03\\2011_10_03_drive_0034_sync\\image_02\\data\0000003025.jpg
        ->  ~.txt
        """
        bbox_path = img_str.replace('jpg', 'txt')
        bboxes = []
        with open(bbox_path, 'r') as df:
            for i in range(2):
                bboxes.append(list(map(int, df.readline().split())))
        assert len(bboxes) == 2 and len(bboxes[0]) == 4, \
            "Bbox list should have size of (2, 4), got ({}, {})".format(len(bboxes), len(bboxes[0]))
        # print("bboxes before", bboxes)
        bboxes = self.adapt_bbox_for_center_crop(bboxes)    # np
        bboxes = self.scale_box_to_feed_size(bboxes)
        return bboxes

    def scale_box_to_feed_size(self, bboxes):
        """Scale the cropped bboxes to fit feed_size of network"""
        scaling = [self.dataset.full_res_shape[i] / self.inp_size[i] for i in range(2)]
        bboxes[..., [1, 3]] = bboxes[..., [1, 3]] // scaling[0]   # height
        bboxes[..., [0, 2]] = bboxes[..., [0, 2]] // scaling[1]   # width
        return bboxes

    def collect_image_files(self):
        """extract info from split, converted to file path"""
        self.read_filenames()
        file_path_all = [''] * self.num_items
        for i, line in enumerate(self.filenames):
            folder, file_idx, side = line.split()
            folder = folder.replace('/', '\\')
            path = self.dataset.get_image_path(folder, int(file_idx), side)
            if self.data_path.startswith('/'):
                path = os.path.join(*(path.replace('/', '\\').split('\\')))
                path = '/' + path
            file_path_all[i] = path
        if not os.path.isfile(file_path_all[0]):
            raise ValueError("file path wrong, e.g. {} doesn't exit".format(file_path_all[0]))
        return file_path_all

    def collect_depth_files(self):
        if not self.has_depth_file():
            return None
        file_path_all = [''] * self.num_items
        for i, line in enumerate(self.filenames):
            folder, file_idx, side = line.split()
            folder = folder.replace('/', '\\')
            velo_filename = os.path.join(
                self.data_path,
                folder,
                os.path.join('velodyne_points', 'data', '{:010d}.bin'.format(int(file_idx))),
                side
            )
            velo_filename = os.path.join(*(velo_filename.replace('/', '\\').split('\\')))
            if self.data_path.startswith('/'):
                velo_filename = '/' + velo_filename
            file_path_all[i] = velo_filename
        return file_path_all

    def derive_depth_path_from_img(self, img_str):
        """Derive the path of depth values acoording to the image path
        E.g. F:\\Dataset\\kitti_raw\\2011_10_03\\2011_10_03_drive_0034_sync\\image_02\\data\0000003025.jpg
        ->  F:\\Dataset\\kitti_raw\\2011_10_03\\2011_10_03_drive_0034_sync\\velodyne_points\\data\\0000003025.bin
        """
        img_str_split = img_str.split('/')
        file_idx = img_str_split[-1].split('.')[0]
        folder_path = os.path.join(*img_str_split[-5:-3])  # 2011_10_03\2011_10_03_drive_0034_sync
        side = self.dataset.side_map[img_str_split[-3][-1]]
        file_path = os.path.join('velodyne_points', 'data', '{:010d}.bin'.format(int(file_idx)))
        depth_full_path = os.path.join(self.data_path, folder_path, file_path)
        calib_path = os.path.join(self.data_path, img_str_split[-5])
        return calib_path, depth_full_path, side

    def has_bbox_file(self):
        scene_name, file_idx, _ = self.filenames[0].replace('\\', '/').split()
        bbox_filname = os.path.join(self.data_path, scene_name, file_idx + '.txt').replace('\\', '/')
        return os.path.isfile(bbox_filname)

    def has_depth_file(self):
        line = self.filenames[0].split()
        if len(line) > 1:
            scene_name = line[0].replace('/', '\\')
            file_idx = int(line[1])
        else:
            return False
        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            os.path.join('velodyne_points', 'data', '{:010d}.bin'.format(int(file_idx)))
        )
        velo_filename = os.path.join(*(velo_filename.replace('/', '\\').split('\\')))
        if self.data_path.startswith('/'):
            velo_filename = '/' + velo_filename
        return os.path.isfile(velo_filename)

    def print_info(self):
        if 'odom' in self.dataset.data_path:
            print('\tLoading KITTI-Odometry dataset, no depth gt available')
        elif 'raw' in self.dataset.data_path:
            print('\tLoading KITTI-Raw dataset, depth gt should be available')

    """ For experiments

    def build_test_sequence(self):
        # check if the sequence is correct
        return self.build_combined_dataset(is_train=True, include_depth=False, drop_last=False, shuffle=False)

    def parse_func_depth(self, filepath):
        #  F:\\Dataset\\kitti_data\\2011_09_26\\2011_09_26_drive_0001_sync\\velodyne_points\\data\\xxx.bin\\<side>
        filepath = bytes.decode(filepath.numpy())
        calib_path = os.path.join(*filepath.split('\\')[:-5])  # F:\\Dataset\\kitti_data\\2011_09_26\\2011_09_26_drive_0001_sync
        velo_path = os.path.join(*filepath.split('\\')[:-1])   # calib_path + \\data\\xxx.bin
        side = os.path.join(*filepath.split('\\')[-1])         # <side>
        depth_gt = generate_depth_map(calib_path, velo_path, self.dataset.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.dataset.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')
        depth_gt = tf.constant(depth_gt, dtype=tf.float32)
        return depth_gt

    def build_depth_dataset(self):
        data = self.collect_depth_files()
        if data is None:
            return None
        dataset = tf.data.Dataset.from_tensor_slices(data)
        # Decode image files (png or jpeg) and stack them
        dataset = dataset.map(lambda x: tf.py_function(self.parse_func_depth, [x], Tout=tf.float32),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if self.is_train:
            dataset = dataset.shuffle(buffer_size=self.buffer_size, reshuffle_each_iteration=True)
        dataset = dataset.batch(self.batch_size, True)  # Define batch image stack
        if self.is_train:
            dataset = dataset.repeat(self.num_epoch)
        dataset = dataset.prefetch(1)
        return dataset
    """


