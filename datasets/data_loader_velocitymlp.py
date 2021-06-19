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
import json

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class DataLoader(object):
    def __init__(self, dataset, opt, num_epochs=None, batch_size=None, inp_size=None):
        self.frame_interval = opt.frame_interval
        self.bbox_folder = opt.bbox_folder
        self.fixed_length = 4   # fixed input length of target, including number of bboxes and motion data
        self.dataset = dataset
        self.num_epochs = num_epochs if num_epochs is not None else opt.num_epochs
        self.batch_size = batch_size if batch_size is not None else opt.batch_size
        self.buffer_size = 1000  # todo: how to set good buffer size?
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
        self.inp_size = inp_size if inp_size is not None else list(map(int, opt.feed_size))
        self.sharpen_factor = None
        self.h_range_crop = self.get_valid_h_range()

    def read_filenames(self):
        # e.g. splits\\eigen_zhou\\train.txt
        split_path = os.path.join(self.dataset.split_folder, self.dataset.split_name).replace('\\', '/')
        self.filenames = readlines(split_path)
        self.num_items = len(self.filenames)
        self.steps_per_epoch = self.num_items // self.batch_size

    def build_train_dataset(self, buffer_size=None, include_bbox=True):
        return self.build_combined_dataset(num_repeat=self.num_epochs, buffer_size=buffer_size)

    def build_val_dataset(self, buffer_size, shuffle):
        return self.build_combined_dataset(num_repeat=None, buffer_size=buffer_size, shuffle=shuffle)

    def build_eval_dataset(self):
        # todo: verify file numbers, num_repeat=??
        return self.build_combined_dataset(num_repeat=1, shuffle=False, drop_last=False)

    def build_combined_dataset(self, num_repeat,
                               drop_last=True, shuffle=True, buffer_size=None):
        # Data path
        data = self.collect_image_files()
        dataset = tf.data.Dataset.from_tensor_slices(data)
        # ----------
        # Generate dataset by file paths.
        # - if include depth, generate dataset giving (images, depths), other wise only images
        # - can be manually turned off
        out_maps = [tf.float32, tf.float32, tf.float32, tf.int32]
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

    def parse_helper_get_gt_motion(self, root_dir):
        """Get motion_gt: [dy, dx, vy, vx]
        shape: (B, N, 4), where N is target number
        """
        with open(os.path.join(root_dir, 'annotation.json'), 'r') as jf:
            gt_list = json.load(jf)
        valid_num = len(gt_list)
        motion_gt = self.transform_annotation(gt_list)
        return motion_gt, valid_num

    def transform_annotation(self, data_list):
        motion_list = []
        for instance in data_list:
            motion = [*instance['position'], *instance['velocity']]
            motion_list.append(motion)
        motion_list = self.fix_length_complete([motion_list])
        return motion_list

    def parse_helper_get_image(self, folder_path):
        # "benchmark_velocity_train/clips/208/imgs 027 l" -> "benchmark_velocity_train/clips/208/imgs"
        folder_path = bytes.decode(folder_path.numpy())
        clip_folder = folder_path.replace('\\', '/').rsplit('/', 1)[0]     # "benchmark_velocity_train/clips/208"
        file_idx_pair = [40-self.frame_interval, 40]
        image_pair = []
        bboxes_pair = []

        for idx in file_idx_pair:
            image_path = os.path.join(folder_path, '{:03}.jpg'.format(idx)).replace('\\', '/')
            image_string = tf.io.read_file(image_path)  # uint8
            image = tf.image.decode_jpeg(image_string)

            # --------------------
            # Extra modification for VelocityChallenge
            # --------------------
            if 'velo' in image_path:
                image = self.extra_mod_for_VelocityChallenge(image)

            image = tf.image.convert_image_dtype(image, tf.float32)  # after possible preprocessing
            image_pair.append(tf.image.resize(image, self.inp_size))

        # ---------------------------
        # get bboxes, one for val_mask, one for far_object
        # ---------------------------

        output_imgs = tf.concat(image_pair, axis=2)    # B,H,W,6
        return output_imgs, clip_folder

    def extra_mod_for_VelocityChallenge(self, image):
        """Extra preprocessing for Velocity Challenge dataset
        The scene is too dim for detector, feature-extractor and depth net
        -> crop_to_aspect_ratio: to fit (192, 640)
        -> adjust gamma: to brighten scene
        -> sharpen: to de-blur, strength of 3 or 5 are proper
        """
        # image = self.center_crop_image_to_asp_r(image)
        image = tf.image.adjust_gamma(image, 0.7)
        # image = self.sharpen(image)
        return image

    def sharpen(self, image, strength=3):
        if strength is not None:
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
        output_imgs, clip_folder = self.parse_helper_get_image(filepath)
        outs = [output_imgs]

        # include bbox
        clip_idx = int(clip_folder.rsplit('/', 1)[1])                      # 208
        bboxes_pair, valid_num_bbox = self.derive_bbox_pair_from_clip_idx(clip_idx)
        assert len(bboxes_pair) > 0, "bboxes number must be > 0, got {}".format(len(bboxes_pair))
        outs.append(
            tf.constant(bboxes_pair, dtype=tf.float32))  # B,2,4

        # parse and generate gt_motion
        gt_motion, valid_num_gt = self.parse_helper_get_gt_motion(clip_folder)
        outs.append(tf.constant(gt_motion))

        assert valid_num_gt == valid_num_bbox, \
            "valid number of gt and bbox doesn't match, got {} vs. {}".format(valid_num_gt, valid_num_bbox)
        outs.append(valid_num_gt)

        return outs

    def derive_bbox_pair_from_clip_idx(self, clip_idx):
        """Derive the path of depth values acoording to the image path. Just replace the extension to 'txt
        E.g. F:\\Dataset\\VelocityChallenge\\benchmark_velocity_train\\clips\\1
        ->  ~\\annotation.json
        """
        bbox_dir = os.path.join(self.bbox_folder, '{:03}.json'.format(clip_idx)).replace('\\', '/')
        with open(bbox_dir, 'r') as jf:
            bboxes_per_frame = json.load(jf)
        bboxes_pair = [bboxes_per_frame[-1-self.frame_interval],
                       bboxes_per_frame[-1]]
        valid_num = len(bboxes_pair[0])
        bboxes_pair = self.fix_length_complete(bboxes_pair)     # (2, fix_len, 4)
        return bboxes_pair, valid_num

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
            folder, file_idx, side = line.split()   # benchmark_velocity_train/clips/208/imgs 027 l
            folder = folder.replace('\\', '/')
            file_path_all[i] = os.path.join(self.data_path, folder).replace('\\', '/')
        test_file = os.path.join(file_path_all[0], '001.jpg').replace('\\', '/')
        if not os.path.isfile(test_file):
            raise ValueError("file path wrong, e.g. {} doesn't exit".format(test_file))
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
            self.dataset.data_path,
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

    def fix_length_complete(self, data_lists):
        """Make each list in data_list have 4 elements.
        If not enough, duplicate; if exceeds, crop the first 4
        NOTE: this only applies, when `self.fixed_length==4`
        """
        for i in range(len(data_lists)):
            num_elem = len(data_lists[i])
            if num_elem == 1:
                data_lists[i] = data_lists[i] * 4
            elif num_elem == 2:
                data_lists[i] = data_lists[i] * 2
            elif num_elem == 3:
                data_lists[i].append(data_lists[i][0])
            elif num_elem > 4:
                data_lists[i] = data_lists[i][:4]
        return data_lists
