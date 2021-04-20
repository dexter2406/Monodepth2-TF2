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
    def __init__(self, dataset,
                 num_epoch, batch_size, frame_idx,
                 dataset_for='train'):
        # self.root = r"D:\MA\Struct2Depth\KITTI_odom_02\image_2\left"
        self.dataset = dataset
        self.is_train = True if 'train' in dataset_for else False
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.buffer_size = 1000  # todo: how to set good buffer size?
        self.frame_idx = frame_idx
        self.K = self.dataset.K
        self.steps_per_epoch:int = None
        self.filenames: list = None
        self.num_items: int = None
        self.read_filenames()
        self.data_path = self.dataset.data_path
        self.include_depth = self.has_depth_file()

    def print_info(self):
        if 'odom' in self.dataset.data_path:
            print('\tLoading KITTI-Odometry dataset, no depth gt available')
        elif 'raw' in self.dataset.data_path:
            print('\tLoading KITTI-Raw dataset, depth gt should be available')

    def read_filenames(self):
        split_path = os.path.join(self.dataset.split_folder, self.dataset.split_name)   # e.g. splits\\eigen_zhou\\train.txt
        self.filenames = readlines(split_path)
        self.num_items = len(self.filenames)
        self.steps_per_epoch = self.num_items // self.batch_size

    def collect_image_files(self):
        """extract info from split, converted to file path"""
        self.read_filenames()
        file_path_all = [''] * self.num_items
        for i, line in enumerate(self.filenames):
            folder, file_idx, side = line.split()
            folder = folder.replace('/', '\\')
            path = self.dataset.get_image_path(folder, int(file_idx), side)
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
                "velodyne_points\\data\\{:010d}.bin".format(int(file_idx)),
                side
            )
            file_path_all[i] = velo_filename
        return file_path_all

    def derive_depth_path_from_img(self, img_str):
        """Derive the path of depth values acoording to the image path
        E.g. F:\\Dataset\\kitti_raw\\2011_10_03\\2011_10_03_drive_0034_sync\\image_02\\data\0000003025.jpg
        ->  F:\\Dataset\\kitti_raw\\2011_10_03\\2011_10_03_drive_0034_sync\\velodyne_points\\data\\0000003025.bin
        """
        img_str_split = img_str.split('\\')
        file_idx = img_str_split[-1].split('.')[0]
        folder_path = os.path.join(*img_str_split[-5:-3])  # 2011_10_03\2011_10_03_drive_0034_sync
        side = self.dataset.side_map[img_str_split[-3][-1]]
        file_path = "velodyne_points\\data\\{:010d}.bin".format(int(file_idx))
        depth_full_path = os.path.join(self.data_path, folder_path, file_path)
        calib_path = os.path.join(self.data_path, img_str_split[-5])
        return calib_path, depth_full_path, side

    def parse_include_depth(self, tgt_path):
        """Get depth gt, helper for parse_func_combined()"""
        calib_path, depth_path, side = self.derive_depth_path_from_img(tgt_path)
        depth_gt = generate_depth_map(calib_path, depth_path, side)
        depth_gt = skimage.transform.resize(
            depth_gt, self.dataset.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')
        depth_gt = tf.constant(depth_gt, dtype=tf.float32)
        return depth_gt

    def parse_func_combined(self, filepath, resized_to=(192, 640)):
        """Decode filenames to images
        Givn one file, find the neighbors and concat to shape [192,640,9]
        """
        # index, img_ext = bytes.decode(filepath.numpy()).split("\\")[-1].split(".")
        full_path = bytes.decode(filepath.numpy())
        file_root, file_name = os.path.split(full_path)
        image_idx, image_ext = file_name.split('.')
        image_stack = [None] * len(self.frame_idx)    # 3 - training; 2 - evaluate pose; 1 - evaluate depth
        tgt_path: str = None
        for i, f_i in enumerate(self.frame_idx):
            idx_num = int(image_idx) + f_i
            idx_num = max(0, idx_num)
            if 'raw' in self.dataset.data_path:
                image_name = ''.join(["{:010d}".format(idx_num), '.', image_ext])
            elif 'odom' in self.dataset.data_path:
                image_name = ''.join(["{:06d}".format(idx_num), '.', image_ext])
            else:
                raise NotImplementedError
            image_path = os.path.join(file_root, image_name)
            if f_i == 0:
                tgt_path = image_path
            # Get images
            image_string = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image_string)
            image = tf.image.convert_image_dtype(image, tf.float32)
            image_stack[i] = tf.image.resize(image, resized_to)
        output_imgs = tf.concat(image_stack, axis=2)

        depth_gt = None
        if self.include_depth:
            depth_gt = self.parse_include_depth(tgt_path)

        return output_imgs, depth_gt

    def build_combined_dataset(self, include_depth=None):
        # Data path
        data = self.collect_image_files()
        dataset = tf.data.Dataset.from_tensor_slices(data)

        # ----------
        # Generate dataset by file paths.
        # - if include depth, generate dataset giving (images, depths), other wise only images
        # - can be manually turned off
        if include_depth is not None and include_depth is False:
            self.include_depth = include_depth  # manual override is only for turning-off
        elif include_depth is True and not self.include_depth:
            warnings.warn('Will not use depth gt, because no available depth file found')
        # the outputs are implicitly handled by 'out_maps'
        out_maps = [tf.float32, tf.float32] if self.include_depth else tf.float32
        dataset = dataset.map(lambda x: tf.py_function(self.parse_func_combined, [x], Tout=out_maps),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # ----------

        if self.is_train:
            dataset = dataset.shuffle(buffer_size=self.buffer_size, reshuffle_each_iteration=True)
        dataset = dataset.batch(self.batch_size, True)
        if self.is_train:
            dataset = dataset.repeat(self.num_epoch)
        dataset = dataset.prefetch(1)
        data_iter = iter(dataset)
        return data_iter

    def has_depth_file(self):
        line = self.filenames[0].split()
        scene_name = line[0].replace('/', '\\')
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points\\data\\{:010d}.bin".format(int(frame_index)))
        return os.path.isfile(velo_filename)


    """ For experiments
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


