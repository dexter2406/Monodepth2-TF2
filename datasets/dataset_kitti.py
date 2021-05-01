import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datasets.data_loader_kitti import DataLoader


class KITTIDataset:
    """Parent Class for all KITTI datasets"""
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}    # KITTI
        self.img_ext = '.jpg'
        self.full_res_shape = (1242, 375)


class KITTIRaw(KITTIDataset):
    """KITTi Raw dataset"""
    name = 'KITTI_Raw'

    def __init__(self, split_folder, split_name, data_path="F:\\Dataset\\kitti_raw", *args, **kwargs):
        super(KITTIRaw, self).__init__(*args, **kwargs)
        self.data_path = data_path
        # todo: define it as passed parameter
        self.split_folder = split_folder    # os.path.join(root_dir, 'splits', split_name
        self.split_name = split_name        # '{}_files.txt'.format(dataset_for)

    def get_image_path(self, folder, frame_index, side):
        """helper function to get image name and path"""
        """"""
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}\\data".format(self.side_map[side]), f_str)
        return image_path


class KITTIOdom(KITTIDataset):
    """KITTi Odometry dataset"""
    name = 'KITTI_Odom'

    def __init__(self, split_folder, split_name, data_path="F:\\Dataset\\kitti_odom", *args, **kwargs):
        super(KITTIOdom, self).__init__(*args, **kwargs)
        self.data_path = data_path
        self.split_folder = split_folder    # splits\\odom
        self.split_name = split_name        # test_files_09.txt

    def get_image_path(self, folder, frame_index, side):
        """helper function to get image name and path"""
        """"""
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "sequences\\{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str)
        return image_path


if __name__ == '__main__':
    split_folder = 'splits\\eigen_zhou'
    split_name = 'val_files.txt'
    kitti_raw = KITTIRaw(split_folder, split_name)
    data_loader = DataLoader(kitti_raw, num_epoch=1, batch_size=2, frame_idx=[0,-1,1])
    depth_files = data_loader.collect_depth_files()
    depth_gt = data_loader.parse_func_depth(depth_files[0])



