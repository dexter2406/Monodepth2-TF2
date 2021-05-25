import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datasets.data_loader_kitti import DataLoader


class VeloChallenge:
    """KITTi Raw dataset"""
    def __init__(self, split_folder, split_name, data_path="F:/Dataset/VelocityChallenge"):
        self.name = 'velocity_challenge'
        self.data_path = data_path
        self.split_folder = split_folder    # os.path.join(root_dir, 'splits', split_name
        self.split_name = split_name        # '{}_files.txt'.format(dataset_for)
        self.img_ext = '.jpg'

        self.full_res_shape = (720, 1280)   # H, W
        orig_K = np.array([[714, 0, 675, 0],
                           [0, 710, 376, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.K = orig_K
        self.K[0, :] /= self.full_res_shape[1]
        self.K[1, :] /= self.full_res_shape[0]

    def get_image_path(self, folder, frame_index, side):
        """helper function to get image name and path
        parent_folder: "benchmark_velocity_train/clips", fixed
        folder: "%d/imgs"
        frame_index: "031" alike
        side: no use
        """
        del side
        f_str = "{:03d}{}".format(frame_index, self.img_ext)    # f_str: "031.jpg" alike
        image_path = os.path.join(self.data_path, folder, f_str).replace('\\', '/')
        return image_path


class KITTIDataset:
    """Parent Class for all KITTI datasets"""
    full_res_shape = (375, 1242)

    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}    # KITTI
        self.img_ext = '.jpg'


class KITTIRaw(KITTIDataset):
    """KITTi Raw dataset"""
    def __init__(self, split_folder, split_name, data_path="F:/Dataset/kitti_raw", *args, **kwargs):
        super(KITTIRaw, self).__init__(*args, **kwargs)
        self.name = 'kitti_raw'
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
    def __init__(self, split_folder, split_name, data_path="F:/Dataset/kitti_odom", *args, **kwargs):
        super(KITTIOdom, self).__init__(*args, **kwargs)
        self.name = 'kitti_odom'
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


class GeneralDataset:
    """Parent Class for all custom datasets"""
    def __init__(self, *args, **kwargs):
        super(GeneralDataset, self).__init__(*args, **kwargs)
        self.img_ext = '.jpg'
        self.K = None   # load from external

    def get_image_path(self, folder, frame_index, side):
        raise NotImplementedError


class CustomDataset(GeneralDataset):
    def __init__(self, split_folder, split_name, data_path, *args, **kwargs):
        super(CustomDataset, self).__init__(*args, **kwargs)
        self.data_path = data_path
        self.split_folder = split_folder    # os.path.join(root_dir, 'splits', split_name
        self.split_name = split_name        # '{}_files.txt'.format(dataset_for)
        self.side_map = None

    def get_image_path(self, subfolder, frame_index, side):
        """helper function to get image name and path"""
        """"""
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, subfolder, f_str
        )
        return image_path


if __name__ == '__main__':
    split_folder = 'splits\\eigen_zhou'
    split_name = 'val_files.txt'
    kitti_raw = KITTIRaw(split_folder, split_name)
    data_loader = DataLoader(kitti_raw, num_epoch=1, batch_size=2, frame_idx=[0,-1,1])
    depth_files = data_loader.collect_depth_files()
    depth_gt = data_loader.parse_func_depth(depth_files[0])



