import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator as ImgDataGen
import matplotlib.pyplot as plt
import os
import cv2 as cv
from utils import readlines


class DataLoader_KITTI_Raw(object):
    def __init__(self, num_epoch, batch_size, dataset_for='train', split_name='eigen_zhou'):
        # self.root = r"D:\MA\Struct2Depth\KITTI_odom_02\image_2\left"
        self.data_path = r"F:\Dataset\kitti_data"
        self.split_folder = r"D:\MA\Recources\monodepth2-torch\splits\{}".format(split_name)
        self.split_name = '{}_files.txt'.format(dataset_for)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        self.img_ext = '.jpg'
        self.num_items:int = None
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.steps_per_epoch:int = None
        self.buffer_size = 1000  # todo: how to set good buffer size?
        self.frame_idx = [0, -1, 1]
        self.zero_convention_in_name = 0 if 'odom' not in split_name else 1

    def collect_file_paths_from_split(self):
        """extract info from split, converted to file path"""

        def get_image_path(folder, frame_index, side):
            """helper function to get image name and path"""
            """"""
            f_str = "{:010d}{}".format(frame_index, self.img_ext)
            image_path = os.path.join(
                self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
            return image_path

        split_path = os.path.join(self.split_folder, self.split_name)
        lines = readlines(split_path)
        self.num_items = len(lines)
        self.steps_per_epoch = self.num_items // self.batch_size
        file_path_all = ['']*self.num_items
        for i, line in enumerate(lines):
            folder, file_idx, side = line.split()
            path = get_image_path(folder, int(file_idx), side)
            file_path_all[i] = path
        if not os.path.isfile(file_path_all[0]):
            raise ValueError("file path wrong, e.g. {} doesn't exit".format(file_path_all[0]))
        return file_path_all


    def parse_function(self, filepath, resized_to=(192, 640)):
        """Decode filenames to images
        Givn one file, find the neighbors and concat to shape [192,640,9]
        """
        #F:\Dataset\kitti_data\2011_09_26/2011_09_26_drive_0022_sync\image_03/data\0000000473.jpg
        # print(filepath)
        # exit("1")
        # index, img_ext = bytes.decode(filepath.numpy()).split("\\")[-1].split(".")
        full_path = bytes.decode(filepath.numpy())
        file_root, file_name = os.path.split(full_path)
        image_idx, image_ext = file_name.split('.')
        triplet = []
        for i in self.frame_idx:
            idx_num = int(image_idx) + i
            idx_num = max(0, idx_num)
            if self.zero_convention_in_name == 0:
                image_name = ''.join(["{:010d}".format(idx_num), '.', image_ext])
            elif self.zero_convention_in_name == 1:
                image_name = ''.join(["{:06d}".format(idx_num), '.', image_ext])
            else:
                raise NotImplementedError
            image_path = os.path.join(file_root, image_name)

            image_string = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image_string)
            image = tf.image.convert_image_dtype(image, tf.float32)
            triplet.append(tf.image.resize(image, resized_to))
        output = tf.concat(triplet, axis=2)
        return output

    def build_dataset(self):
        # Data path
        data = self.collect_file_paths_from_split()
        dataset = tf.data.Dataset.from_tensor_slices(data)
        # Decode image file (png or jpeg)
        dataset = dataset.map(lambda x: tf.py_function(self.parse_function, [x], Tout=tf.float32),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=self.buffer_size, reshuffle_each_iteration=True)
        dataset = dataset.batch(self.batch_size, True)  # Define batch triplets
        dataset = dataset.repeat(self.num_epoch)        # Define iteration params
        dataset = dataset.prefetch(1)
        return dataset

    def test_tf_string_split(self):
        data = tf.constant([os.path.join(self.split_folder, i) for i in os.listdir(self.split_folder)])
        dataset = tf.data.Dataset.from_tensor_slices(data)
        train_dataset = dataset.map(lambda x: tf.py_function(self.load_audio_file, [x], [tf.string]))
        for one_element in train_dataset:
            pass

def test_read_file():
    image_path = os.path.join(r"D:\MA\Struct2Depth\KITTI_odom_02\image_2\test", "000209.png")
    print(image_path)
    image_string = tf.io.read_file(image_path)
    image = tf.image.decode_png(image_string)
    plt.imshow(image), plt.show()


def test_training_loop(data_loader):
    dataset = data_loader.build_dataset()
    dataset_iter = iter(dataset)
    for epoch in range(data_loader.num_epoch):
        # for i, batch in enumerate(dataset):
        for i in range(data_loader.steps_per_epoch):
            batch = dataset_iter.get_next()
            tgt_image_stack, src_image_stack = batch[0,...,:3], batch[0,...,3:]
            print(src_image_stack.shape)
            fig = plt.figure(figsize=(3,1))
            fig.add_subplot(3,1,1)
            plt.imshow(src_image_stack[...,:3])
            fig.add_subplot(3,1,2)
            plt.imshow(tgt_image_stack)
            fig.add_subplot(3,1,3)
            plt.imshow(src_image_stack[..., 3:])
            plt.show()

if __name__ == '__main__':
    data_loader = DataLoader_KITTI_Raw(num_epoch=10, batch_size=6, dataset_for='train', split_name='eigen_zhou')
    test_training_loop(data_loader)
