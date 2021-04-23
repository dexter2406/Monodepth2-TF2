import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator as ImgDataGen
import matplotlib.pyplot as plt
import os
import cv2 as cv


class DataLoader(object):
    def __init__(self, num_epoch, batch_size):
        self.root = r"D:\MA\Struct2Depth\KITTI_odom_02\image_2\test"
        self.num_items = len(os.listdir(self.root))
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.num_frames_per_batch = 3
        self.steps_per_epoch = self.num_items // self.batch_size

    def parse_function(self, filepath):
        """Decode filenames to images
        Givn one file, find the neighbors and concat to shape [192,640,9]
        """
        index, img_ext = bytes.decode(filepath.numpy()).split("\\")[-1].split(".")
        triplet = []
        frame_idx = [0, -1, 1]
        for i in frame_idx:
            new_index = int(index)+i
            if new_index < 0 or new_index > self.num_items-1:
                new_index = int(index)
            image_name = "".join(["%06d"%new_index, '.', img_ext])
            image_path = os.path.join(self.root, image_name)
            image_string = tf.io.read_file(image_path)
            image = tf.image.decode_png(image_string)
            image = tf.image.convert_image_dtype(image, tf.float32)
            triplet.append(tf.image.resize(image, [192, 640]))
        output = tf.concat(triplet, axis=2)
        return output


    def triplet_parse_func(self,raw_images):
        "Make triplet in an unit of a batch"
        tgt = raw_images[1]
        src = tf.concat([raw_images[0], raw_images[2]], axis=2)
        return tgt, src


    def train_preprocess(self,image):
        """Augmentation
        Note: to get raw images, this function will not be used.
        Augmentation will be implemented after the images are loaded
        """
        image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image


    # TODO: add camera intrinsics into dataset as ZIP
    def build_dataset(self):
        root = r"D:\MA\Struct2Depth\KITTI_odom_02\image_2\test"
        data = tf.constant([os.path.join(root, i) for i in os.listdir(root)])
        dataset = tf.data.Dataset.from_tensor_slices(data)
        # Decode png
        dataset = dataset.map(lambda x: tf.py_function(self.parse_function, [x], Tout=tf.float32),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=self.num_items, reshuffle_each_iteration=True)
        # Define batch triplets
        dataset = dataset.batch(self.batch_size, True)
        # Define iteration params
        dataset = dataset.repeat(self.num_epoch)
        dataset = dataset.prefetch(1)

        return dataset


    def test_training_loop(self):
        dataset = self.build_dataset()
        for epoch in range(self.num_epoch):
            for i, batch in enumerate(dataset):
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

    def test_tf_string_split(self):
        root = r"D:\MA\Struct2Depth\KITTI_odom_02\image_2\test"
        data = tf.constant([os.path.join(root, i) for i in os.listdir(root)])
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

if __name__ == '__main__':
    data_loader = DataLoader()
    data_loader.test_training_loop(3,4)
    # data_loader.test_tf_string_split()
    # image_string = test_read_file()
