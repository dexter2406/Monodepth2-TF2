import tensorflow as tf
from tensorflow.keras.layers import Conv2D
import numpy as np


class PoseDecoder(tf.keras.Model):
    def __init__(self, num_ch_enc, num_input_features=1, num_frames_to_predict_for=2, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features
        self.pose_scale = 0.01

        self.angles = None
        self.translations = None

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.relu = tf.keras.activations.relu

        self.convs_squeeze = Conv2D(filters=256, kernel_size=1, name='Conv_squeeze')

        pose_0_nopad = Conv2D(256, kernel_size=3, strides=stride, padding="valid", name='Conv_pose_0')
        pose_1_nopad = Conv2D(256, kernel_size=3, strides=stride, padding='valid', name='Conv_pose_1')
        pose_2_nopad = Conv2D(6 * num_frames_to_predict_for, kernel_size=1, strides=1, name='Conv_pose_2')
        self.convs_pose = [pose_0_nopad, pose_1_nopad, pose_2_nopad]

    def call(self, input_features, training=None, mask=None):
        # input_features = [f[-1] for f in input_features]  # if input have multiple sources: [5-features] * N
        # cat_features = [self.relu(self.convs_squeeze(f)) for f in input_features]
        # cat_features = tf.concat(out, 3)  # ch=3 for 'channel-last'
        """ pass encoder-features one by one, not pairwise """
        last_features = input_features[-1]
        out = self.convs_squeeze(last_features)
        out = self.relu(out)
        for i in range(3):
            if i != 2:
                out = tf.pad(out, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
            out = self.convs_pose[i](out)
            if i != 2:
                out = self.relu(out)

        out = tf.reduce_mean(out, [1, 2], keepdims=True)
        out = tf.reshape(out, [-1, self.num_frames_to_predict_for, 1, 6])
        out = tf.cast(self.pose_scale, dtype=tf.float32) * out

        self.angles = out[..., 3:]
        self.translations = out[..., :3]
        return {"angles": self.angles, "translations": self.translations}


def build_posenet():
    num_ch_enc = [64, 64, 128, 256, 512]
    shapes = [(1, 64, 96, 320), (1, 48, 160, 64), (1, 24, 80, 128), (1, 12, 40, 256), (1, 6, 20, 512)]
    dummy_inputs = [tf.random.uniform(shape=(shapes[i])) for i in range(len(shapes))]
    pose_decoder = PoseDecoder(num_ch_enc, num_input_features=1, num_frames_to_predict_for=2, stride=1)
    pose_decoder.predict(dummy_inputs)
    return pose_decoder, dummy_inputs


# -------- Archived below ---------


def load_weights_from_pkl(weights_path=None):
    import pickle
    with open("D:\MA\Recources\monodepth2-torch\models\pose_decoder.pkl", 'rb') as df:
        weights_dict = pickle.load(df)
    return weights_dict


def load_weights_pose_decoder():
    pose_decoder, dummpy_inputs = build_posenet()
    weights_dict = load_weights_from_pkl()
    print(weights_dict.keys())
    for layer in pose_decoder.layers:
        print(layer.name)
        name_weight = layer.name + '/weight'
        name_bias = layer.name + '/bias'
        print("load weights from dict file")
        weights = [weights_dict[name_weight],
                   weights_dict[name_bias]]
        layer.set_weights(weights)
    print("weights loaded")

    model_path = "pose_decoder"
    tf.keras.models.save_model(pose_decoder, model_path)
    decoder_reload = tf.saved_model.load(model_path)
    infer = decoder_reload.signatures['serving_default']
    feed_dict = {}
    for i in range(5):
        feed_dict['input_%d' % (i+1)] = dummpy_inputs[i]
    poses = infer(**feed_dict)
    for k, v in poses.items():
        print(k)
        print(v.shape)
        print(np.squeeze(v).shape)


if __name__ == '__main__':
    load_weights_pose_decoder()
