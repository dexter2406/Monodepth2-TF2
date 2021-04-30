import tensorflow as tf
from tensorflow.keras.layers import Conv2D
import numpy as np


class PoseDecoder(tf.keras.Model):
    def __init__(self, num_ch_enc=(64, 64, 128, 256, 512),
                 num_input_features=1, num_frames_to_predict_for=2, stride=1):
        super(PoseDecoder, self).__init__()
        # (64, 64, 128, 256, 512)
        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features
        self.pose_scale = 0.01
        self.num_frames_to_predict_for = num_frames_to_predict_for
        self.relu = tf.keras.activations.relu

        self.convs_squeeze = Conv2D(filters=256, kernel_size=1, name='Conv_squeeze')

        pose_0_nopad = Conv2D(256, kernel_size=3, strides=stride, padding="valid", name='Conv_pose_0')
        pose_1_nopad = Conv2D(256, kernel_size=3, strides=stride, padding='valid', name='Conv_pose_1')
        pose_2_nopad = Conv2D(6*self.num_frames_to_predict_for, kernel_size=1, strides=1, name='Conv_pose_2')
        self.convs_pose = [pose_0_nopad, pose_1_nopad, pose_2_nopad]

    def call(self, input_features, training=None, mask=None):
        """ pass encoder-features pairwise
        output: [Batch, 2, 3] for angles and translations, respectively,
        - output[:, 0] is current->previous; output[:,1] for current->next
        """
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
        out = out * tf.cast(self.pose_scale, dtype=tf.float32)

        # todo: angles=out[... 3:] ??
        angles = out[..., 3:]
        translations = out[..., :3]

        return {"angles": angles, "translations": translations}


class PoseDecoder_new(tf.keras.Model):
    """num_frames_to_predict=2, i.e. only frame 1->2, no 2->1"""
    def __init__(self, num_frames_to_predict_for, num_ch_enc=(64, 64, 128, 256, 512),
                 num_input_features=1, stride=1, include_intrinsics=False):
        super(PoseDecoder_new, self).__init__()
        # (64, 64, 128, 256, 512)
        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features
        self.pose_scale = 0.01
        self.num_frames_to_predict_for = num_frames_to_predict_for
        self.include_intrinsics = include_intrinsics

        self.relu = tf.keras.activations.relu

        self.convs_squeeze = Conv2D(filters=256, kernel_size=1, name='Conv_squeeze')

        pose_0_nopad = Conv2D(256, kernel_size=3, strides=stride, padding="valid", name='Conv_pose_0')
        pose_1_nopad = Conv2D(256, kernel_size=3, strides=stride, padding='valid', name='Conv_pose_1')
        pose_2_nopad = Conv2D(6*self.num_frames_to_predict_for, kernel_size=1, strides=1, name='Conv_pose_2')
        self.convs_pose = [pose_0_nopad, pose_1_nopad, pose_2_nopad]

        conv_prop = {'filters': 2, 'kernel_size': [1, 1], 'strides': 1, 'padding': 'same'}
        self.conv_intrinsics = [
            Conv2D(**conv_prop, activation=tf.nn.softplus, name='Conv_foci'),
            Conv2D(**conv_prop, use_bias=False, name='Conv_offsets')
        ]

    def call(self, input_features, training=None, mask=None):
        """ pass encoder-features pairwise
        output: [Batch, 2, 3] for angles and translations, respectively,
        - output[:, 0] is current->previous; output[:,1] for current->next
        """
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
        out = out * tf.cast(self.pose_scale, dtype=tf.float32)

        # todo: angles=out[... 3:] ??
        angles = out[..., :3]
        translations = out[..., 3:]
        intrinsics_mat = tf.constant(0)
        if self.include_intrinsics:
            intrinsics_mat = self.add_intrinsics_head(last_features)
        return {"angles": angles, "translations": translations, "K": intrinsics_mat}

    def add_intrinsics_head(self, bottleneck, image_height=192, image_width=640):
        # todo: Image height and width - Original/Scaled?
        """Adds a head the preficts camera intrinsics.
        Args:
          bottleneck: A tf.Tensor of shape [B, 1, 1, C]
          image_height: A scalar tf.Tensor or an python scalar, the image height in pixels.
          image_width: the image width

        image_height and image_width are used to provide the right scale for the focal
        length and the offest parameters.

        Returns:
          a tf.Tensor of shape [B, 3, 3], and type float32, where the 3x3 part is the
          intrinsic matrix: (fx, 0, x0), (0, fy, y0), (0, 0, 1).
        """
        image_size = tf.constant([[image_width, image_height]], dtype=tf.float32)
        focal_lens = self.conv_intrinsics[0](bottleneck)
        focal_lens = tf.squeeze(focal_lens, axis=(1, 2)) * image_size

        offsets = self.conv_intrinsics[1](bottleneck)
        offsets = (tf.squeeze(offsets, axis=(1, 2)) + 0.5) * image_size

        foc_inv = tf.linalg.diag(focal_lens)
        intrinsic_mat = tf.concat([foc_inv, tf.expand_dims(offsets, -1)], axis=2)
        last_row = tf.cast(tf.tile([[[0.0, 0.0, 1.0]]], [bottleneck.shape[0], 1, 1]), dtype=tf.float32)
        intrinsic_mat = tf.concat([intrinsic_mat, last_row], axis=1)
        return intrinsic_mat


def build_posenet():
    num_ch_enc = [64, 64, 128, 256, 512]
    shapes = [(1, 64, 96, 320), (1, 48, 160, 64), (1, 24, 80, 128), (1, 12, 40, 256), (1, 6, 20, 512)]
    dummy_inputs = [tf.random.uniform(shape=(shapes[i])) for i in range(len(shapes))]
    pose_decoder = PoseDecoder(num_ch_enc, num_input_features=1, num_frames_to_predict_for=1, stride=1)
    outputs = pose_decoder.predict(dummy_inputs)
    return pose_decoder, dummy_inputs, outputs


def combine_pose_params(pred_pose_raw, curr2prev=False, curr2next=False):
    """Combine pose angles and translations, from raw dictionary"""

    if curr2prev and curr2next:
        raise NotImplementedError
    if not curr2prev and not curr2next:
        print("concat poses for both frames")
        return tf.concat([pred_pose_raw['angles'], pred_pose_raw['translations']], axis=3)
    else:
        print("concat poses for one frame")
        seq_choice = 0 if curr2prev else 1
        return tf.concat([pred_pose_raw['angles'][:, seq_choice],
                          pred_pose_raw['translations'][:, seq_choice]], axis=2)

if __name__ == '__main__':
    # pose_decoder, dummy_inputs, outputs = build_posenet()
    # for k,v in outputs.items():
    #     print(k,"\t", v.shape)
    res = {}
    res["angles"] = tf.random.uniform(shape=(1,2,1,3))
    res["translations"] = tf.random.uniform(shape=(1,2,1,3))
    out_ctp = combine_pose_params(res, curr2prev=True)
    out_ctn = combine_pose_params(res, curr2next=True)
    a = tf.concat([out_ctn, out_ctn], axis=1)
    print(a.shape)


# -------- Archived below ---------

"""
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
"""


