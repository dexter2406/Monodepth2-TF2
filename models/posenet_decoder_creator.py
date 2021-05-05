import tensorflow as tf
from tensorflow.keras.layers import Conv2D
import numpy as np
import time


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

        angles = out[..., 3:]
        translations = out[..., :3]
        return {"angles": angles, "translations": translations}


class PoseDecoder_exp(tf.keras.Model):
    """num_frames_to_predict=2, i.e. only frame 1->2, no 2->1"""
    def __init__(self, pose_num=1, num_ch_enc=(64, 64, 128, 256, 512),
                 stride=1):
        super(PoseDecoder_exp, self).__init__()
        self.num_ch_enc = num_ch_enc
        # self.num_ch_dec = [input_ch, 16, 32, 64, 128, 256]
        self.num_ch_dec = [16, 32, 64, 128, 256]
        self.pose_scale = tf.cast(0.01, dtype=tf.float32)
        self.pose_num = pose_num
        self.relu = tf.keras.activations.relu

        self.convs_squeeze = Conv2D(filters=256, kernel_size=1, name='Conv_squeeze')

        pose_0_nopad = Conv2D(256, kernel_size=3, strides=stride, padding="valid", name='Conv_pose_0')
        pose_1_nopad = Conv2D(256, kernel_size=3, strides=stride, padding='valid', name='Conv_pose_1')
        pose_2_nopad = Conv2D(self.pose_num*6, kernel_size=1, strides=1, name='Conv_pose_2')

        self.convs_pose = [pose_0_nopad, pose_1_nopad, pose_2_nopad]

    def call(self, input_features, training=None, mask=None):
        """ pass encoder-features pairwise
        """
        last_feature = input_features[-1]
        print(last_feature.shape)
        squeezed_feature = self.convs_squeeze(last_feature)
        print(squeezed_feature.shape)
        backgrd_motion = self.decode_background_motion(squeezed_feature)
        backgrd_motion *= self.pose_scale
        angles = backgrd_motion[..., :, :, :3]
        trans = backgrd_motion[..., :, :, 3:]
        # assert angles.shape[1] == trans.shape[1] == 1 and trans.shape[-1] == 3

        return {"angles": angles, "translations": trans}

    def padded_conv(self, conv, inp, padding='CONSTANT'):
        inp = tf.pad(inp, [[0, 0], [1, 1], [1, 1], [0, 0]], mode=padding)
        return conv(inp)

    def decode_background_motion(self, squeezed_feature, num_scales=3):
        out = self.relu(squeezed_feature)
        for i in range(num_scales):
            if i != 2:
                out = tf.pad(out, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
            out = self.convs_pose[i](out)
            if i != 2:
                out = self.relu(out)

        out = tf.reduce_mean(out, [1, 2], keepdims=True)
        backgrd_motion = tf.reshape(out, [-1, self.pose_num, 1, 6])
        return backgrd_motion


class ResidualTranslationNet(PoseDecoder_exp):
    def __init__(self, has_depth, lite_mode_idx=(4,3,2,1), do_automasking=False):
        super(ResidualTranslationNet, self).__init__(has_depth)
        self.out_ch = 8 if has_depth else 6
        conv_prop_1 = {'kernel_size': 1, 'strides': 1, 'use_bias': False, 'padding': 'valid'}
        self.conv_res_motion = Conv2D(self.out_ch, **conv_prop_1, name='unrefined_res_trans')
        self.num_ch_dec = [self.out_ch, 16, 32, 64, 128, 256]
        self.lite_mode_idx = lite_mode_idx
        self.do_automasking = do_automasking
        self.conv_refine_motion = []
        for i, ch in enumerate(self.num_ch_dec):
            idx = len(self.num_ch_dec) - i - 1     # 5->0
            pref = 'Refine%d' % idx
            conv_prop_2 = {'filters': ch, 'kernel_size': 3, 'strides': 1,
                           'activation': 'relu', 'padding': 'valid'}
            tmp = []
            for j in range(3):
                tmp.append(Conv2D(**conv_prop_2, name=pref + 'Conv%d' % (j + 1)))
            tmp.append(Conv2D(self.out_ch, [1, 1], strides=1,
                              activation=None, use_bias=False, name=pref + 'Conv4'))
            self.conv_refine_motion.append(tmp)

    def call(self, features, training=None, mask=None):
        """ Decoding residual translation
        features: [input_image, *encoded_features]
        res_trans_field: Tensor, [B,H,W,1]
            Residual translation field
        """
        squeezed_feature = tf.keras.layers.AveragePooling2D()(features[-1])
        res_trans_field = self.conv_res_motion(squeezed_feature)

        for i in range(len(features)-1, -1, -1):
            # t1 = time.perf_counter()
            use_lite_mode = i in self.lite_mode_idx
            # print('idx=%d uses lite_mode? ' % i, use_lite_mode)
            res_trans_field = self._refine_motion_field(res_trans_field, features[i],
                                                        idx=i, lite_mode=use_lite_mode)
            # print(res_trans_field.shape, input_features[i].shape, "%.2fms" % ((time.perf_counter() - t1) * 1000))

        res_trans_field *= self.pose_scale
        if self.do_automasking:
            res_trans_field = self.apply_automask_to_res_trans(res_trans_field)

        return res_trans_field

    def _refine_motion_field(self, motion_field, conv, idx, lite_mode=False):
        """Refine residual motion map"""
        # print('motion filed', motion_field.shape)
        # print('conv', conv.shape)
        upsamp_motion_field = tf.cast(tf.image.resize(motion_field, conv.shape[1:3]), dtype=tf.float32)
        # print('upsamp_motion_field shape', upsamp_motion_field.shape)
        conv_inp = tf.concat([upsamp_motion_field, conv], axis=3)
        i = len(self.conv_refine_motion) - 1 - idx  # backwards index in list
        if not lite_mode:
            output_1 = self.padded_conv(self.conv_refine_motion[i][0], conv_inp)
            conv_inp = self.padded_conv(self.conv_refine_motion[i][1], conv_inp)
            output_2 = self.padded_conv(self.conv_refine_motion[i][2], conv_inp)
            conv_inp = tf.concat([output_1, output_2], axis=-1)
        else:
            # use lite mode to save computation
            conv_inp = self.padded_conv(self.conv_refine_motion[i][0], conv_inp)

        output = self.conv_refine_motion[i][3](conv_inp)
        # print('output shape', conv_inp.shape)
        output = upsamp_motion_field + output
        return output

    @staticmethod
    def apply_automask_to_res_trans(residual_trans):
        """Masking out the residual translations by thresholding on their mean values."""
        sq_residual_trans = tf.sqrt(
            tf.reduce_sum(residual_trans ** 2, axis=3, keepdims=True))
        mean_sq_residual_trans = tf.reduce_mean(
            sq_residual_trans, axis=[0, 1, 2])
        # A mask of shape [B, h, w, 1]
        mask_residual_translation = tf.cast(
            sq_residual_trans > mean_sq_residual_trans, residual_trans.dtype
        )
        residual_trans *= mask_residual_translation
        return residual_trans


    # def add_intrinsics_head(self, bottleneck, image_height=192, image_width=640):
    #     """Adds a head the preficts camera intrinsics.
    #     Args:
    #       bottleneck: A tf.Tensor of shape [B, 1, 1, C]
    #       image_height: A scalar tf.Tensor or an python scalar, the image height in pixels.
    #       image_width: the image width
    #
    #     image_height and image_width are used to provide the right scale for the focal
    #     length and the offest parameters.
    #
    #     Returns:
    #       a tf.Tensor of shape [B, 3, 3], and type float32, where the 3x3 part is the
    #       intrinsic matrix: (fx, 0, x0), (0, fy, y0), (0, 0, 1).
    #     """
    #     image_size = tf.constant([[image_width, image_height]], dtype=tf.float32)
    #     focal_lens = self.conv_intrinsics[0](bottleneck)
    #     focal_lens = tf.squeeze(focal_lens, axis=(1, 2)) * image_size
    #
    #     offsets = self.conv_intrinsics[1](bottleneck)
    #     offsets = (tf.squeeze(offsets, axis=(1, 2)) + 0.5) * image_size
    #
    #     foc_inv = tf.linalg.diag(focal_lens)
    #     intrinsic_mat = tf.concat([foc_inv, tf.expand_dims(offsets, -1)], axis=2)
    #     last_row = tf.cast(tf.tile([[[0.0, 0.0, 1.0]]], [bottleneck.shape[0], 1, 1]), dtype=tf.float32)
    #     intrinsic_mat = tf.concat([intrinsic_mat, last_row], axis=1)
    #     return intrinsic_mat

def make_data(new_version=True):
    if new_version:
        shapes = [(2,192,640,4), (2, 96, 320, 64), (2, 48, 160, 64), (2, 24, 80, 128), (2, 12, 40, 256), (2, 6, 20, 512)]
    else:
        shapes = [(2, 96, 320, 64), (2, 48, 160, 64), (2, 24, 80, 128), (2, 12, 40, 256), (2, 6, 20, 512)]
    dummy_inputs = [tf.random.uniform(shape=(shapes[i])) for i in range(len(shapes))]
    return dummy_inputs

def build_posenet(pose_dec, dummy_inputs):
    # num_ch_enc = [64, 64, 128, 256, 512]
    # pose_dec = PoseDecoder(num_ch_enc, num_input_features=1, num_frames_to_predict_for=1, stride=1)
    outputs = pose_dec(dummy_inputs)
    return pose_dec, dummy_inputs, outputs


def exp_res_trans():
    # pose_dec = PoseDecoder_exp(pose_num=1, has_depth=True)
    dummy_inp = make_data(new_version=True)
    # pose_dec, dum_inp, outputs = build_posenet(pose_dec)
    # pose_dec1 = PoseDecoder(num_frames_to_predict_for=1)
    # pose_dec1, dum_inp1, outputs1 = build_posenet(pose_dec1, new_version=False)
    res_trans_net = ResidualTranslationNet(has_depth=True)
    res_trans_net(dummy_inp)

    # pose_dec.summary()
    t0 = time.perf_counter()
    for i in range(100):
        t1 = time.perf_counter()
        output = res_trans_net(dummy_inp)
        print("{:.3f}".format(time.perf_counter()-t1))
    print("average:", (time.perf_counter() - t0)/100)

    print(output.shape)


if __name__ == '__main__':
    # pose_decoder, dummy_inputs, outputs = build_posenet()
    # for k,v in outputs.items():
    #     print(k,"\t", v.shape)
    # print(a.shape)
    exp_res_trans()


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


