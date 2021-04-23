import tensorflow as tf
import cv2 as cv
from collections import OrderedDict
from tensorflow.keras.layers import Conv2D
import numpy as np


class DepthDecoder_full(tf.keras.Model):
    def __init__(self):
        super(DepthDecoder_full, self).__init__()
        self.num_ch_enc = [64, 64, 128, 256, 512]
        self.num_ch_dec = [16, 32, 64, 128, 256]
        self.scales = [0,1,2,3]  # range(4)
        self.num_output_channels = 1

        self.convs_1 = [None]*len(self.num_ch_dec)
        self.convs_2 = [None]*len(self.num_ch_dec)

        # todo: dispconv can be multiple output
        self.dispconv = [None]*len(self.scales)

        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            # in: 512, 256, 128, 64, 32
            # out: 256, 128, 64, 32, 16
            # convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            self.convs_1[i] = self.make_conv(num_ch_in, num_ch_out, pad_mode='reflect', activate_type='elu',
                                             type='conv_0', index=i)
            # self.convs[("upconv", i, 0)] = TF_ConvBlock(num_ch_in, num_ch_out, pad_mode='reflect')

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            # in: 256+256, 128+128, 64+64, 32+64, 16
            # out: 256, 128, 64, 32, 16
            # self.convs[("upconv", i, 1)] = TF_ConvBlock(num_ch_in, num_ch_out, pad_mode='reflect')
            self.convs_2[i] = self.make_conv(num_ch_in, num_ch_out, pad_mode='reflect', activate_type='elu',
                                             type='conv_1', index=i)

        for s in self.scales:
            # self.convs[("dispconv", s)] = TF_Conv3x3(self.num_ch_dec[s], self.num_output_channels, pad_mode='reflect')
            num_ch_in = self.num_ch_dec[s]
            num_ch_out = self.num_output_channels
            self.dispconv[s] = self.make_conv(num_ch_in, num_ch_out, activate_type=None,
                                              pad_mode='reflect', type='disp', index=s)
        # self.dispconv_0 = self.make_conv(self.num_ch_dec[0], self.num_output_channels, activate_type=None,
        #                                  pad_mode='reflect', type='disp', index=0)

    def make_conv(self, input_channel, filter_num, activate_type=None, pad_mode='reflect',
                  type:str=None, index=-1, input_shape:tuple=None):
        name = None
        if type is not None and index != -1:
            name = ''.join([type, '_%d'%index])
        if pad_mode == 'reflect':
            padding = 'valid'
        else:
            padding = 'same'
        conv = Conv2D(filters=filter_num, kernel_size=3, activation=activate_type,
                      strides=1, padding=padding, use_bias=True, name=name)

        return conv

    """
    # connect layers
    """
    def call(self, input_features, training=None, mask=None):
        ch_axis = 3
        x = input_features[-1]
        outputs = {}
        for i in range(4, -1, -1):
            x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            # x = self.convs[("upconv", i, 0)](x)
            x = self.convs_1[i](x)
            x = [tf.keras.layers.UpSampling2D()(x)]
            if i > 0:
                x += [input_features[i - 1]]
            x = tf.concat(x, ch_axis)
            x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            # x = self.convs[("upconv", i, 1)](x)
            x = self.convs_2[i](x)
            if i in self.scales:
                out = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
                out = self.dispconv[i](out)
                outputs["output_%d" % i] = tf.math.sigmoid(out)
        # x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        # x = self.convs[("dispconv", 0)](x)
        # x = self.dispconv[0](x)
        # disp0 = tf.math.sigmoid(x)

        return outputs


class DepthDecoder_one_output(tf.keras.Model):
    def __init__(self):
        super(DepthDecoder_one_output, self).__init__()
        self.num_ch_enc = [64, 64, 128, 256, 512]
        self.num_ch_dec = [16, 32, 64, 128, 256]
        self.scales = [0,1,2,3]  # range(4)
        self.num_output_channels = 1

        self.convs_0 = [None]*len(self.num_ch_dec)
        self.convs_1 = [None]*len(self.num_ch_dec)

        # todo: dispconv can be multiple output
        out_nums = len(self.scales)
        self.dispconv = [None] * out_nums
        self.disp = [None] * out_nums   # result

        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            # in: 512, 256, 128, 64, 32
            # out: 256, 128, 64, 32, 16
            # convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            self.convs_0[i] = self.make_conv(num_ch_in, num_ch_out, pad_mode='reflect', activate_type='elu',
                                             type='conv_0', index=i)
            # self.convs[("upconv", i, 0)] = TF_ConvBlock(num_ch_in, num_ch_out, pad_mode='reflect')

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            # in: 256+256, 128+128, 64+64, 32+64, 16
            # out: 256, 128, 64, 32, 16
            # self.convs[("upconv", i, 1)] = TF_ConvBlock(num_ch_in, num_ch_out, pad_mode='reflect')
            self.convs_1[i] = self.make_conv(num_ch_in, num_ch_out, pad_mode='reflect', activate_type='elu',
                                             type='conv_1', index=i)

        for s in self.scales:
            # self.convs[("dispconv", s)] = self.make_conv(self.num_ch_dec[s], self.num_output_channels, pad_mode='reflect')
            self.dispconv[s] = self.make_conv(self.num_ch_dec[s], self.num_output_channels, activate_type=None,
                                              pad_mode='reflect', type='disp', index=s)
        # self.dispconv_0 = self.make_conv(self.num_ch_dec[0], self.num_output_channels, activate_type=None,
        #                                  pad_mode='reflect', type='disp', index=0)

    def make_conv(self, input_channel, filter_num, activate_type=None, pad_mode='reflect',
                  type:str=None, index=-1, input_shape:tuple=None):
        name = None
        if type is not None and index != -1:
            name = ''.join([type, '_%d'%index])
        if pad_mode == 'reflect':
            padding = 'valid'
        else:
            padding = 'same'
        conv = Conv2D(filters=filter_num, kernel_size=3, activation=activate_type,
                      strides=1, padding=padding, use_bias=True, name=name)

        return conv

    """
    # connect layers, can be 1- or 4-output
    """
    def call(self, input_features, training=None, mask=None):
        outputs = []
        ch_axis = 3
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            # x = self.convs[("upconv", i, 0)](x)
            x = self.convs_0[i](x)
            x = [tf.keras.layers.UpSampling2D()(x)]
            if i > 0:
                x += [input_features[i - 1]]
            x = tf.concat(x, ch_axis)
            x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
            # x = self.convs[("upconv", i, 1)](x)
            x = self.convs_1[i](x)
            if i in self.scales:
                x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
                x = self.dispconv[i](x)
                outputs.append(tf.math.sigmoid(x))
        # x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        # x = self.dispconv_0(x)
        # disp0 = tf.math.sigmoid(x)

        return outputs

def decoder_load_weights(decoder,
                         weights_path=None):
    if weights_path is None:
        exit("No weights path")

    decoder_weights = np.load(weights_path, allow_pickle=True)
    # decoder_weights = decoder_weights[::-1]
    reind = [8,6,4,2,0,9,7,5,3,1,10]
    weights_grouped = [decoder_weights[i] for i in reind]
    ind = 0
    for l in decoder.layers:
        print(l.name)
        weights = l.get_weights()
        if len(weights) == 0:
            print("no weigths")
        else:
            print(weights[0].shape, "\t", weights[1].shape)
            print(weights_grouped[ind][0].shape, "\t", weights_grouped[ind][1].shape)
            new_weights = weights_grouped[ind]
            l.set_weights(new_weights)
            print("loading the %dnd conv layer...", ind)
            ind += 1
    print("DONE")
    return decoder


def build_model(inputs, model_type=None, verbose=True, check_output=False, save=False):
    decoder = DepthDecoder()
    outputs = decoder.predict(inputs)
    # decoder = decoder_load_weights(decoder)
    if verbose:
        decoder.summary()
        print("output shapes")
        print(type(outputs))
        for k, v in outputs.items():
            print(k,"\t",v.shape)

    save_name = 'decoder_test'
    if model_type is None:
        return

    elif model_type == 'keras':
        print("Testing keras model...")
        if save:
            tf.keras.models.save_model(save_name+'.h5')
            if check_output:
                dec_reload = tf.keras.models.load_model(save_name+'.h5')
                outputs = dec_reload.predict(inputs)
                for out in outputs:
                    print(out.shape)

    elif model_type == 'pb':
        if save:
            tf.keras.models.save_model(save_name)
            if check_output:
                decoder_import = tf.saved_model.load(save_name)
                decoder_pb = decoder_import.signatures['serving_default']
                feed_dict = {}
                for i in range(1,6):
                    feed_dict['input_%d'%i] = inputs[i-1]
                outputs = decoder_pb(**feed_dict)
                for k, v in outputs.items():
                    print(v.shape)
    else:
        raise NotImplementedError

    return decoder


def add_dispconv_weights(decoder, weights_path, record):
    with open(weights_path, 'rb') as df:
        weights_dict = pickle.load(df)

    print(weights_dict['conv_0_0'][0].shape)
    print(weights_dict.keys())
    for layer in decoder.layers:
        print(layer.name)
        weights = layer.get_weights()
        if "disp" in layer.name:
            new_weights = weights_dict[layer.name]
            layer.set_weights(new_weights)
            record.append(layer.name)
            print(weights[0].shape, "\t", weights[1].shape)
            # print(weights_dict[l.name][0].shape, "\t", weights_dict[l.name][1].shape)
            # new_weights = (weights_dict[l.name])
            print("loading the layer: %s...", layer.name)
    print(" ------- disp DONE ---------")
    return decoder


def add_otherconvs_weights(decoder, saved_model_path, record):
    decoder_reload = tf.keras.models.load_model(saved_model_path)
    weights_dict = {}
    for layer in decoder_reload.layers:
        if "conv" in layer.name:
            print("save layer %s in dict"%layer.name)
            weights_dict[layer.name] = layer.get_weights()

    for layer in decoder.layers:
        weights = layer.get_weights()
        if "conv" in layer.name:
            new_weights = weights_dict[layer.name]
            layer.set_weights(new_weights)
            record.append(layer.name)
            print(weights[0].shape, "\t", weights[1].shape)
            print("loading the layer: %s...", layer.name)
    print(" ------- rest DONE ---------")
    return decoder


if __name__ == '__main__':
    inputs = [tf.random.uniform(shape=(1,96, 320, 64)),
              tf.random.uniform(shape=(1,48, 160, 64)),
              tf.random.uniform(shape=(1,24, 80, 128)),
              tf.random.uniform(shape=(1,12, 40, 256)),
              tf.random.uniform(shape=(1,6, 20, 512))]

    decoder = build_model(inputs, model_type='keras', verbose=True)
    weights_path_disp = r"D:\MA\Recources\monodepth2-torch\models\depthdecoder_weights_full_outs.pkl"
    saved_model_path = r"D:\MA\Recources\monodepth2-torch\models\depth_decoder_singlet"
    loaded_layer_record = []
    decoder = add_dispconv_weights(decoder, weights_path_disp, loaded_layer_record)
    decoder = add_otherconvs_weights(decoder, saved_model_path, loaded_layer_record)
    print("loaded layers are: ")
    print(loaded_layer_record)

    tf.keras.models.save_model(decoder, 'models/decoder_full-test')
