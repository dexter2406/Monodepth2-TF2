import tensorflow as tf
import cv2 as cv
from collections import OrderedDict
from tensorflow.keras.layers import Conv2D
import numpy as np


def ConvBlock(input_channel, filter_num, activate_type=None, pad_mode='reflect', id=-1, index=-1):
    name = None
    if id!=-1 and index!=-1:
        if activate_type is None:
            name = 'Conv3x3_%d_%d'%(id, index)
        else:
            name = 'ConvBlock_%d_%d' % (id, index)
    if pad_mode == 'reflect':
        padding = 'valid'
    else:
        padding = 'same'
    conv = Conv2D(filters=filter_num, kernel_size=3, activation=activate_type,
                  strides=1, padding=padding, use_bias=True, name=name)

    return conv


class DepthDecoder(tf.keras.Model):
    def __init__(self):
        super(DepthDecoder, self).__init__()
        self.num_ch_enc = [64, 64, 128, 256, 512]
        self.num_ch_dec = [16, 32, 64, 128, 256]
        self.scales = [0,1,2,3]  # range(4)
        self.num_output_channels = 1

        self.convs_0 = [None]*len(self.num_ch_dec)
        self.convs_1 = [None]*len(self.num_ch_dec)

        # todo: dispconv can be multiple output
        self.dispconv = [None]*len(self.scales)

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

        # for s in self.scales:
        #     self.convs[("dispconv", s)] = TF_Conv3x3(self.num_ch_dec[s], self.num_output_channels, pad_mode='reflect')
            # self.dispconv[s] = self.make_conv(self.num_ch_dec[s], self.num_output_channels, activate_type=None,
            #                                   pad_mode='reflect', type='disp', index=s)
        self.dispconv_0 = self.make_conv(self.num_ch_dec[0], self.num_output_channels, activate_type=None,
                                         pad_mode='reflect', type='disp', index=0)

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
            # if i in scales:
            #     x = tf.pad(x, **self.reflect_pad_kwargs)
            #     x = convs[("dispconv", i)](x)
                # outputs.append(tf.math.sigmoid(x))
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        # x = self.convs[("dispconv", 0)](x)
        x = self.dispconv_0(x)
        disp0 = tf.math.sigmoid(x)

        return disp0


def decoder_load_weights(decoder,
                         weights_path=r"D:\MA\Recources\monodepth2-torch\models\decoder_weights_dim2310.npy"):
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


def build_model(inputs):
    decoder = DepthDecoder()
    outputs = decoder.predict(inputs)
    decoder = decoder_load_weights(decoder)

    tf.keras.models.save_model(decoder, "decoder_test")

    # decoder.build(input_shape=[(1,96, 320, 64), (1,48, 160, 64),
    #               (1,24, 80, 128), (1,12, 40, 256),(1,6, 20, 512)])
    decoder.summary()
    # tf.keras.models.save_model(decoder, "decoder_test_1")

    print("Testing keras model...")
    # decoder_k = tf.keras.models.load_model("decoder_test_2")
    # decoder_k.summary()
    # outputs = decoder_k.predict(inputs)
    # for out in outputs:
    #     print(out.shape)

    print("Testing saved_model...")
    decoder_import = tf.saved_model.load("decoder_test")
    decoder_pb = decoder_import.signatures['serving_default']
    inputs = tuple(inputs)
    outputs = decoder_pb(inputs)
    for k, v in outputs:
        print(v.shape)


if __name__ == '__main__':
    inputs = [tf.random.uniform(shape=(1,96, 320, 64)),
              tf.random.uniform(shape=(1,48, 160, 64)),
              tf.random.uniform(shape=(1,24, 80, 128)),
              tf.random.uniform(shape=(1,12, 40, 256)),
              tf.random.uniform(shape=(1,6, 20, 512))]
    build_model(inputs)