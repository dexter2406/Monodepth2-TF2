import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense


class BasicBlock_pad(tf.keras.layers.Layer):

    def __init__(self, filter_num, stride=1, padding_mode='reflect', block_id=-1, layer_id=-1):
        super(BasicBlock_pad, self).__init__()
        padding_options = {'reflect': {'mode': 'CONSTANT', 'paddings': [[0, 0], [1, 1], [1, 1], [0, 0]]},
                           'constant': {'mode': 'CONSTANT', 'paddings': [[0, 0], [1, 1], [1, 1], [0, 0]]}}
        self.pad_1 = padding_options[padding_mode]
        padding = 'valid' if padding_mode != 'same' else 'same'
        prefix = 'conv%d_%d/' % (block_id, layer_id)
        # 1
        self.conv1 = Conv2D(filters=filter_num, kernel_size=(3, 3), strides=stride,
                            use_bias=False, padding=padding, name=''.join([prefix, 'conv_1']))
        self.bn1 = BatchNormalization(name=''.join([prefix, 'BatchNorm_1']))
        self.a1 = Activation('relu')
        # 2
        self.conv2 = Conv2D(filters=filter_num, kernel_size=(3, 3),
                            use_bias=False, padding=padding)
        self.bn2 = BatchNormalization(name=''.join([prefix, 'BatchNorm_2']))
        # residual_path为True时，对输入进行下采样，即用1x1的卷积核做卷积操作，保证x能和F(x)维度相同，顺利相加
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(Conv2D(filters=filter_num, kernel_size=(1, 1), strides=stride,
                                       use_bias=False, padding='same', name=''.join([prefix, 'downsample'])))
            self.downsample.add(BatchNormalization(name='downsample/BatchNorm_3'))
        else:
            self.downsample = lambda x: x
        # 最后的relu
        self.a2 = Activation('relu')

    def call(self, x, training=None, **kwargs):

        # ----- PADDING -----
        out = tf.pad(x, **self.pad_1)
        # -------------------
        out = self.conv1(out)
        out = self.bn1(out, training=training)
        out = self.a1(out)

        # ----- PADDING -----
        out = tf.pad(out, **self.pad_1)
        # -------------------
        out = self.conv2(out, training=training)
        out = self.bn2(out)

        identity = self.downsample(x)

        out = tf.keras.layers.add([identity, out])
        out = self.a2(out)
        return out


def check_weights(enc_weights):
    for layer in enc_weights:
        print(type(layer))
        for i in range(len(layer)):
            print(layer[i].shape)
        print("==========")


def load_ext_weights_enc(model=None):
    import numpy as np
    path = r'D:\MA\Recources\monodepth2-torch\models\pose_enc_weights_enum.pkl'
    enc_weights = np.load(path, allow_pickle=True)
    weight_ind = 0
    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        if i == 0:
            # conv
            layer.set_weights([enc_weights[weight_ind]])
            print(enc_weights[weight_ind].shape, "vs. ", weights[0].shape)
            weight_ind += 1
            print("Conv0: weight_%d / total_%d--------" % (weight_ind, len(enc_weights)))
        if i ==1 :
            # BN
            layer.set_weights(enc_weights[weight_ind: weight_ind+4])
            print(enc_weights[weight_ind].shape, "vs. ", weights[0].shape)
            weight_ind += 4
            print("Conv0/BN: weight_%d / total_%d--------" % (weight_ind, len(enc_weights)))
        if i >= 4:
            reind = [0,1,2,5,6,7, 3,4,8,9, 10,11,12,15,16,17, 13,14,18,19] if i == 4 else \
                [0,1,2,5,6,7,10,11,12, 3,4,8,9,13,14, 15,16,17,20,21,22, 18,19,23,24]
            print("reorganizing indexes: ", reind)
            w_imported = enc_weights[weight_ind: weight_ind + len(reind)]
            print(enc_weights[weight_ind].shape, "vs. ", weights[0].shape)
            weight_ind += len(reind)
            print("weights count: ", len(w_imported))
            w_regroup = [w_imported[j] for j in reind]
            for w1, w2 in zip(w_regroup, weights):
                print(w1.shape," vs. ",w2.shape)

            layer.set_weights(w_regroup)
            print("Seq.%d weight_%d / total_%d--------" % (i - 3, weight_ind, len(enc_weights)))

    assert weight_ind == len(enc_weights)
    # model_test.save("models/res18_encoder.h5", include_optimizer=False)


def set_weights_enc():
    # inputs = tf.keras.layers.Input(shape=(192,640,3))
    pose_encoder = ResNet18_new([2, 2, 2, 2])
    # outs = encoder.call(inputs=inputs, training=True)
    pose_encoder.build(input_shape=(None, 192, 640, 3))
    load_ext_weights_enc(model=pose_encoder)

    input_arr = tf.random.uniform(shape=(1, 192, 640, 3))
    outputs = pose_encoder.predict(input_arr)
    print(outputs[4].shape)
    #
    pose_encoder.summary()
    tf.keras.models.save_model(pose_encoder, "pose_encoder_one_input")
    print("saved")


def check_model(model_path=None, keras=False, pb=False):
    if model_path is None:
        print("please specify name")
        return
    if keras:
        encoder_reload = tf.keras.models.load_model(model_path)
        print("reloaded")
        encoder_reload.summary()
    elif pb:
        encoder_reload = tf.saved_model.load(model_path)
        infer = encoder_reload.signatures['serving_default']
        print("reloaded")
        dummy_in = tf.random.uniform(shape=(1, 192, 640, 3))
        res = infer(dummy_in)
        for k, v in res.items():
            print(k,"\t", v.numpy().shape)
    return encoder_reload


def load_weights_multi_image(model_multi=None, num_input=2):
    """Load weigths from one-input resnet18 to pair-wise input mode"""
    model_single = check_model("models/pose_encoder_one_input", keras=True, pb=False)
    conv0_weights_no_bias = []

    for layer in model_single.layers:
        if layer.name == 'conv0':
            weights = layer.get_weights()
            weights_double = tf.concat([weights[0], weights[0]], axis=2) / num_input
            print(weights_double.shape)
            conv0_weights_no_bias.append(weights_double)

    for layer in model_multi.layers:
        if layer.name == 'conv0':
            layer.set_weights(conv0_weights_no_bias)


def build_pose_encoder_pair_input(verbose=False):
    """Create Multi-input resnet18 for PoseEncoder"""
    pose_encoder = ResNet18_new([2,2,2,2])
    dummy_in = tf.concat([tf.random.uniform(shape=(1, 192, 640, 3)),
                          tf.random.uniform(shape=(1, 192, 640, 3))], axis=3)
    pose_encoder.predict(dummy_in)
    if verbose:
        pose_encoder.summary()
    load_weights_multi_image(pose_encoder)
    print("Weigths for  pair-input encoder has done.")
    return pose_encoder


"""Previous Version"""
# BasicBlock and BasicBlock_pad 是分开的
class BasicBlock_nopad(tf.keras.layers.Layer):

    def __init__(self, filter_num, stride=1, block_id=-1, layer_id=-1):
        super(BasicBlock_nopad, self).__init__()
        prefix = 'conv%d_%d/' % (block_id, layer_id)
        # 1
        self.conv1 = Conv2D(filters=filter_num, kernel_size=(3, 3), strides=stride,
                            use_bias=False, padding="same", name=''.join([prefix, 'conv_1']))
        self.bn1 = BatchNormalization(name=''.join([prefix, 'BatchNorm_1']))
        self.a1 = Activation('relu')
        # 2
        self.conv2 = Conv2D(filters=filter_num, kernel_size=(3, 3),
                            use_bias=False, padding="same")
        self.bn2 = BatchNormalization(name=''.join([prefix, 'BatchNorm_2']))
        # residual_path为True时，对输入进行下采样，即用1x1的卷积核做卷积操作，保证x能和F(x)维度相同，顺利相加
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(Conv2D(filters=filter_num, kernel_size=(1, 1), strides=stride,
                                       use_bias=False, padding="same", name=''.join([prefix, 'downsample'])))
            self.downsample.add(BatchNormalization(name='downsample/BatchNorm_3'))
        else:
            self.downsample = lambda x: x
        # 最后的relu
        self.a2 = Activation('relu')

    def call(self, x, training=None, **kwargs):
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.a1(out)
        out = self.conv2(out, training=training)
        out = self.bn2(out)

        identity = self.downsample(x)

        out = tf.keras.layers.add([identity, out])
        out = self.a2(out)
        return out


class ResNet18_new(tf.keras.Model):
    def __init__(self, block_list=(2, 2, 2, 2), initial_filters=64, padding_mode='constant'):
        super(ResNet18_new, self).__init__()
        padding = 'valid' if padding_mode != 'same' else 'same'
        padding_options = {'reflect': {'mode': 'CONSTANT', 'paddings': [[0, 0], [3, 3], [3, 3], [0, 0]]},
                           'constant': {'mode': 'CONSTANT', 'paddings': [[0, 0], [3, 3], [3, 3], [0, 0]]}}
        self.pad_3 = padding_options[padding_mode]

        self.num_blocks = len(block_list)  # 共有几个block
        self.block_list = block_list
        self.out_filters = initial_filters

        self.conv1 = Conv2D(filters=self.out_filters, kernel_size=(7, 7), strides=2,
                            use_bias=False, padding=padding, name='conv0')
        self.bn1 = BatchNormalization(name='conv0/BatchNorm')
        self.a1 = Activation('relu', name='conv0/ReLU')

        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same", name='maxpool')

        self.layer1 = self._make_basic_block_layer(filter_num=64, blocks=block_list[0], block_id=1)
        self.layer2 = self._make_basic_block_layer(filter_num=128, blocks=block_list[1], stride=2, block_id=2)
        self.layer3 = self._make_basic_block_layer(filter_num=256, blocks=block_list[2], stride=2, block_id=3)
        self.layer4 = self._make_basic_block_layer(filter_num=512, blocks=block_list[3], stride=2, block_id=4)

    def _make_basic_block_layer(self, filter_num, blocks, stride=1, block_id=-1):
        res_block = tf.keras.Sequential(name='seq_%d' % block_id)
        res_block.add(BasicBlock_pad(filter_num, stride=stride, block_id=block_id, layer_id=1, padding_mode='constant'))

        for i in range(1, blocks):
            res_block.add(BasicBlock_nopad(filter_num, stride=1, block_id=block_id, layer_id=i + 1))

        return res_block

    def call(self, inputs, training=None, mask=None):
        outputs = []
        # ----- PADDING -----
        x = tf.pad(inputs, **self.pad_3)
        # ----- PADDING -----
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.a1(x)
        outputs.append(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)

        outputs.append(x)
        x = self.layer2(x, training=training)
        outputs.append(x)
        x = self.layer3(x, training=training)
        outputs.append(x)
        x = self.layer4(x, training=training)
        outputs.append(x)

        return outputs


if __name__ == '__main__':
    # set_weights_enc()
    # check_model("models/pose_encoder_one_input", keras=True, pb=False)
    pose_encoder = build_pose_encoder_pair_input(verbose=False)
    pose_encoder.summary()
    tf.keras.models.save_model(pose_encoder, "models/pose_encoder")


