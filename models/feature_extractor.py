import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Dense, Input


class WideResNet(tf.keras.Model):
    """https://arxiv.org/pdf/1605.07146.pdf"""
    def __init__(self, initial_filters=64, padding_mode='reflect', norm_inp=True, input_shape:tuple=None):
        super(WideResNet, self).__init__()
        self.norm_inp = norm_inp
        padding = 'valid' if padding_mode != 'same' else 'same'
        padding_options = {'reflect': {'mode': 'REFLECT', 'paddings': [[0, 0], [1, 1], [1, 1], [0, 0]]},
                           'constant': {'mode': 'CONSTANT', 'paddings': [[0, 0], [1, 1], [1, 1], [0, 0]]}}
        self.pad_1 = padding_options[padding_mode]

        self.conv1 = Conv2D(filters=initial_filters, kernel_size=(3, 3), strides=1,
                            use_bias=False, padding=padding, name='conv0')
        self.bn1 = BatchNormalization(name='conv0/BatchNorm')
        self.a1 = Activation('relu', name='conv0/ReLU')
        self.conv2 = Conv2D(filters=initial_filters, kernel_size=(3, 3), strides=1,
                            use_bias=False, padding=padding, name='conv1')
        self.bn2 = BatchNormalization(name='conv1/BatchNorm')
        self.a2 = Activation('relu', name='conv1/ReLU')

        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same", name='maxpool')

        self.resblock1 = BasicBlock_pad(filter_num=64, stride=1, block_id=1, layer_id=1, padding_mode='reflect')
        self.resblock2 = BasicBlock_pad(filter_num=64, stride=1, block_id=1, layer_id=2, padding_mode='reflect')

        self.resblock3 = BasicBlock_pad(filter_num=128, stride=2, block_id=2, layer_id=1, padding_mode='reflect')
        self.resblock4 = BasicBlock_pad(filter_num=128, stride=1, block_id=2, layer_id=2, padding_mode='reflect')

        self.resblock5 = BasicBlock_pad(filter_num=256, stride=2, block_id=3, layer_id=1, padding_mode='reflect')
        self.resblock6 = BasicBlock_pad(filter_num=256, stride=1, block_id=3, layer_id=2, padding_mode='reflect')

        self.ga = tf.keras.layers.GlobalAvgPool2D()
        self.dense = Dense(units=128)

        self.bn3 = BatchNormalization(name='out/BatchNorm')
        self.l2 = tf.math.l2_normalize
        if input_shape is not None:
            self.call(Input(input_shape))

    def call(self, inputs, training=None, mask=None):
        if self.norm_inp:
            inputs = (inputs - 0.5) * 2
        x = tf.pad(inputs, **self.pad_1)
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.a1(x)

        x = tf.pad(x, **self.pad_1)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.a2(x)
        x = self.pool1(x)

        x = self.resblock1(x, training=training)
        x = self.resblock2(x, training=training)        # /2, 32
        extract_0 = self.pool1(x)                       # flow to the next block output

        x = self.resblock3(x, training=training)
        x = self.resblock4(x, training=training)        # /4, 64
        extract_1 = tf.concat([x, extract_0], axis=-1)  # /4, 96, receive flow from previous block

        x = self.resblock5(extract_1, training=training)
        x = self.resblock6(x, training=training)        # /8, 128

        x = self.ga(x)
        x = self.dense(x)                           # 128
        x = self.bn3(x)
        out = self.l2(x)

        return out

    # def call(self, inputs, training=None, mask=None):
    #     if self.norm_inp:
    #         inputs = (inputs - 0.5) * 2
    #     x = tf.pad(inputs, **self.pad_1)
    #     x = self.conv1(x)
    #     x = self.bn1(x, training=training)
    #     x = self.a1(x)
    #
    #     x = tf.pad(x, **self.pad_1)
    #     x = self.conv2(x)
    #     x = self.bn2(x, training=training)
    #     x = self.a2(x)
    #     x = self.pool1(x)
    #
    #     x = self.resblock1(x, training=training)
    #     x = self.resblock2(x, training=training)    # 32
    #
    #     x = self.resblock3(x, training=training)
    #     x = self.resblock4(x, training=training)    # 64
    #
    #     x = self.resblock5(x, training=training)
    #     x = self.resblock6(x, training=training)    # 128
    #
    #     x = self.ga(x)
    #     x = self.dense(x)                           # 128
    #     x = self.bn3(x)
    #     out = self.l2(x)
    #     return out


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
        # self.downsample = []
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(Conv2D(filters=filter_num, kernel_size=(1, 1), strides=stride,
                                       use_bias=False, padding='same', name=''.join([prefix, 'downsample'])))
            self.downsample.add(BatchNormalization(name='downsample/BatchNorm_3'))
            # self.downsample.append(Conv2D(filters=filter_num, kernel_size=(1, 1), strides=stride,
            #                               use_bias=False, padding='same', name=''.join([prefix, 'downsample'])))
            # self.downsample.append(BatchNormalization(name='downsample/BatchNorm_3'))
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
        out = self.conv2(out)
        out = self.bn2(out, training=training)

        # for downsample in self.downsample:
        identity = self.downsample(x)

        out = tf.keras.layers.add([identity, out])
        out = self.a2(out)
        return out


if __name__ == '__main__':
    import time
    input_shape = (80, 80, 3)
    net = WideResNet(input_shape=input_shape)
    net(tf.keras.Input(input_shape))
    net.summary()
    inp = tf.random.uniform((10,80,80,3))
    out = net(inp)
    t1 = time.perf_counter()
    for i in range(1000):
        out = net(inp)
    print("{}s".format(time.perf_counter() - t1))
