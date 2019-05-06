import tensorflow as tf
from model.utils import GroupNormalization


class ConvLayer(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size=(3, 3, 3),
                 data_format='channels_last',
                 groups=8):
        super(ConvLayer, self).__init__()
        self.groupnorm = GroupNormalization(
                            groups=groups,
                            axis=-1 if data_format == 'channels_last' else 1)
        self.relu = tf.keras.layers.Activation('relu')
        self.conv3d = tf.keras.layers.Conv3D(
                            filters=filters,
                            kernel_size=kernel_size,
                            strides=1,
                            padding='same',
                            data_format=data_format)

    def call(self, x):
        x = self.groupnorm(x)
        x = self.relu(x)
        x = self.conv3d(x)
        return x


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 input_shape,
                 kernel_size=(3, 3, 3),
                 data_format='channels_last',
                 groups=8):
        super(ConvBlock, self).__init__()
        self.conv3d_ptwise = tf.keras.layers.Conv3D(
                                filters=filters,
                                input_shape=input_shape,
                                kernel_size=(1, 1, 1),
                                strides=1,
                                padding='same',
                                data_format=data_format)
        self.conv_layer1 = ConvLayer(filters, kernel_size=kernel_size, data_format=data_format, groups=8)
        self.conv_layer2 = ConvLayer(filters, kernel_size=kernel_size, data_format=data_format, groups=8)
        self.residual = tf.keras.layers.Add()

    def call(self, x):
        res = self.conv3d_ptwise(x)
        x = self.conv_layer1(res)
        x = self.conv_layer2(x)
        x = self.residual([res, x])
        return x
