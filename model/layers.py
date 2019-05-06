import tensorflow as tf
from utils import GroupNormalization

class ConvLayer(tf.keras.Model):
    def __init__(self,
                 filters,
                 kernel_size=(3, 3, 3),
                 data_format='channels_first',
                 groups=8):
        super(ConvLayer, self).__init__()
        self.groupnorm = GroupNormalization(
                            groups=groups,
                            axis=1 if data_format == 'channels_first' else 0)
        self.activation = tf.keras.layers.Activation('relu')
        self.conv3d = tf.keras.layers.Conv3D(
                            filters=filters,
                            kernel_size=kernel_size,
                            strides=1,
                            padding='same',
                            data_format=data_format)

    def call(x):
        x = self.groupnorm(x)
        x = self.activation(x)
        x = self.conv3d(x)
        return x

class ConvBlock(tf.keras.Model):
    def __init__(self,
                 filters,
                 n_layers=2,
                 kernel_size=(3, 3, 3),
                 data_format='channels_first',
                 groups=8):
        super(ConvBlock, self).__init__()
        self.conv3d_ptwise = tf.keras.layers.Conv3D(
                                filters=filters,
                                kernel_size=(1, 1, 1),
                                strides=1,
                                padding='same',
                                data_format=data_format)
        self.conv_layer1 = ConvLayer(filters, kernel_size=kernel_size, data_format=data_format, groups=8)
        self.conv_layer2 = ConvLayer(filters, kernel_size=kernel_size, data_format=data_format, groups=8)
        self.down_sample = tf.keras.layers.Conv3D(
                                filters=filters,
                                kernel_size=kernel_size,
                                strides=1,
                                padding='same',
                                data_format=data_format)
        self.residual = tf.keras.layers.Add()

    def call(x):
        res = self.conv3d_ptwise(x)
        x = self.conv_layer1(res)
        x = self.conv_layer2(x)
        x = self.residual([res, x])
        return x
