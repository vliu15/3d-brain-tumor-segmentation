import tensorflow as tf

from model.layer_utils.group_norm import GroupNormalization


class ConvDownsample(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 data_format='channels_last',
                 kernel_size=3,
                 groups=8,
                 kernel_regularizer=tf.keras.regularizers.l2(l=1e-5),
                 kernel_initializer='he_normal',
                 **kwargs):
        super(ConvDownsample, self).__init__()
        self.config = super(ConvDownsample, self.).get_config()
        self.config.update({'filters': filters,
                            'data_format': data_format,
                            'kernel_size': kernel_size,
                            'groups': groups,
                            'kernel_regularizer': kernel_regularizer,
                            'kernel_initializer': kernel_initializer})

        self.conv = tf.keras.layers.Conv3D(
                                filters=filters,
                                kernel_size=kernel_size,
                                strides=2,
                                padding='same',
                                data_format=data_format,
                                kernel_regularizer=kernel_regularizer,
                                kernel_initializer=kernel_initializer)
        self.groupnorm = GroupNormalization(
                                groups=groups,
                                axis=-1 if data_format == 'channels_last' else 1)
        self.relu = tf.keras.layers.Activation('relu')

    def __call__(self, inputs):
        inputs = self.conv(inputs)
        inputs = self.groupnorm(inputs)
        inputs = self.relu(inputs)
        return inputs

    def get_config(self):
        return self.config


class AvgDownsample(tf.keras.layers.Layer):
    def __init__(self,
                 data_format='channels_last',
                 **kwargs):
        super(AvgDownsample, self).__init__()
        self.config = super(AvgDownsample, self).get_config()
        self.config.update({'data_format': data_format})

        self.avgpool = tf.keras.layers.AveragePooling3D(
                            pool_size=2,
                            strides=2,
                            padding='same',
                            data_format=data_format)

    def __call__(self, inputs):
        inputs = self.avgpool(inputs)
        return inputs

    def get_config(self):
        return self.config


class MaxDownsample(tf.keras.layers.Layer):
    def __init__(self,
                 data_format='channels_last',
                 **kwargs):
        super(MaxDownsample, self).__init__()
        self.config = super(MaxDownsample, self).get_config()
        self.config.update({'daat_format': data_format})

        self.maxpool = tf.keras.layers.MaxPooling3D(
                            pool_size=2,
                            strides=2,
                            padding='same',
                            data_format=data_format)

    def __call__(self, inputs):
        inputs = self.maxpool(inputs)
        return inputs

    def get_config(self):
        return self.config
