"""Contains custom upsampling classes."""
import tensorflow as tf

from model.layer_utils.getters import get_normalization


class ConvUpsample(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size=3,
                 groups=8,
                 data_format='channels_last',
                 kernel_regularizer=tf.keras.regularizers.l2(l=1e-5),
                 kernel_initializer='he_normal',
                 normalization='group',
                 **kwargs):
        super(ConvUpsample, self).__init__()
        super(ConvDownsample, self).__init__()
        self.config = super(ConvUpsample, self).get_config()
        self.config.update({'filters': filters,
                            'data_format': data_format,
                            'kernel_size': kernel_size,
                            'groups': groups,
                            'kernel_regularizer': kernel_regularizer,
                            'kernel_initializer': kernel_initializer,
                            'normalization': normalization})

        # Retrieve normalization layer.
        Normalization = get_normalization(normalization)

        self.conv = tf.keras.layers.Conv3DTranspose(
                                filters=filters,
                                kernel_size=kernel_size,
                                strides=2,
                                padding='same',
                                data_format=data_format)
        try:
            self.norm = Normalization(
                            groups=groups,
                            axis=-1 if data_format == 'channels_last' else 1)
        except:
            self.norm = Normalization(
                            axis=-1 if data_format == 'channels_last' else 1)
        self.relu = tf.keras.layers.Activation('relu')

    def __call__(self, inputs, training=None):
        inputs = self.conv(inputs)
        inputs = self.norm(inputs, training=training)
        inputs = self.relu(inputs)
        return inputs

    def get_config(self):
        return self.config


class LinearUpsample(tf.keras.layers.Layer):
    def __init__(self,
                 data_format='channels_last',
                 **kwargs):
        super(LinearUpsample, self).__init__()
        self.config = super(LinearUpsample, self).get_config()
        self.config.update({'data_format': data_format})

        self.linear = tf.keras.layers.UpSampling3D(
                                size=2,
                                data_format=data_format)

    def __call__(self, inputs, training=None):
        inputs = self.linear(inputs)
        return inputs

    def get_config(self):
        return self.config
