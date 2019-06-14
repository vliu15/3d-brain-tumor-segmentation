"""Contains custom convolutional decoder class."""
import tensorflow as tf

from model.layer_utils.resnet import ResnetBlock
from model.layer_utils.upsample import get_upsampling
from utils.constants import OUT_CH


class Decoder(tf.keras.layers.Layer):
    def __init__(self,
                 data_format='channels_last',
                 groups=8,
                 reduction=2,
                 l2_scale=1e-5,
                 upsampling='conv',
                 base_filters=16,
                 depth=4):
        """ Initializes the model decoder: consists of an alternating
            series of SENet blocks and upsampling layers.
        """
        super(Decoder, self).__init__()
        # Set up config for self.get_config() to serialize later.
        self.config = super(Decoder, self).get_config()
        self.config.update({'data_format': data_format,
                            'groups': groups,
                            'reduction': reduction,
                            'l2_scale': l2_scale,
                            'upsampling': upsampling,
                            'base_filters': base_filters,
                            'depth': depth})

        # Retrieve upsampling method.
        Upsample = get_upsampling(upsampling)

        # Build layers at all spatial levels.
        self.levels = []
        for i in range(depth-2, -1, -1):
            upsample = Upsample(
                            filters=base_filters*(2**i),
                            groups=groups,
                            data_format=data_format,
                            l2_scale=l2_scale)
            res = tf.keras.layers.Concatenate(
                            axis=-1 if data_format == 'channels_last' else 1)
            conv = ResnetBlock(
                            filters=base_filters*(2**i),
                            groups=groups,
                            reduction=reduction,
                            data_format=data_format,
                            l2_scale=l2_scale)
            self.levels.append([upsample, res, conv])

        # Output layer convolution.
        self.out_conv = tf.keras.layers.Conv3D(
                                filters=OUT_CH,
                                kernel_size=1,
                                strides=1,
                                padding='same',
                                activation='sigmoid',
                                data_format=data_format,
                                kernel_regularizer=tf.keras.regularizers.l2(l=l2_scale),
                                kernel_initializer='glorot_normal')

    def call(self, inputs, training=None):
        inputs, residuals = inputs
        # Iterate through spatial levels.
        for level, residual in zip(self.levels, residuals[::-1]):
            upsample, res, conv = level

            # Upsample.
            inputs = upsample(inputs, training=training)

            # Combine with residual from encoder.
            inputs = res([residual, inputs])

            # Pass through convolutional block.
            inputs = conv(inputs, training=training)

        # Map convolution to number of classes.
        inputs = self.out_conv(inputs)
        return inputs

    def get_config(self):
        return self.config
