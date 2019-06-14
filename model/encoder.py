"""Contains custom convolutional encoder class."""
import tensorflow as tf

from model.resnet_block import ConvBlock, ConvLayer
from model.layer_utils.downsample import get_downsampling


class ConvEncoder(tf.keras.layers.Layer):
    def __init__(self,
                 data_format='channels_last',
                 groups=8,
                 reduction=2,
                 l2_scale=1e-5,
                 downsampling='conv',
                 base_filters=16,
                 depth=4):
        """ Initializes the model encoder: consists of an alternating
            series of SENet blocks and downsampling layers.
        """
        super(ConvEncoder, self).__init__()
        # Set up config for self.get_config() to serialize later.
        self.config = super(ConvEncoder, self).get_config()
        self.config.update({'data_format': data_format,
                            'groups': groups,
                            'reduction': reduction,
                            'l2_scale': l2_scale,
                            'downsampling': downsampling,
                            'base_filters': base_filters,
                            'depth': depth})

        # Retrieve downsampling method.
        Downsample = get_downsampling(downsampling)

        # Build layers at all spatial levels.
        self.levels = []
        for i in range(depth):
            convs = [ConvBlock(
                            filters=base_filters*(2**i),
                            groups=groups,
                            reduction=reduction,
                            data_format=data_format,
                            l2_scale=l2_scale) for _ in range(2)]

            # No downsampling at deepest spatial level.
            if i < depth:
                downsample = [Downsample(
                                filters=base_filters*(2**i),
                                groups=groups,
                                data_format=data_format,
                                l2_scale=l2_scale)]
                self.levels.append(convs + downsample)
            else:
                self.levels.append(convs)

    def call(self, inputs, training=None):
        residuals = []

        # Iterate through spatial levels.
        for i, level in enumerate(self.levels):
            convs = level[:-1]
            downsample = level[-1]

            # Iterate through convolutional blocks.
            for conv in convs:
                inputs = conv(inputs, training=training)
            
            # Store residuals for use in decoder.
            residuals.append(inputs)

            # No downsampling at bottom spatial level.
            if i < len(self.levels) - 1:
                inputs = downsample(inputs, training=training)

        # Return values after each spatial level for decoder.
        return residuals

    def get_config(self):
        return self.config
