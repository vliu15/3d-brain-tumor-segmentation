"""Contains custom convolutional encoder class."""
import tensorflow as tf

from model.layer_utils.resnet import ResnetBlock
from model.layer_utils.downsample import get_downsampling


class Encoder(tf.keras.layers.Layer):
    def __init__(self,
                 data_format='channels_last',
                 groups=8,
                 reduction=2,
                 l2_scale=1e-5,
                 downsampling='conv',
                 base_filters=16,
                 depth=4):
        """ Initializes the model encoder: consists of an alternating
            series of ResNet blocks with DenseNet connections and downsampling layers.

            References:
                - [Densely Connected Residual Networks](https://arxiv.org/pdf/1608.06993.pdf)
        """
        super(Encoder, self).__init__()
        # Set up config for self.get_config() to serialize later.
        self.config = super(Encoder, self).get_config()
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
            convs = []
            for j in range(i + 1):
                conv = ResnetBlock(
                            filters=base_filters*(2**i),
                            groups=groups,
                            reduction=reduction,
                            data_format=data_format,
                            l2_scale=l2_scale)
                dense = tf.keras.layers.Concatenate(
			                axis=-1 if data_format == 'channels_last' else 1) if j > 0 else None
                convs.append([conv, dense])
            
            # Concatenate before downsampling.
            concat = tf.keras.layers.Concatenate(
                            axis=-1 if data_format == 'channels_last' else 1) if i > 0 else None

            # No downsampling at deepest spatial level.
            downsample = Downsample(
                            filters=base_filters*(2**i),
                            groups=groups,
                            data_format=data_format,
                            l2_scale=l2_scale) if i < depth - 1 else None
            
            self.levels.append([convs, concat, downsample])

    def call(self, inputs, training=None):
        residuals = []

        # Iterate through spatial levels.
        for i, level in enumerate(self.levels):
            convs, concat, downsample = level

            # Cache intermediate activations for concatenation.
            cache = []

            # Iterate through convolutional blocks.
            for conv, dense in convs:
                if dense is not None:
                    inputs = dense([inputs] + cache)
                inputs = conv(inputs, training=training)
                cache.append(inputs)

            # Concatenate all activations in the layer.
            if concat is not None:
                inputs = concat(cache)
            
            # Store residuals for use in decoder.
            residuals.append(inputs)

            # No downsampling at bottom spatial level.
            if downsample is not None:
                inputs = downsample(inputs, training=training)

        # Return values after each spatial level for decoder.
        return residuals

    def get_config(self):
        return self.config
