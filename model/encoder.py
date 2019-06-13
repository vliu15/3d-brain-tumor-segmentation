"""Contains custom convolutional encoder class."""
import tensorflow as tf

from model.resnet_block import ConvBlock, ConvLayer
from model.layer_utils.getters import get_downsampling
from utils.constants import *


class ConvEncoder(tf.keras.layers.Layer):
    def __init__(self,
                 data_format='channels_last',
                 kernel_size=3,
                 groups=8,
                 reduction=2,
                 kernel_regularizer=tf.keras.regularizers.l2(l=1e-5),
                 kernel_initializer='he_normal',
                 downsampling='max',
                 normalization='group'):
        """Initializes the model encoder.

            See https://arxiv.org/pdf/1810.11654.pdf for more details.
            The Encoder consists of a series of ConvBlocks, connected
            by convolutional downsampling layers. Each ConvBlock is
            1 pointwise convolutional layer + 2 ConvLayers, each of which
            consists of a [GroupNormalization -> Relu -> Conv] series.

            Args:
                data_format: str, optional
                    The format of the input data. Must be either 'channels_last'
                    or 'channels_first'. Defaults to `channels_last` for CPU
                    development. 'channels_first is used in the paper.
                kernel_size: int, optional
                    The size of all convolutional kernels. Defaults to 3,
                    as used in the paper.
                groups: int, optional
                    The size of each group for GroupNormalization. Defaults to
                    8, as used in the paper.
                reduction: int, optional
                    Reduction ratio for excitation size in squeeze-excitation layer.
                kernel_regularizer: tf.keras.regularizer callable, optional
                    Kernel regularizer for convolutional operations.
                kernel_initializer: tf.keras.initializers callable, optional
                    Kernel initializer for convolutional operations.
        """
        super(ConvEncoder, self).__init__()
        # Set up config for self.get_config() to serialize later.
        self.config = super(ConvEncoder, self).get_config()
        self.config.update({'data_format': data_format,
                            'kernel_size': kernel_size,
                            'groups': groups,
                            'reduction': reduction,
                            'kernel_regularizer': tf.keras.regularizers.serialize(kernel_regularizer),
                            'kernel_initializer': kernel_initializer,
                            'downsampling': downsampling,
                            'normalization': normalization})

        # Retrieve downsampling method.
        Downsample = get_downsampling(downsampling)

        # First level.
        self.conv_block_0 = [ConvBlock(
                                filters=ENC_CONV_BLOCK0_SIZE,
                                kernel_size=kernel_size,
                                groups=groups,
                                reduction=reduction,
                                data_format=data_format,
                                kernel_regularizer=kernel_regularizer,
                                kernel_initializer=kernel_initializer,
                                normalization=normalization) for _ in range(ENC_CONV_BLOCK0_NUM)]
        self.conv_downsamp_0 = Downsample(
                                filters=ENC_CONV_BLOCK0_SIZE,
                                kernel_size=kernel_size,
                                groups=groups,
                                data_format=data_format,
                                kernel_regularizer=kernel_regularizer,
                                kernel_initializer=kernel_initializer,
                                normalization=normalization)

        # Second level.
        self.conv_block_1 = [ConvBlock(
                                filters=ENC_CONV_BLOCK1_SIZE,
                                kernel_size=kernel_size,
                                groups=groups,
                                reduction=reduction,
                                data_format=data_format,
                                kernel_regularizer=kernel_regularizer,
                                kernel_initializer=kernel_initializer,
                                normalization=normalization) for _ in range(ENC_CONV_BLOCK1_NUM)]
        self.conv_downsamp_1 = Downsample(
                                filters=ENC_CONV_BLOCK1_SIZE,
                                kernel_size=kernel_size,
                                groups=groups,
                                data_format=data_format,
                                kernel_regularizer=kernel_regularizer,
                                kernel_initializer=kernel_initializer,
                                normalization=normalization)

        # Third level.
        self.conv_block_2 = [ConvBlock(
                                filters=ENC_CONV_BLOCK2_SIZE,
                                kernel_size=kernel_size,
                                groups=groups,
                                reduction=reduction,
                                data_format=data_format,
                                kernel_regularizer=kernel_regularizer,
                                kernel_initializer=kernel_initializer,
                                normalization=normalization) for _ in range(ENC_CONV_BLOCK2_NUM)]
        self.conv_downsamp_2 = Downsample(
                                filters=ENC_CONV_BLOCK2_SIZE,
                                kernel_size=kernel_size,
                                groups=groups,
                                data_format=data_format,
                                kernel_regularizer=kernel_regularizer,
                                kernel_initializer=kernel_initializer,
                                normalization=normalization)

        # Fourth level.
        self.conv_block_3 = [ConvBlock(
                                filters=ENC_CONV_BLOCK3_SIZE,
                                kernel_size=kernel_size,
                                groups=groups,
                                reduction=reduction,
                                data_format=data_format,
                                kernel_regularizer=kernel_regularizer,
                                kernel_initializer=kernel_initializer,
                                normalization=normalization) for _ in range(ENC_CONV_BLOCK3_NUM)]

    def call(self, inputs, training=None):
        # First level.
        for conv in self.conv_block_0:
            inputs = conv(inputs, training=training)
        conv_out_0 = inputs
        inputs = self.conv_downsamp_0(inputs, training=training)

        # Second level.
        for conv in self.conv_block_1:
            inputs = conv(inputs, training=training)
        conv_out_1 = inputs
        inputs = self.conv_downsamp_1(inputs, training=training)

        # Third level.
        for conv in self.conv_block_2:
            inputs = conv(inputs, training=training)
        conv_out_2 = inputs
        inputs = self.conv_downsamp_2(inputs, training=training)

        # Fourth level.
        for conv in self.conv_block_3:
            inputs = conv(inputs, training=training)

        # Return values after each level for residuals later.
        return (conv_out_0, conv_out_1, conv_out_2, inputs)

    def get_config(self):
        return self.config
