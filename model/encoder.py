import tensorflow as tf

from model.layers import ConvBlock
from utils.constants import *


class ConvEncoder(tf.keras.layers.Layer):
    def __init__(self,
                 data_format='channels_last',
                 kernel_size=3,
                 groups=8,
                 dropout=0.2,
                 reduction=4,
                 kernel_regularizer=None):
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
                dropout: float, optional
                    The dropout rate after initial convolution. Defaults to 0.2,
                    as used in the paper.
                reduction: int, optional
                    Reduction ratio for excitation size in squeeze-excitation layer.
                kernel_regularizer: tf.keras.regularizer callable, optional
                    Kernel regularizer for convolutional operations.
        """
        super(ConvEncoder, self).__init__()
        # Input layers.
        self.inp_conv = tf.keras.layers.Conv3D(
                                filters=ENC_CONV_LAYER_SIZE,
                                kernel_size=kernel_size,
                                strides=1,
                                padding='same',
                                data_format=data_format,
                                kernel_regularizer=kernel_regularizer)
        self.inp_dropout = tf.keras.layers.Dropout(dropout)

        # First ConvBlock: filters=32, x1.
        self.conv_block_0 = [ConvBlock(filters=ENC_CONV_BLOCK0_SIZE,
                                    kernel_size=kernel_size,
                                    data_format=data_format,
                                    groups=groups,
                                    reduction=reduction,
                                    kernel_regularizer=kernel_regularizer) for _ in range(ENC_CONV_BLOCK0_NUM)]
        self.conv_downsamp_0 = tf.keras.layers.Conv3D(
                                    filters=ENC_CONV_BLOCK0_SIZE,
                                    kernel_size=kernel_size,
                                    strides=2,
                                    padding='same',
                                    data_format=data_format,
                                    kernel_regularizer=kernel_regularizer)

        # Second ConvBlock: filters=64, x2.
        self.conv_block_1 = [ConvBlock(filters=ENC_CONV_BLOCK1_SIZE,
                                    kernel_size=kernel_size,
                                    data_format=data_format,
                                    groups=groups,
                                    reduction=reduction,
                                    kernel_regularizer=kernel_regularizer) for _ in range(ENC_CONV_BLOCK1_NUM)]
        self.conv_downsamp_1 = tf.keras.layers.Conv3D(
                                    filters=ENC_CONV_BLOCK1_SIZE,
                                    kernel_size=kernel_size,
                                    strides=2,
                                    padding='same',
                                    data_format=data_format,
                                    kernel_regularizer=kernel_regularizer)

        # Third ConvBlock: filters=128, x2.
        self.conv_block_2 = [ConvBlock(filters=ENC_CONV_BLOCK2_SIZE,
                                    kernel_size=kernel_size,
                                    data_format=data_format,
                                    groups=groups,
                                    reduction=reduction,
                                    kernel_regularizer=kernel_regularizer) for _ in range(ENC_CONV_BLOCK2_NUM)]
        self.conv_downsamp_2 = tf.keras.layers.Conv3D(
                                    filters=ENC_CONV_BLOCK2_SIZE,
                                    kernel_size=kernel_size,
                                    strides=2,
                                    padding='same',
                                    data_format=data_format,
                                    kernel_regularizer=kernel_regularizer)

        # Fourth ConvBlock: filters=256, x4.
        self.conv_block_3 = [ConvBlock(filters=ENC_CONV_BLOCK3_SIZE,
                                    kernel_size=kernel_size,
                                    data_format=data_format,
                                    groups=groups,
                                    reduction=reduction,
                                    kernel_regularizer=kernel_regularizer) for _ in range(ENC_CONV_BLOCK3_NUM)]

    def call(self, x, training=False):
        """Returns the forward pass of the ConvEncoder.

            {
                Initial Conv3D_32 -> Dropout,
                [ConvBlock_32] * 1 -> Downsample_32,
                [ConvBlock_64] * 2 -> Downsample_64,
                [ConvBlock_128] * 2 -> Downsample_128,
                [ConvBlock_256] * 4
            }

            Args:
                x: Tensor or np.ndarray
                    The input image to the encoder.
            Shape:
                If data_format == 'channels_first': shape=(4, 160, 192, 128).
                If data_format == 'channels_last': shape=(160, 192, 128, 4).
                
            Returns:
                -   Outputs from ConvBlocks of filter sizes 32, 64, and 128 for
                    residual connections in the decoder.
                -   Output of the forward pass.
        """
        # Input layers.
        x = self.inp_conv(x)
        if training:
            x = self.inp_dropout(x)

        # First ConvBlock: filters=32, x1.
        for conv in self.conv_block_0:
            x = conv(x)
        conv_out_0 = x
        x = self.conv_downsamp_0(x)

        # Second ConvBlock: filters=64, x2.
        for conv in self.conv_block_1:
            x = conv(x)
        conv_out_1 = x
        x = self.conv_downsamp_1(x)

        # Third ConvBlock: filters=128. x2.
        for conv in self.conv_block_2:
            x = conv(x)
        conv_out_2 = x
        x = self.conv_downsamp_2(x)

        # Fourth ConvBlock: filters=256, x4.
        for conv in self.conv_block_3:
            x = conv(x)
        encoder_out = x

        # Return values after each ConvBlock for residuals later.
        return conv_out_0, conv_out_1, conv_out_2, encoder_out
