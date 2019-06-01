import tensorflow as tf

from model.resnet_block import ConvBlock, ConvLayer
from model.layers.downsample import ConvDownsample, AvgDownsample, MaxDownsample
from utils.constants import *


def get_downsampling(downsampling):
    if downsampling == 'max':
        return MaxDownsample
    elif downsampling == 'avg':
        return AvgDownsample
    else:
        return ConvDownsample


class ConvEncoder(tf.keras.layers.Layer):
    def __init__(self,
                 data_format='channels_last',
                 kernel_size=3,
                 groups=8,
                 reduction=2,
                 kernel_regularizer=tf.keras.regularizers.l2(l=1e-5),
                 kernel_initializer='he_normal',
                 use_se=False,
                 downsampling='max',
                 **kwargs):
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
                            'use_se': use_se})

        # Retrieve downsampling method.
        Downsample = get_downsampling(downsampling)

        # Input layers.
        self.inp_conv = ConvLayer(
                                filters=ENC_CONV_LAYER_SIZE,
                                **kwargs)

        # First ConvBlock: filters=32, x1.
        self.conv_block_0 = [ConvBlock(filters=ENC_CONV_BLOCK0_SIZE, **kwargs)
                                for _ in range(ENC_CONV_BLOCK0_NUM)]
        self.conv_downsamp_0 = Downsample(**kwargs)

        # Second ConvBlock: filters=64, x2.
        self.conv_block_1 = [ConvBlock(filters=ENC_CONV_BLOCK1_SIZE, **kwargs)
                                for _ in range(ENC_CONV_BLOCK1_NUM)]
        self.conv_downsamp_1 = Downsample(**kwargs)

        # Third ConvBlock: filters=128, x2.
        self.conv_block_2 = [ConvBlock(filters=ENC_CONV_BLOCK2_SIZE, **kwargs)
                                for _ in range(ENC_CONV_BLOCK2_NUM)]
        self.conv_downsamp_2 = Downsample(**kwargs)

        # Fourth ConvBlock: filters=256, x4.
        self.conv_block_3 = [ConvBlock(filters=ENC_CONV_BLOCK3_SIZE, **kwargs)
                                for _ in range(ENC_CONV_BLOCK3_NUM)]

    def call(self, inputs):
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
        inputs = self.inp_conv(inputs)

        # First ConvBlock: filters=32, x1.
        for conv in self.conv_block_0:
            inputs = conv(inputs)
        conv_out_0 = inputs
        inputs = self.conv_downsamp_0(inputs)

        # Second ConvBlock: filters=64, x2.
        for conv in self.conv_block_1:
            inputs = conv(inputs)
        conv_out_1 = inputs
        inputs = self.conv_downsamp_1(inputs)

        # Third ConvBlock: filters=128. x2.
        for conv in self.conv_block_2:
            inputs = conv(inputs)
        conv_out_2 = inputs
        inputs = self.conv_downsamp_2(inputs)

        # Fourth ConvBlock: filters=256, x4.
        for conv in self.conv_block_3:
            inputs = conv(inputs)

        # Return values after each ConvBlock for residuals later.
        return (conv_out_0, conv_out_1, conv_out_2, inputs)

    def get_config(self):
        return self.config
