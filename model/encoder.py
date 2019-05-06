import tensorflow as tf

from model.layers import ConvBlock

class ConvEncoder(tf.keras.layers.Layer):
    def __init__(self,
                 data_format='channels_last',
                 kernel_size=(3, 3, 3),
                 groups=8,
                 dropout=0.2):
        super(ConvEncoder, self).__init__()
        # Input layers.
        self.inp_conv = tf.keras.layers.Conv3D(
                                filters=32,
                                kernel_size=kernel_size,
                                strides=1,
                                padding='same',
                                data_format=data_format)
        self.inp_dropout = tf.keras.layers.Dropout(dropout)

        # First ConvBlock: filters=32, x1.
        self.conv_block_32 = [ConvBlock(filters=32,
                                    input_shape=(160, 192, 128, 32),
                                    kernel_size=kernel_size,
                                    data_format=data_format,
                                    groups=groups)]
        self.conv_downsamp_32 = tf.keras.layers.Conv3D(
                                    filters=32,
                                    kernel_size=kernel_size,
                                    strides=2,
                                    padding='same',
                                    data_format=data_format)

        # Second ConvBlock: filters=64, x2.
        self.conv_block_64 = [ConvBlock(filters=64,
                                    input_shape=(80, 96, 64, 32),
                                    kernel_size=kernel_size,
                                    data_format=data_format,
                                    groups=groups),
                              ConvBlock(filters=64,
                                    input_shape=(80, 96, 64, 64),
                                    kernel_size=kernel_size,
                                    data_format=data_format,
                                    groups=groups)]
        self.conv_downsamp_64 = tf.keras.layers.Conv3D(
                                    filters=64,
                                    kernel_size=kernel_size,
                                    strides=2,
                                    padding='same',
                                    data_format=data_format)

        # Third ConvBlock: filters=128, x2.
        self.conv_block_128 = [ConvBlock(filters=128,
                                    input_shape=(40, 48, 32, 64),
                                    kernel_size=kernel_size,
                                    data_format=data_format,
                                    groups=groups),
                               ConvBlock(filters=128,
                                    input_shape=(40, 48, 32, 128),
                                    kernel_size=kernel_size,
                                    data_format=data_format,
                                    groups=groups)]
        self.conv_downsamp_128 = tf.keras.layers.Conv3D(
                                    filters=128,
                                    kernel_size=kernel_size,
                                    strides=2,
                                    padding='same',
                                    data_format=data_format)

        # Fourth ConvBlock: filters=256, x4.
        self.conv_block_256 = [ConvBlock(filters=256,
                                    input_shape=(20, 24, 16, 128),
                                    kernel_size=kernel_size,
                                    data_format=data_format,
                                    groups=groups)] + \
                              [ConvBlock(filters=256,
                                    input_shape=(20, 24, 16, 256),
                                    kernel_size=kernel_size,
                                    data_format=data_format,
                                    groups=groups)]

    def call(self, x):
        """Returns the forward pass of the ConvEncoder.

            See https://arxiv.org/pdf/1810.11654.pdf for more details.
            The Encoder consists of a series of ConvBlocks, connected
            by convolutional downsampling layers. Each ConvBlock is
            1 pointwise convolutional layer + 2 ConvLayers, each of which
            consists of a [GroupNormalization -> Relu -> Conv] series.

            Args:
                x: the input image to the encoder.
            Shape:
                if data_format == 'channels_first': shape=(4, 160, 192, 128)
                if data_format == 'channels_last': shape=(160, 192, 128, 4)
                
            Returns:
                -   Outputs from ConvBlocks of filter sizes 32, 64, and 128 for
                    residual connections in the decoder.
                -   Output of the forward pass.
        """
        x = tf.expand_dims(x, axis=0)

        # Input layers.
        x = self.inp_conv(x)
        x = self.inp_dropout(x)

        # First ConvBlock: filters=32, x1.
        for conv in self.conv_block_32:
            x = conv(x)
        conv_out_32 = x
        x = self.conv_downsamp_32(x)

        # Second ConvBlock: filters=64, x2.
        for conv in self.conv_block_64:
            x = conv(x)
        conv_out_64 = x
        x = self.conv_downsamp_64(x)

        # Third ConvBlock: filters=128. x2.
        for conv in self.conv_block_128:
            x = conv(x)
        conv_out_128 = x
        x = self.conv_downsamp_128(x)

        # Fourth ConvBlock: filters=256, x4.
        for conv in self.conv_block_256:
            x = conv(x)
        encoder_out = x

        # Return values after each ConvBlock for residuals later.
        return conv_out_32, conv_out_64, conv_out_128, encoder_out
