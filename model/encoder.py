import tensorflow as tf

from layers import ConvBlock

class ConvEncoder(tf.keras.Model):
    def __init__(self,
                 input_shape=(4, 160, 192, 128),
                 data_format='channels_first',
                 kernel_size=(3, 3, 3),
                 groups=8,
                 dropout=0.2):
        super(tf.keras.Model, self).__init__()
        # Input layers.
        self.inp_layer = tf.keras.layers.Input(input_shape)
        self.inp_conv = tf.keras.layers.Conv3D(
                                filters=32,
                                kernel_size=kernel_size,
                                strides=1,
                                padding='same',
                                data_format=data_format)
        self.inp_dropout = tf.keras.layers.Dropout(dropout)

        # First ConvBlock: filters=32, x1.
        self.conv_block_32 = [ConvBlock(filters=32,
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
                                    kernel_size=kernel_size,
                                    data_format=data_format,
                                    groups=groups)] * 2
        self.conv_downsamp_64 = tf.keras.layers.Conv3D(
                                    filters=64,
                                    kernel_size=kernel_size,
                                    strides=2,
                                    padding='same',
                                    data_format=data_format)

        # Third ConvBlock: filters=128, x2.
        self.conv_block_128 = [ConvBlock(filters=128,
                                    kernel_size=kernel_size,
                                    data_format=data_format,
                                    groups=groups)] * 2
        self.conv_downsamp_128 = tf.keras.layers.Conv3D(
                                    filters=128,
                                    kernel_size=kernel_size,
                                    strides=2,
                                    padding='same',
                                    data_format=data_format)

        # Fourth ConvBlock: filters=256, x4.
        self.conv_block_256 = [ConvBlock(filters=128,
                                    kernel_size=kernel_size,
                                    data_format=data_format,
                                    groups=groups)] * 4

    def call(self, x):
        # Input layers.
        x = self.inp_layer(x)
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
        x = self.conv_downsamp(x)

        # Fourth ConvBlock: filters=256, x4.
        for conv in self.conv_block_256:
            x = conv(x)
        encoder_out = x

        # Return values after each ConvBlock for residuals later.
        return conv_out_32, conv_out_64, conv_out_128, encoder_out
