import tensorflow as tf

from model.layers import ConvBlock


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 input_shape,
                 kernel_size=(3, 3, 3),
                 data_format='channels_last',
                 groups=8):
        """Initializes one level of upsampling in the convolutional decoder.

            Each decoder block consists of a pointwise 3D-convolution followed
            by a 3D-upsampling. The output of these 2 layers is summed with
            the corresponding channel-depth output from the encoder and fed
            into a convolutional block.

            Args:
                filters: int
                    The number of filters to use in the 3D convolutional
                    block. The output layer of this green block will have
                    this many number of channels.
                input_shape: (int, int, int, int)
                    The input shape required by tf.keras.layers.Conv3D being the first
                    layer in this custom layer.
                kernel_size: kernel_size: (int, int, int), optional
                    The size of all convolutional kernels. Defaults to (3, 3, 3),
                    as used in the paper.
                data_format: str, optional
                    The format of the input data. Must be either 'channels_last'
                    or 'channels_first'. Defaults to `channels_last` for CPU
                    development. 'channels_first is used in the paper.
                groups: int, optional
                    The size of each group for GroupNormalization. Defaults to
                    8, as used in the paper.
        """
        super(DecoderBlock, self).__init__()

        self.conv3d_ptwise = tf.keras.layers.Conv3D(
                                filters=filters,
                                input_shape=input_shape,
                                kernel_size=(1, 1, 1),
                                strides=1,
                                padding='same',
                                data_format=data_format)
        self.upsample = tf.keras.layers.UpSampling3D(
                                size=2,
                                data_format=data_format)
        new_shape = (input_shape[0] * 2, 
                     input_shape[1] * 2,
                     input_shape[2] * 2,
                     filters)
        self.residual = tf.keras.layers.Add()
        self.conv_block = ConvBlock(
                                filters=filters,
                                input_shape=new_shape,
                                kernel_size=kernel_size,
                                data_format=data_format,
                                groups=groups)

    def call(self, x, enc_res):
        """Returns the forward pass of one DecoderBlock.

            { Conv3D_ptwise -> Upsample3D -> Residual -> ConvBlock }
        """
        x = self.conv3d_ptwise(x)
        x = self.upsample(x)
        x = self.residual([x, enc_res])
        x = self.conv_block(x)
        return x


class ConvDecoder(tf.keras.layers.Layer):
    def __init__(self,
                 data_format='channels_last',
                 kernel_size=(3, 3, 3),
                 groups=8):
        """Initializes the model decoder.

            See https://arxiv.org/pdf/1810.11654.pdf for more details.
            The model decoder takes the encoder outputs at each level
            of downsampling and upsamples, connecting each recovered
            level with its corresponding encoder level residually. The
            final logit layer outputs a 3-channel Tensor the same size
            as the original input to the encoder.

            Args:
                data_format: str, optional
                    The format of the input data. Must be either 'channels_last'
                    or 'channels_first'. Defaults to `channels_last` for CPU
                    development. 'channels_first is used in the paper.
                kernel_size: kernel_size: (int, int, int), optional
                    The size of all convolutional kernels. Defaults to (3, 3, 3),
                    as used in the paper.
                groups: int, optional
                    The size of each group for GroupNormalization. Defaults to
                    8, as used in the paper.
        """
        super(ConvDecoder, self).__init__()
        self.dec_block_128 = DecoderBlock(
                                filters=128,
                                input_shape=(20, 24, 16, 256),
                                kernel_size=kernel_size,
                                data_format=data_format,
                                groups=groups)
        self.dec_block_64 = DecoderBlock(
                                filters=64,
                                input_shape=(40, 48, 32, 128),
                                kernel_size=kernel_size,
                                data_format=data_format,
                                groups=groups)
        self.dec_block_32 = DecoderBlock(
                                filters=32,
                                input_shape=(80, 96, 64, 64),
                                kernel_size=kernel_size,
                                data_format=data_format,
                                groups=groups)

        self.out_conv = tf.keras.layers.Conv3D(
                                filters=32,
                                kernel_size=kernel_size,
                                strides=1,
                                padding='same',
                                data_format=data_format)

        self.ptwise_logits = tf.keras.layers.Conv3D(
                                filters=3,
                                kernel_size=(1, 1, 1),
                                strides=1,
                                padding='same',
                                data_format=data_format,
                                activation='sigmoid')

    def call(self, enc_outs):
        """Returns the forward pass of the ConvDecoder.

            {
                DecoderBlock_128 -> DecoderBlock_64 -> DecoderBlock_32,
                OutputConv_32 -> LogitConv_3 + Sigmoid
            }

            Args:
                enc_outs: (Tensor, Tensor, Tensor, Tensor)
                    Contains residual outputs of the encoder from forward
                    pass. Must contain (in order) the ConvBlock outputs at
                    the 32, 64, 128, and 256 filter sizes.

            Returns:
                x: Tensor
                    Output 'image' the same size as the original input image
                    to the encoder, but with 3 channels.
        """
        enc_out_32, enc_out_64, enc_out_128, x = enc_outs

        x = self.dec_block_128(x, enc_out_128)
        x = self.dec_block_64(x, enc_out_64)
        x = self.dec_block_32(x, enc_out_32)

        x = self.out_conv(x)
        x = self.ptwise_logits(x)

        return x
