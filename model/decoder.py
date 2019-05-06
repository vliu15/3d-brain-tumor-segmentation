import tensorflow as tf

from model.layers import ConvBlock


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 input_shape,
                 data_format='channels_last',
                 kernel_size=(3, 3, 3),
                 groups=8):
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
        enc_out_32, enc_out_64, enc_out_128, x = enc_outs

        x = self.dec_block_128(x, enc_out_128)
        x = self.dec_block_64(x, enc_out_64)
        x = self.dec_block_32(x, enc_out_32)

        x = self.out_conv(x)
        x = self.ptwise_logits(x)

        return x
