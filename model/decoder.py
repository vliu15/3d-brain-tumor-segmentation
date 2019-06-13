"""Contains custom convolutional decoder class."""
import tensorflow as tf

from model.resnet_block import ConvBlock
from model.layer_utils.getters import get_upsampling
from utils.constants import *


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size=3,
                 data_format='channels_last',
                 groups=8,
                 reduction=2,
                 kernel_regularizer=tf.keras.regularizers.l2(l=1e-5),
                 kernel_initializer='he_normal',
                 upsampling='linear',
                 normalization='group'):
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
                kernel_size: int, optional
                    The size of all convolutional kernels. Defaults to 3,
                    as used in the paper.
                data_format: str, optional
                    The format of the input data. Must be either 'channels_last'
                    or 'channels_first'. Defaults to `channels_last` for CPU
                    development. 'channels_first is used in the paper.
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
        super(DecoderBlock, self).__init__()
        # Set up config for self.get_config() to serialize later.
        self.config = super(DecoderBlock, self).get_config()
        self.config.update({'filters': filters,
                            'kernel_size': kernel_size,
                            'data_format': data_format,
                            'groups': groups,
                            'reduction': reduction,
                            'kernel_regularizer': tf.keras.regularizers.serialize(kernel_regularizer),
                            'kernel_initializer': kernel_initializer,
                            'upsampling': upsampling,
                            'normalization': normalization})

        # Retrieve upsampling method.
        Upsample = get_upsampling(upsampling)

        self.conv3d_ptwise = tf.keras.layers.Conv3D(
                                filters=filters,
                                kernel_size=1,
                                strides=1,
                                padding='same',
                                data_format=data_format,
                                kernel_regularizer=kernel_regularizer,
                                kernel_initializer=kernel_initializer)
        self.upsample = Upsample(
                                filters=filters,
                                kernel_size=kernel_size,
                                groups=groups,
                                data_format=data_format,
                                kernel_regularizer=kernel_regularizer,
                                kernel_initializer=kernel_initializer,
                                normalization=normalization)
        self.residual = tf.keras.layers.Add()
        self.conv_block = ConvBlock(
                                filters=filters,
                                kernel_size=kernel_size,
                                groups=groups,
                                reduction=reduction,
                                data_format=data_format,
                                kernel_regularizer=kernel_regularizer,
                                kernel_initializer=kernel_initializer,
                                normalization=normalization)

    def call(self, inputs, training=None):
        inputs, enc_res = inputs
        inputs = self.conv3d_ptwise(inputs)
        inputs = self.upsample(inputs, training=training)
        inputs = self.residual([inputs, enc_res])
        inputs = self.conv_block(inputs, training=training)
        return inputs

    def get_config(self):
        return self.config


class ConvDecoder(tf.keras.layers.Layer):
    def __init__(self,
                 data_format='channels_last',
                 kernel_size=3,
                 groups=8,
                 reduction=2,
                 kernel_regularizer=tf.keras.regularizers.l2(l=1e-5),
                 kernel_initializer='he_normal',
                 upsampling='linear',
                 normalization='group'):
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
        super(ConvDecoder, self).__init__()
        # Set up config for self.get_config() to serialize later.
        self.config = super(ConvDecoder, self).get_config()
        self.config.update({'data_format': data_format,
                            'kernel_size': kernel_size,
                            'groups': groups,
                            'reduction': reduction,
                            'kernel_regularizer': tf.keras.regularizers.serialize(kernel_regularizer),
                            'kernel_initializer': kernel_initializer,
                            'upsampling': upsampling,
                            'normalization': normalization})

        self.dec_block_2 = DecoderBlock(
                                filters=DEC_CONV_BLOCK2_SIZE,
                                kernel_size=kernel_size,
                                groups=groups,
                                reduction=reduction,
                                data_format=data_format,
                                kernel_regularizer=kernel_regularizer,
                                kernel_initializer=kernel_initializer,
                                upsampling=upsampling,
                                normalization=normalization)
        self.dec_block_1 = DecoderBlock(
                                filters=DEC_CONV_BLOCK1_SIZE,
                                kernel_size=kernel_size,
                                groups=groups,
                                reduction=reduction,
                                data_format=data_format,
                                kernel_regularizer=kernel_regularizer,
                                kernel_initializer=kernel_initializer,
                                upsampling=upsampling,
                                normalization=normalization)
        self.dec_block_0 = DecoderBlock(
                                filters=DEC_CONV_BLOCK0_SIZE,
                                kernel_size=kernel_size,
                                groups=groups,
                                reduction=reduction,
                                data_format=data_format,
                                kernel_regularizer=kernel_regularizer,
                                kernel_initializer=kernel_initializer,
                                upsampling=upsampling,
                                normalization=normalization)

        self.ptwise_logits = tf.keras.layers.Conv3D(
                                filters=OUT_CH,
                                kernel_size=1,
                                strides=1,
                                padding='same',
                                activation='sigmoid',
                                data_format=data_format,
                                kernel_regularizer=kernel_regularizer,
                                kernel_initializer=kernel_initializer)

    def call(self, inputs, training=None):
        enc_out_0, enc_out_1, enc_out_2, inputs = inputs

        inputs = self.dec_block_2((inputs, enc_out_2), training=training)
        inputs = self.dec_block_1((inputs, enc_out_1), training=training)
        inputs = self.dec_block_0((inputs, enc_out_0), training=training)

        inputs = self.ptwise_logits(inputs)

        return inputs

    def get_config(self):
        return self.config
