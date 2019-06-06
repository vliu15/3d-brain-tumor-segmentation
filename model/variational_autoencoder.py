"""Contains custom variational autoencoder class."""
import tensorflow as tf

from model.layer_utils.group_norm import GroupNormalization
from model.layer_utils.getters import get_downsampling, get_upsampling
from model.resnet_block import ConvBlock, ConvLayer
from utils.constants import *


def sample(latent_args):
    """Samples from the Gaussian given by mean and variance."""
    z_mean, z_logvar = latent_args
    eps = tf.random.normal(shape=z_mean.shape, dtype=tf.float32)
    return z_mean + tf.math.exp(0.5 * z_logvar) * eps


class VariationalAutoencoderBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size=3,
                 reduction=2,
                 data_format='channels_last',
                 groups=8,
                 kernel_regularizer=tf.keras.regularizers.l2(l=1e-5),
                 kernel_initializer='he_normal',
                 use_se=False,
                 upsampling='linear',
                 normalization='group'):
        """Initializes a variational autoencoder block.

            See https://arxiv.org/pdf/1810.11654.pdf for more details.
            Similar to a decoder block, a variational autoencoder block
            consists of pointwise convolution, upsampling, and a conv
            block. There is no residual passed through from the encoder
            in this branch of the network.

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
                kernel_regularizer: regularizer callable for convolutional
                    kernels.
        """
        super(VariationalAutoencoderBlock, self).__init__()
        # Set up config for self.get_config() to serialize later.
        self.config = super(VariationalAutoencoderBlock, self).get_config()
        self.config.update({'filters': filters,
                            'kernel_size': kernel_size,
                            'data_format': data_format,
                            'groups': groups,
                            'reduction': reduction,
                            'kernel_regularizer': tf.keras.regularizers.serialize(kernel_regularizer),
                            'kernel_initializer': kernel_initializer,
                            'use_se': use_se,
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
        self.conv_block = ConvBlock(
                                filters=filters,
                                kernel_size=kernel_size,
                                groups=groups,
                                reduction=reduction,
                                data_format=data_format,
                                kernel_regularizer=kernel_regularizer,
                                kernel_initializer=kernel_initializer,
                                use_se=use_se,
                                normalization=normalization)

    def call(self, inputs, training=None):
        """Returns the forward pass of one VariationalAutoencoderBlock.

            { Conv3D_ptwise -> Upsample3D -> ConvBlock }
        """
        inputs = self.conv3d_ptwise(inputs)
        inputs = self.upsample(inputs, training=training)
        inputs = self.conv_block(inputs, training=training)
        return inputs

    def get_config(self):
        return self.config


class VariationalAutoencoder(tf.keras.layers.Layer):
    def __init__(self,
                 data_format='channels_last',
                 groups=8,
                 reduction=2,
                 kernel_size=3,
                 kernel_regularizer=tf.keras.regularizers.l2(l=1e-5),
                 kernel_initializer='he_normal',
                 use_se=False,
                 downsampling='max',
                 upsampling='linear',
                 normalization='group'):
        """Initializes the Variational Autoencoder branch.
        
            See https://arxiv.org/pdf/1810.11654.pdf for more details.
            The variational autoencoder reconstructs the original image
            given to the encoder to regularize the encoder. To do so,
            it samples from a learned Gaussian and upsamples.

            Args:
                data_format: str, optional
                    The format of the input data. Must be either 'channels_last'
                    or 'channels_first'. Defaults to `channels_last` for CPU
                    development. 'channels_first is used in the paper.
                groups: int, optional
                    The size of each group for GroupNormalization. Defaults to
                    8, as used in the paper.
                kernel_size: kernel_size: int, optional
                    The size of all convolutional kernels. Defaults to 3,
                    as used in the paper.
                kernel_regularizer: tf.keras.regularizers callable, optional
                    Kernel regularization for convolutional and dense operations.
                kernel_initializer: tf.keras.initializers callable, optional
                    Kernel initializer for convolutional operations.
        """
        super(VariationalAutoencoder, self).__init__()
        # Set up config for self.get_config() to serialize later.
        self.config = super(VariationalAutoencoder, self).get_config()
        self.config.update({'data_format': data_format,
                            'groups': groups,
                            'reduction': reduction,
                            'kernel_size': kernel_size,
                            'kernel_regularizer': tf.keras.regularizers.serialize(kernel_regularizer),
                            'kernel_initializer': kernel_initializer,
                            'use_se': use_se,
                            'downsampling': downsampling,
                            'upsampling': upsampling,
                            'normalization': normalization})

        # Retrieve downsampling method for VD block.
        Downsample = get_downsampling(downsampling)

        # Retrieve upsampling method for VU block.
        Upsample = get_upsampling(upsampling)

        # VD Block
        self.downsample_VD = Downsample(
                                filters=VAE_VD_CONV_SIZE,
                                kernel_size=kernel_size,
                                groups=groups,
                                data_format=data_format,
                                kernel_regularizer=kernel_regularizer,
                                kernel_initializer=kernel_initializer,
                                normalization=normalization)
        self.flatten_VD = tf.keras.layers.Flatten(data_format)
        self.proj_VD = tf.keras.layers.Dense(
                                units=VAE_VD_BLOCK_SIZE,
                                kernel_regularizer=kernel_regularizer,
                                kernel_initializer=kernel_initializer)

        # VDraw Block
        self.sample = tf.keras.layers.Lambda(sample)

        # VU Block
        h = int(H/16)
        w = int(W/16)
        d = int(D/16)
        self.proj_VU = tf.keras.layers.Dense(
                                    units=h * w * d * 1,
                                    kernel_regularizer=kernel_regularizer,
                                    kernel_initializer=kernel_initializer)
        self.relu_VU = tf.keras.layers.Activation('relu')
        self.reshape_VU = tf.keras.layers.Reshape((h, w, d, 1) if data_format == 'channels_last' else (1, h, w, d))
        self.conv1d_VU = tf.keras.layers.Conv3D(
                                    filters=VAE_VU_BLOCK_SIZE,
                                    kernel_size=1,
                                    strides=1,
                                    padding='same',
                                    data_format=data_format,
                                    kernel_regularizer=kernel_regularizer,
                                    kernel_initializer=kernel_initializer)
        self.upsample_VU = Upsample(
                                    filters=VAE_VU_BLOCK_SIZE,
                                    kernel_size=kernel_size,
                                    groups=groups,
                                    data_format=data_format,
                                    kernel_regularizer=kernel_regularizer,
                                    kernel_initializer=kernel_initializer,
                                    normalization=normalization)

        self.conv_block_2 = VariationalAutoencoderBlock(
                                    filters=VAE_CONV_BLOCK2_SIZE,
                                    kernel_size=kernel_size,
                                    groups=groups,
                                    reduction=reduction,
                                    data_format=data_format,
                                    kernel_regularizer=kernel_regularizer,
                                    kernel_initializer=kernel_initializer,
                                    upsampling=upsampling,
                                    use_se=use_se,
                                    normalization=normalization)

        self.conv_block_1 = VariationalAutoencoderBlock(
                                    filters=VAE_CONV_BLOCK1_SIZE,
                                    kernel_size=kernel_size,
                                    groups=groups,
                                    reduction=reduction,
                                    data_format=data_format,
                                    kernel_regularizer=kernel_regularizer,
                                    kernel_initializer=kernel_initializer,
                                    upsampling=upsampling,
                                    use_se=use_se,
                                    normalization=normalization)

        self.conv_block_0 = VariationalAutoencoderBlock(
                                    filters=VAE_CONV_BLOCK0_SIZE,
                                    kernel_size=kernel_size,
                                    groups=groups,
                                    reduction=reduction,
                                    data_format=data_format,
                                    kernel_regularizer=kernel_regularizer,
                                    kernel_initializer=kernel_initializer,
                                    upsampling=upsampling,
                                    use_se=use_se,
                                    normalization=normalization)

        self.conv_out = tf.keras.layers.Conv3D(
                                    filters=IN_CH,
                                    kernel_size=kernel_size,
                                    strides=1,
                                    padding='same',
                                    data_format=data_format,
                                    kernel_regularizer=kernel_regularizer,
                                    kernel_initializer=kernel_initializer)

    def call(self, inputs, training=None):
        """Returns the forward pass of the VariationalAutoencoder.

            {
                VD:        Reduce dimensionality and flatten.
                VDraw:     Sample mean and variance.
                VU:        Reconstruct volumetric input.
                VUp+Block: Upsample and convolve.
                Vend:      Final output convolution.
            }
        """
        # VD Block
        inputs = self.downsample_VD(inputs, training=training)
        inputs = self.flatten_VD(inputs)
        inputs = self.proj_VD(inputs)

        # VDraw Block
        z_mean = inputs[:, :VAE_LATENT_SIZE]
        z_logvar = inputs[:, VAE_LATENT_SIZE:]
        inputs = self.sample([z_mean, z_logvar])

        # VU Block
        inputs = self.proj_VU(inputs)
        inputs = self.relu_VU(inputs)
        inputs = self.reshape_VU(inputs)
        inputs = self.conv1d_VU(inputs)
        inputs = self.upsample_VU(inputs, training=training)

        # VUp2 and VBlock2
        inputs = self.conv_block_2(inputs, training=training)

        # VUp1 and VBlock1
        inputs = self.conv_block_1(inputs, training=training)

        # VUp0 and VBlock0
        inputs = self.conv_block_0(inputs, training=training)

        # Vend
        inputs = self.conv_out(inputs)

        return inputs, z_mean, z_logvar

    def get_config(self):
        return self.config
