import tensorflow as tf
from model.group_norm import GroupNormalization
from model.layers import ConvBlock


def sample(latent_args):
    """Samples from the Gaussian given by mean and variance."""
    mean, var = latent_args
    eps = tf.random.normal(shape=mean.shape, dtype=tf.float32)
    return mean + tf.math.exp(0.5 * var) * eps


class VariationalAutoencoderBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size=3,
                 data_format='channels_last',
                 groups=8,
                 kernel_regularizer=None):
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
                kernel_size: kernel_size: (int, int, int), optional
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

        self.conv3d_ptwise = tf.keras.layers.Conv3D(
                                filters=filters,
                                kernel_size=1,
                                strides=1,
                                padding='same',
                                data_format=data_format,
                                kernel_regularizer=kernel_regularizer)
        self.upsample = tf.keras.layers.UpSampling3D(
                                size=2,
                                data_format=data_format)
        self.conv_block = ConvBlock(
                                filters=filters,
                                kernel_size=kernel_size,
                                data_format=data_format,
                                groups=groups,
                                kernel_regularizer=kernel_regularizer)

    def call(self, x):
        """Returns the forward pass of one VariationalAutoencoderBlock.

            { Conv3D_ptwise -> Upsample3D -> ConvBlock }
        """
        x = self.conv3d_ptwise(x)
        x = self.upsample(x)
        x = self.conv_block(x)
        return x


class VariationalAutoencoder(tf.keras.layers.Layer):
    def __init__(self,
                 data_format='channels_last',
                 groups=8,
                 kernel_size=3,
                 kernel_regularizer=None):
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
                kernel_size: kernel_size: (int, int, int), optional
                    The size of all convolutional kernels. Defaults to 3,
                    as used in the paper.
                kernel_regularizer: regularizer callable for convolutional
                    kernels.
        """
        super(VariationalAutoencoder, self).__init__(name='variational_autoencoder')

        # VD Block
        self.groupnorm_256 = GroupNormalization(
                                groups=groups,
                                axis=-1 if data_format == 'channels_last' else 1)
        self.relu_256 = tf.keras.layers.Activation('relu')
        self.conv3d_256 = tf.keras.layers.Conv3D(
                                filters=16,
                                kernel_size=kernel_size,
                                strides=2,
                                padding='same',
                                data_format=data_format,
                                kernel_regularizer=kernel_regularizer)
        self.flatten_256 = tf.keras.layers.Flatten(data_format)
        self.proj_256 = tf.keras.layers.Dense(256)

        # VDraw Block
        self.mean = tf.keras.layers.Dense(128)
        self.var = tf.keras.layers.Dense(128)
        self.sample = tf.keras.layers.Lambda(sample)

        # VU Block
        self.proj_VU = tf.keras.layers.Dense(10*12*8*1)
        self.relu_VU = tf.keras.layers.Activation('relu')
        self.reshape = tf.keras.layers.Reshape((10, 12, 8, 1))
        self.conv1d_VU = tf.keras.layers.Conv3D(
                                filters=256,
                                kernel_size=1,
                                strides=1,
                                padding='same',
                                data_format=data_format,
                                kernel_regularizer=kernel_regularizer)
        self.conv3d_upsample_VU = tf.keras.layers.UpSampling3D(
                                size=2,
                                data_format=data_format)

        self.conv_block_128 = VariationalAutoencoderBlock(
                                filters=128,
                                data_format=data_format,
                                groups=groups,
                                kernel_size=kernel_size,
                                kernel_regularizer=kernel_regularizer)

        self.conv_block_64 = VariationalAutoencoderBlock(
                                filters=64,
                                data_format=data_format,
                                groups=groups,
                                kernel_size=kernel_size,
                                kernel_regularizer=kernel_regularizer)

        self.conv_block_32 = VariationalAutoencoderBlock(
                                filters=32,
                                data_format=data_format,
                                groups=groups,
                                kernel_size=kernel_size,
                                kernel_regularizer=kernel_regularizer)

        self.conv_4 = tf.keras.layers.Conv3D(
                                filters=4,
                                kernel_size=kernel_size,
                                strides=1,
                                padding='same',
                                data_format=data_format,
                                kernel_regularizer=kernel_regularizer)

    def call(self, x):
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
        x = self.groupnorm_256(x)
        x = self.relu_256(x)
        x = self.conv3d_256(x)
        x = self.flatten_256(x)
        x = self.proj_256(x)

        # VDraw Block
        mu = self.mean(x)
        z_mean = mu
        sigma = self.var(x)
        z_var = sigma
        x = self.sample([mu, sigma])

        # VU Block
        x = self.proj_VU(x)
        x = self.relu_VU(x)
        x = self.reshape(x)
        x = self.conv1d_VU(x)
        x = self.conv3d_upsample_VU(x)

        # VUp2 and VBlock2
        x = self.conv_block_128(x)

        # VUp1 and VBlock1
        x = self.conv_block_64(x)

        # VUp0 and VBlock0
        x = self.conv_block_32(x)

        # Vend
        x = self.conv_4(x)

        y_vae = x

        return y_vae, z_mean, z_var
