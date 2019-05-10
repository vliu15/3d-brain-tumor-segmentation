import tensorflow as tf
from model.group_norm import GroupNormalization
from model.layers import ConvBlock

def sample(latent_args):
    mean, var = latent_args
    print(mean.shape)
    print(var.shape)
    batch_size = tf.keras.backend.shape(mean)[0]
    dim = tf.keras.backend.shape(mean)[1]
    eps = tf.keras.backend.random_normal(shape=(batch_size, dim), dtype='double')
    print(eps.shape)
    return mean + tf.keras.backend.exp(0.5*var) * eps

class VariationalAutoEncoderBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size=3,
                 data_format='channels_last',
                 groups=8,
                 kernel_regularizer=None):
        super(VariationalAutoEncoderBlock, self).__init__()

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
        """Returns the forward pass of one DecoderBlock.

            { Conv3D_ptwise -> Upsample3D -> Residual -> ConvBlock }
        """
        x = self.conv3d_ptwise(x)
        x = self.upsample(x)
        x = self.conv_block(x)
        return x

class VariationalAutoEncoder(tf.keras.layers.Layer):
    def __init__(self,
                 data_format='channels_last',
                 groups=8,
                 kernel_size=3,
                 kernel_regularizer=None):
        super(VariationalAutoEncoder, self).__init__()

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
            kernel_regularizer=kernel_regularizer
        )
        self.conv3d_upsample_VU = tf.keras.layers.UpSampling3D(
            size=2,
            data_format=data_format,
        )

        self.conv_block_128 = VariationalAutoEncoderBlock(
            filters=128,
            data_format=data_format,
            groups=groups,
            kernel_size=kernel_size,
            kernel_regularizer=kernel_regularizer,
        )

        self.conv_block_64 = VariationalAutoEncoderBlock(
            filters=64,
            data_format=data_format,
            groups=groups,
            kernel_size=kernel_size,
            kernel_regularizer=kernel_regularizer,
        )

        self.conv_block_32 = VariationalAutoEncoderBlock(
            filters=32,
            data_format=data_format,
            groups=groups,
            kernel_size=kernel_size,
            kernel_regularizer=kernel_regularizer,
        )

        self.conv_4 = tf.keras.layers.Conv3D(
            filters=4,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            data_format=data_format,
            kernel_regularizer=kernel_regularizer,
        )

    def call(self, x):

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
