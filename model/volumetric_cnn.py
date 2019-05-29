import tensorflow as tf

from model.encoder import ConvEncoder
from model.decoder import ConvDecoder
from model.variational_autoencoder import VariationalAutoencoder


class VolumetricCNN(tf.keras.models.Model):
    def __init__(self,
                 data_format='channels_last',
                 kernel_size=3,
                 groups=8,
                 reduction=4,
                 kernel_regularizer=None,
                 kernel_initializer=tf.initializers.he_normal,
                 use_se=False):
        """Initializes the VolumetricCNN model.
        
            Modified model with SENet blocks instead of ResNet blocks:
            see https://arxiv.org/pdf/1810.11654.pdf for more details.

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
                use_se: bool, optional
                    Whether to apply squeeze-excitation layer to residual blocks.
        """
        super(VolumetricCNN, self).__init__()
        self.epoch = tf.Variable(0, name='epoch', trainable=False)
        self.encoder = ConvEncoder(
                                data_format=data_format,
                                kernel_size=kernel_size,
                                groups=groups,
                                reduction=reduction,
                                kernel_regularizer=kernel_regularizer,
                                kernel_initializer=kernel_initializer,
                                use_se=use_se)
        self.decoder = ConvDecoder(
                                data_format=data_format,
                                kernel_size=kernel_size,
                                groups=groups,
                                reduction=reduction,
                                kernel_regularizer=kernel_regularizer,
                                kernel_initializer=kernel_initializer,
                                use_se=use_se)
        self.vae = VariationalAutoencoder(
                                data_format=data_format,
                                kernel_size=kernel_size,
                                groups=groups,
                                kernel_regularizer=kernel_regularizer,
                                kernel_initializer=kernel_initializer,
                                use_se=use_se)

    def call(self, inputs, training=False):
        """Returns the forward pass of the VolumetricCNN model.
        
            { Encoder -> [Decoder + Residuals, VAE] }
        """
        enc_outs = self.encoder(inputs)
        y_pred = self.decoder(enc_outs)
        y_vae, z_mean, z_logvar = self.vae(enc_outs[-1])

        return y_pred, y_vae, z_mean, z_logvar


class EncDecCNN(tf.keras.models.Model):
    def __init__(self,
                 data_format='channels_last',
                 kernel_size=3,
                 groups=8,
                 reduction=4,
                 kernel_regularizer=None,
                 kernel_initializer=tf.initializers.he_normal,
                 use_se=False):
        """Initializes the EncDecCNN model.
        
            This is the VolumetricCNN model without the variational
            autoencoder branch for regularization.

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
                use_se: bool, optional
                    Whether to apply squeeze-excitation layer to residual blocks.
        """
        super(EncDecCNN, self).__init__()
        self.epoch = tf.Variable(0, name='epoch', trainable=False)
        self.encoder = ConvEncoder(
                                data_format=data_format,
                                kernel_size=kernel_size,
                                groups=groups,
                                reduction=reduction,
                                kernel_regularizer=kernel_regularizer,
                                kernel_initializer = kernel_initializer,
                                use_se=use_se)
        self.decoder = ConvDecoder(
                                data_format=data_format,
                                kernel_size=kernel_size,
                                groups=groups,
                                reduction=reduction,
                                kernel_regularizer=kernel_regularizer,
                                kernel_initializer=kernel_initializer,
                                use_se=use_se)

    def call(self, inputs, training=False):
        """Returns the forward pass of the EncDecCNN model.
        
            { Encoder -> Decoder + Residuals }
        """
        inputs = self.encoder(inputs)
        inputs = self.decoder(inputs)

        return inputs
