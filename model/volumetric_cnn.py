import tensorflow as tf

from model.encoder import ConvEncoder
from model.decoder import ConvDecoder
from model.variational_autoencoder import VariationalAutoencoder

class VolumetricCNN(tf.keras.models.Model):
    def __init__(self,
                 data_format='channels_last',
                 kernel_size=3,
                 groups=8,
                 dropout=0.2,
                 kernel_regularizer=None):
        super(VolumetricCNN, self).__init__()

        self.encoder = ConvEncoder(
                                data_format=data_format,
                                kernel_size=kernel_size,
                                groups=groups,
                                dropout=dropout,
                                kernel_regularizer=kernel_regularizer)
        self.decoder = ConvDecoder(
                                data_format=data_format,
                                kernel_size=kernel_size,
                                groups=groups,
                                kernel_regularizer=kernel_regularizer)
        self.vae = VariationalAutoencoder(
                                data_format=data_format,
                                kernel_size=kernel_size,
                                groups=groups,
                                kernel_regularizer=kernel_regularizer)

    def call(self, x, training=False):
        enc_outs = self.encoder(x, training=training)
        y_pred = self.decoder(enc_outs)
        y_vae, z_mean, z_logvar = self.vae(enc_outs[-1])

        return y_pred, y_vae, z_mean, z_logvar
