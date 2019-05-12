import tensorflow as tf

from model.encoder import ConvEncoder
from model.decoder import ConvDecoder
from model.variational_autoencoder import VariationalAutoEncoder

class VolumetricCNN(tf.keras.Model):
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
        self.vae = VariationalAutoEncoder(
                                data_format=data_format,
                                kernel_size=kernel_size,
                                groups=groups,
                                kernel_regularizer=kernel_regularizer)

    def call(self, x):
        enc_outs = self.encoder(x)
        y_pred = self.decoder(enc_outs)
        y_vae, z_mean, z_var = self.vae(enc_outs[-1])

        return y_pred, y_vae, z_mean, z_var
