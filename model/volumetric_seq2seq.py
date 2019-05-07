import tensorflow as tf

from model.encoder import ConvEncoder
from model.decoder import ConvDecoder
from model.variational_autoencoder import VariationalAutoEncoder

class VolumetricSeq2Seq(tf.keras.Model):
    def __init__(self,
                 data_format='channels_last',
                 kernel_size=3,
                 groups=8,
                 dropout=0.2):
        super(VolumetricSeq2Seq, self).__init__()

        self.encoder = ConvEncoder(
                                data_format=data_format,
                                kernel_size=kernel_size,
                                groups=groups,
                                dropout=dropout)
        self.decoder = ConvDecoder(
                                data_format=data_format,
                                kernel_size=kernel_size,
                                groups=groups)
        self.vae = VariationalAutoEncoder()

    def call(self, x):
        enc_outs = self.encoder(x)
        y_hat = self.decoder(enc_outs)
        y_vae = self.vae(enc_outs[-1])

        return y_hat, y_vae
