"""Contains custom models for 3D semantic segmentation."""
import tensorflow as tf

from model.encoder import Encoder
from model.decoder import Decoder
from model.variational_autoencoder import VariationalAutoencoder


class VolumetricCNN(tf.keras.models.Model):
    def __init__(self,
                 data_format='channels_last',
                 groups=8,
                 reduction=2,
                 l2_scale=1e-5,
                 downsampling='conv',
                 upsampling='conv',
                 base_filters=16,
                 depth=4):
        """ Initializes the VolumetricCNN model, a cross between the 3D U-net
            and 2018 BraTS Challenge top model with VAE regularization.

            References:
                - [3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://arxiv.org/pdf/1606.06650.pdf)
                - [3D MRI brain tumor segmentation using autoencoder regularization](https://arxiv.org/pdf/1810.11654.pdf)
        """
        super(VolumetricCNN, self).__init__()
        self.epoch = tf.Variable(0, name='epoch', trainable=False)
        self.encoder = Encoder(
                            data_format=data_format,
                            groups=groups,
                            reduction=reduction,
                            l2_scale=l2_scale,
                            downsampling=downsampling,
                            base_filters=base_filters,
                            depth=depth)
        self.decoder = Decoder(
                            data_format=data_format,
                            groups=groups,
                            reduction=reduction,
                            l2_scale=l2_scale,
                            upsampling=upsampling,
                            base_filters=base_filters,
                            depth=depth)
        self.vae = VariationalAutoencoder(
                            data_format=data_format,
                            groups=groups,
                            reduction=reduction,
                            l2_scale=l2_scale,
                            upsampling=upsampling,
                            base_filters=base_filters,
                            depth=depth)

    def call(self, inputs, training=None, inference=None):
        # Inference mode does not evaluate VAE branch.
        assert (not inference or not training), \
            'Cannot run training and inference modes simultaneously.'

        inputs = self.encoder(inputs, training=training)
        y_pred = self.decoder((inputs[-1], inputs[:-1]), training=training)

        if inference:
            return (y_pred, None, None, None)
        y_vae, z_mean, z_logvar = self.vae(inputs[-1], training=training)

        return (y_pred, y_vae, z_mean, z_logvar)
