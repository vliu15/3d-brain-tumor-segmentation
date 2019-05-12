import argparse
import tensorflow as tf
import numpy as np

from model.encoder import ConvEncoder
from model.decoder import ConvDecoder
from model.variational_autoencoder import VariationalAutoencoder
from model.volumetric_cnn import VolumetricCNN
from utils.loss import compute_myrnenko_loss, get_myrnenko_loss_fn
from utils.optimizer import ScheduledAdam


def test_encoder(x):
    """Tests that the encoder output shapes are correct."""
    encoder = ConvEncoder(
                    data_format='channels_last',
                    kernel_size=3,
                    groups=8,
                    dropout=0.2,
                    kernel_regularizer=tf.keras.regularizers.l2(l=1e-5))
    
    conv_out_32, conv_out_64, conv_out_128, encoder_out = encoder(x)
    assert conv_out_32.shape == (1, 160, 192, 128, 32)
    assert conv_out_64.shape == (1, 80, 96, 64, 64)
    assert conv_out_128.shape == (1, 40, 48, 32, 128)
    assert encoder_out.shape == (1, 20, 24, 16, 256)

    print('All encoder output shapes match.')

    params = sum(tf.reduce_prod(var.shape) for var in encoder.trainable_variables)
    print(f'Number of trainable parameters: {params}.')

    return (conv_out_32, conv_out_64, conv_out_128, encoder_out)


def test_decoder(enc_outs):
    """Tests that the decoder output shapes are correct."""
    decoder = ConvDecoder(
                    data_format='channels_last',
                    kernel_size=3,
                    groups=8,
                    kernel_regularizer=tf.keras.regularizers.l2(l=1e-5))

    logits = decoder(enc_outs)
    assert logits.shape == (1, 160, 192, 128, 3)

    print('Decoder output shape matches.')

    params = sum(tf.reduce_prod(var.shape) for var in decoder.trainable_variables)
    print(f'Number of trainable parameters: {params}.')

    return logits


def test_vae(enc_256):
    """Tests that the encoder output shapes are correct."""
    vae = VariationalAutoencoder(
        data_format='channels_last',
        kernel_size=3,
        groups=8,
        kernel_regularizer=tf.keras.regularizers.l2(l=1e-5))

    y_vae, z_mean, z_var = vae(enc_256)
    assert(z_mean.shape == (1, 128))
    assert(z_var.shape == (1, 128))
    assert(y_vae.shape == (1, 160, 192, 128, 4))
    print('All vae output shapes match.')

    params = sum(tf.reduce_prod(var.shape) for var in vae.trainable_variables)
    print(f'Number of trainable parameters: {params}.')

    return (y_vae, z_mean, z_var)


def test_optimizer():
    """Tests that the custom optimizer schedules correctly."""
    optimizer = ScheduledAdam(learning_rate=1e-4, n_epochs=300)

    assert optimizer.learning_rate.numpy() == np.array(1e-4, dtype=np.float32)

    optimizer.update_lr(100)
    assert optimizer.learning_rate.numpy() == np.array(1e-4 * ((1.0 - 100.0 / 300) ** 0.9), dtype=np.float32)
    
    optimizer.update_lr(300)
    assert optimizer.learning_rate.numpy() == np.array(0, dtype=np.float32)

    optimizer.update_lr(0)
    assert optimizer.learning_rate.numpy() == np.array(1e-4, dtype=np.float32)

    print('Passed basic sanity checks.')


def test_loss(x, y_true, y_pred, y_vae, z_mean, z_var, eps=1e-8):
    functional_loss = compute_myrnenko_loss(x, y_true, y_pred, y_vae, z_mean, z_var, eps=eps)
    keras_loss = get_myrnenko_loss_fn(x, y_vae, z_mean, z_var)(y_true, y_pred)

    assert functional_loss.shape == ()
    assert keras_loss.shape == ()

    print(f'Functional loss: {functional_loss}.')
    print(f'Keras-style loss: {keras_loss}.')


def test_cnn(x, y_true):
    cnn = VolumetricCNN(
                data_format='channels_last',
                kernel_size=3,
                groups=8,
                dropout=0.2,
                kernel_regularizer=tf.keras.regularizers.l2(l=1e-5))

    y_pred, y_vae, z_mean, z_var = cnn(x)

    assert y_vae.shape == x.shape

    print('Model output shapes match model input shape.')

    params = sum(tf.reduce_prod(var.shape) for var in cnn.trainable_variables)
    print(f'Number of trainable parameters: {params}.')

    loss = compute_myrnenko_loss(x, y_true, y_pred, y_vae, z_mean, z_var, eps=1e-8)
    print(f'Loss: {loss}.')

    return y_hat, y_vae, z_mean, z_var


def main(args):
    if args.test_encoder:
        x = np.random.randn(1, 160, 192, 128, 4)
        _ = test_encoder(x)

    if args.test_decoder:
        enc32 = np.random.randn(1, 160, 192, 128, 32)
        enc64 = np.random.randn(1, 80, 96, 64, 64)
        enc128 = np.random.randn(1, 40, 48, 32, 128)
        enc256 = np.random.randn(1, 20, 24, 16, 256)
        enc_outs = (enc32, enc64, enc128, enc256)

        _ = test_decoder(enc_outs)

    if args.test_vae:
        enc256 = np.random.randn(1, 20, 24, 16, 256)
        _ = test_vae(enc256)

    if args.test_optimizer:
        test_optimizer()

    if args.test_loss:
        x = np.random.randn(1, 160, 192, 128, 4)
        y_true = np.random.randn(1, 160, 192, 128, 1)
        y_pred = np.random.randn(1, 160, 192, 128, 3)
        y_vae = np.random.randn(1, 160, 192, 128, 4)
        z_mean = np.random.randn(128, 1)
        z_var = np.random.randn(128, 1)
        test_loss(x, y_true, y_pred, y_vae, z_mean, z_var)

    if args.test_cnn:
        x = np.random.randn(1, 160, 192, 128, 4)
        y_true = np.random.randn(1, 160, 192, 128, 1)
        _ = test_cnn(x, y_true)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_decoder', action='store_true', default=False)
    parser.add_argument('--test_encoder', action='store_true', default=False)
    parser.add_argument('--test_vae', action='store_true', default=False)
    parser.add_argument('--test_optimizer', action='store_true', default=False)
    parser.add_argument('--test_loss', action='store_true', default=False)
    parser.add_argument('--test_cnn', action='store_true', default=False)
    args = parser.parse_args()

    main(args)
    