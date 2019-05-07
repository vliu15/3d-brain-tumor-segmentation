import argparse
import tensorflow as tf
import numpy as np

from model.encoder import ConvEncoder
from model.decoder import ConvDecoder

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

    return logits


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_decoder', action='store_true', default=False)
    args = parser.parse_args()

    if args.test_decoder:
        enc32 = np.random.randn(1, 160, 192, 128, 32)
        enc64 = np.random.randn(1, 80, 96, 64, 64)
        enc128 = np.random.randn(1, 40, 48, 32, 128)
        enc256 = np.random.randn(1, 20, 24, 16, 256)
        enc_outs = (enc32, enc64, enc128, enc256)
    else:
        x = np.random.randn(160, 192, 128, 4)
        enc_outs = test_encoder(x)
    
    logits = test_decoder(enc_outs)
