import tensorflow as tf
import numpy as np

from model.encoder import ConvEncoder

def test_encoder():
    """Tests that the encoder output shapes are correct."""
    enc = ConvEncoder(data_format='channels_last',
                      kernel_size=(3, 3, 3),
                      groups=8,
                      dropout=0.2)
    x = np.random.randn(160, 192, 128, 4)

    conv_out_32, conv_out_64, conv_out_128, encoder_out = enc(x)
    assert conv_out_32.shape == (1, 160, 192, 128, 32)
    assert conv_out_64.shape == (1, 80, 96, 64, 64)
    assert conv_out_128.shape == (1, 40, 48, 32, 128)
    assert encoder_out.shape == (1, 20, 24, 16, 256)

    print('All shapes match.')


if __name__ == '__main__':
    test_encoder()
