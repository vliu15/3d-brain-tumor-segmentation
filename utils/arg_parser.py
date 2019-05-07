import argparse


def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_format', type=str, default='channels_last',
            help='Format of input data, either `channel_first` or `channels_last`.')
    parser.add_argument('--lr', type=float, default=1e-4,
            help='Initial learning rate of Adam optimizer.')
    parser.add_argument('--conv_kernel_size', type=int, default=3,
            help='Size of convolutional kernels throughout the model.')
    parser.add_argument('--gn_groups', type=int, default=8,
            help='Size of groups for group normalization.')
    parser.add_argument('--dropout', type=float, default=0.2,
            help='Dropout rate for dropout layer after initial convolution.')
    parser.add_argument('--l2_scale', type=float, default=1e-5,
            help='Scale of L2-regularization for convolution kernel weights.')

    args = parser.parse_args()
    return args