import argparse


def train_parser():
    parser = argparse.ArgumentParser()

    # Hardware.
    parser.add_argument('--gpu', action='store_true',
            help='Whether to use GPU in training.')

    # Data.
    parser.add_argument('--train_loc', type=str, required=True,
            help='Location of preprocessed training data.')
    parser.add_argument('--val_loc', type=str, required=True,
            help='Location of preprocessed validation data.')
    parser.add_argument('--data_format', type=str, default='channels_last',
            help='Format of input data: `channel_first` or `channels_last`.')

    # Training.
    parser.add_argument('--n_epochs', type=int, default=300,
            help='Total number of epochs to train for.')
    parser.add_argument('--lr', type=float, default=1e-4,
            help='Initial learning rate of Adam optimizer.')
    parser.add_argument('--batch_size', type=int, default=1,
            help='Batch size to be used in training.')
    parser.add_argument('--log_file', type=str, default='',
            help='File for output logs.')

    # Model.
    parser.add_argument('--conv_kernel_size', type=int, default=3,
            help='Size of convolutional kernels throughout the model.')
    parser.add_argument('--gn_groups', type=int, default=8,
            help='Size of groups for group normalization.')
    parser.add_argument('--dropout', type=float, default=0.2,
            help='Dropout rate for dropout layer after initial convolution.')
    parser.add_argument('--l2_scale', type=float, default=1e-5,
            help='Scale of L2-regularization for convolution kernel weights.')

    args = parser.parse_args()

    if not args.use_gpu:
        assert args.data_format == 'channels_last', \
            'tf.keras.layers.Conv3D only supports `channels_last` input for CPU.'

    return args
