import argparse


def prepro_parser():
    parser = argparse.ArgumentParser()

    # Logistics.
    parser.add_argument('--brats_folder', type=str, required=True,
            help='Location of unzipped BraTS data.')
    parser.add_argument('--out_folder', type=str, default='./data',
            help='Location to write preprocessed data.')
    parser.add_argument('--shard_size', type=int, default=12,
            help='Number of shards to split training data into.')
    parser.add_argument('--data_format', type=str, default='channels_last',
            choices=['channels_last', 'channels_first'],
            help='Format of preprocessed data: `channels_last` or `channels_first`.')

    # Specifics.
    parser.add_argument('--create_val', action='store_true', default=False,
            help='Whether to create validation split from training data.')
    parser.add_argument('--intensify', action='store_true', default=False,
            help='Whether to perform intensity shift after normalization.')
    parser.add_argument('--intensity_shift', type=float, default=0.1,
            help='Scale of intensity shift to apply per channel after normalization.')
    parser.add_argument('--intensity_scale', type=float, default=0.1,
            help='Epsilon (from 1.0) of intensity scaling per channel.')
    parser.add_argument('--mirror_prob', type=float, default=0.5,
            help='Probability that each inputs are flipped across all 3 axes.')
    parser.add_argument('--n_crops', type=int, default=1,
            help='Number of random crops to sample per image.')

    args = parser.parse_args()

    return args


def train_parser():
    parser = argparse.ArgumentParser()

    # Hardware.
    parser.add_argument('--gpu', action='store_true', default=False,
            help='Whether to use GPU in training.')

    # Data.
    parser.add_argument('--train_loc', type=str, required=True,
            help='Location of preprocessed training data.')
    parser.add_argument('--val_loc', type=str, required=True,
            help='Location of preprocessed validation data.')
    parser.add_argument('--data_format', type=str, default='channels_last',
            choices=['channels_first', 'channels_last'],
            help='Format of input data: `channel_first` or `channels_last`.')

    # Training.
    parser.add_argument('--log_file', type=str, default='train.log',
            help='File for output logs.')
    parser.add_argument('--log_steps', type=int, default=100,
            help='Number of steps to output progress.')
    parser.add_argument('--save_file', type=str, default='chkpt.hdf5',
            help='File path to save best checkpoint.')
    parser.add_arggument('--load_file', type=str, default='',
            help='File path to load weights.')
    parser.add_argument('--patience', type=int, default=10,
            help='Number of epochs if validation scores have not improved \
                  before stopping training')

    # Optimization.
    parser.add_argument('--n_epochs', type=int, default=300,
            help='Total number of epochs to train for.')
    parser.add_argument('--lr', type=float, default=1e-4,
            help='Initial learning rate of Adam optimizer.')
    parser.add_argument('--batch_size', type=int, default=1,
            help='Batch size to be used in training.')

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

    args.device = '/device:GPU:0' if args.gpu else '/cpu:0'
    if not args.gpu:
        assert args.data_format == 'channels_last', \
            'tf.keras.layers.Conv3D only supports `channels_last` input for CPU.'

    return args
