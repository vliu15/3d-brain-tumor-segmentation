"""Handles all command-line argument parsing."""
import argparse
import os


def add_model_args(parser):
    # Architectural
    parser.add_argument('--downsamp', type=str, default='max',
            choices=['max', 'avg', 'conv'],
            help='Method of downsampling.')
    parser.add_argument('--upsamp', type=str, default='linear',
            choices=['linear', 'conv'],
            help='Method of upsampling.')
    parser.add_argument('--norm', type=str, default='group',
            choices=['group', 'batch', 'layer'],
            help='Which normalization to use throughout the network.')
    
    # Parameters
    parser.add_argument('--conv_kernel_size', type=int, default=3,
            help='Size of convolutional kernels throughout the model.')
    parser.add_argument('--gn_groups', type=int, default=8,
            help='Size of groups for group normalization.')
    parser.add_argument('--use_se', action='store_true', default=False,
            help='Whether to use SENet blocks instead of ResNet blocks.')
    parser.add_argument('--se_reduction', type=int, default=2,
            help='Reduction ratio in excitation layers of SENet blocks.')
    parser.add_argument('--l2_scale', type=float, default=1e-5,
            help='Scale of L2-regularization for convolution kernel weights.')
    parser.add_argument('--kernel_init', type=str, default='he_normal',
            choices=['he_normal', 'he_uniform', 'glorot_normal', 'glorot_uniform'],
            help='Kernel initialization to use for weight initialization.')
    return parser


def prepro_parser():
    parser = argparse.ArgumentParser()

    # Logistics.
    parser.add_argument('--brats_folder', type=str, required=True,
            help='Location of unzipped BraTS data.')
    parser.add_argument('--out_folder', type=str, default='./data',
            help='Location to write preprocessed data.')
    parser.add_argument('--data_format', type=str, default='channels_last',
            choices=['channels_last', 'channels_first'],
            help='Format of preprocessed data: `channels_last` or `channels_first`.')

    # Preprocessing.
    parser.add_argument('--create_val', action='store_true', default=False,
            help='Whether to create validation split from training data.')
    parser.add_argument('--norm', type=str, default='image',
            choices=['image', 'pixel'],
            help='Type of normalization to apply.')
    parser.add_argument('--mirror_prob', type=float, default=0.75,
            help='Probability that each inputs are flipped across all 3 axes.')
    parser.add_argument('--n_crops', type=int, default=3,
            help='Number of random crops to sample per image.')

    args = parser.parse_args()

    if not os.path.isdir(args.out_folder):
        os.mkdir(args.out_folder)

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
    parser.add_argument('--save_file', type=str, default='chkpt.hdf5',
            help='File path to save best checkpoint.')
    parser.add_argument('--load_file', type=str, default='',
            help='File path to load complete checkpoint.')
    parser.add_argument('--log_steps', type=int, default=-1,
            help='Frequency at which to output training statistics.')
    parser.add_argument('--patience', type=int, default=10,
            help='Number of epochs if validation scores have not improved \
                  before stopping training')

    # Optimization.
    parser.add_argument('--n_epochs', type=int, default=300,
            help='Total number of epochs to train for.')
    parser.add_argument('--lr', type=float, default=1e-5,
            help='Initial learning rate of Adam optimizer.')
    parser.add_argument('--warmup_epochs', type=int, default=10,
            help='Number of epochs for learning rate warmup.')
    parser.add_argument('--batch_size', type=int, default=1,
            help='Batch size to be used in training.')

    # Model.
    parser = add_model_args(parser)

    args = parser.parse_args()

    args.device = '/device:GPU:0' if args.gpu else '/cpu:0'
    if not args.gpu:
        assert args.data_format == 'channels_last', \
            'tf.keras.layers.Conv3D only supports `channels_last` input for CPU.'

    return args


def test_parser():
    parser = argparse.ArgumentParser()

    # Hardware.
    parser.add_argument('--gpu', action='store_true', default=False,
            help='Whether to use GPU on evaluation.')

    # Data.
    parser.add_argument('--test_folder', type=str, required=True,
            help='Location of test set data.')
    parser.add_argument('--data_format', type=str, default='channels_last',
            choices=['channels_first', 'channels_last'],
            help='Format of input data: `channel_first` or `channels_last`.')

    # Model path.
    parser.add_argument('--chkpt_file', type=str, required=True,
            help='Path to weights dump of model training.')
    parser.add_argument('--prepro_file', type=str, required=True,
            help='Path to dumped preprocessed stats.')

    # Model.
    parser = add_model_args(parser)

    args = parser.parse_args()

    args.device = '/device:GPU:0' if args.gpu else '/cpu:0'
    if not args.gpu:
        assert args.data_format == 'channels_last', \
            'tf.keras.layers.Conv3D only supports `channels_last` input for CPU.'

    return args
