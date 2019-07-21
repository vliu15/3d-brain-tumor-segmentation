"""Handles all command-line argument parsing."""
import argparse
import json
import os


def prepro_parser():
    parser = argparse.ArgumentParser()

    # Logistics.
    parser.add_argument('--brats_folder', type=str, required=True,
            help='Location of unzipped BraTS data.')
    parser.add_argument('--out_folder', type=str, default='./data',
            help='Location to write preprocessed data.')

    # Preprocessing.
    parser.add_argument('--create_val', action='store_true', default=False,
            help='Whether to create validation split from training data.')
    parser.add_argument('--norm', type=str, default='image',
            choices=['image', 'pixel', 'scale'],
            help='Type of normalization to apply.')

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
    parser.add_argument('--prepro_loc', type=str, required=True,
            help='Location of preprocessed statistics.')

    # Training.
    parser.add_argument('--log_file', type=str, default='train.log',
            help='File for output logs.')
    parser.add_argument('--save_file', type=str, default='chkpt.hdf5',
            help='File path to save best checkpoint.')
    parser.add_argument('--load_file', type=str, default='',
            help='File path to load weights checkpoint.')
    parser.add_argument('--args_file', type=str, default='',
            help='File path to load model args corresponding to checkpoint.')
    parser.add_argument('--patience', type=int, default=-1,
            help='Number of epochs without validation loss decrease to stop training.')

    # Optimization.
    parser.add_argument('--n_epochs', type=int, default=300,
            help='Total number of epochs to train for.')
    parser.add_argument('--lr', type=float, default=1e-4,
            help='Initial learning rate of Adam optimizer.')
    parser.add_argument('--batch_size', type=int, default=1,
            help='Batch size to be used in training.')

    args = parser.parse_args()

    args.device = '/device:GPU:0' if args.gpu else '/cpu:0'
    if not args.gpu:
        assert args.model_args.data_format == 'channels_last', \
            'tf.keras.layers.Conv3D only supports `channels_last` input for CPU.'

    if args.load_file:
        assert args.args_file == True, \
            'Loading from checkpoint requires a corresponding set of model args.'

    # Set model initialization arguments.
    if args.args_file:
        with open(args.args_file, 'r') as f:
            args.model_args = json.load(f)
    else:
        args.model_args = {}

    args.data_format = args.model_args['data_format']

    return args


def test_parser():
    parser = argparse.ArgumentParser()

    # Hardware.
    parser.add_argument('--gpu', action='store_true', default=False,
            help='Whether to use GPU on evaluation.')

    # Model.
    parser.add_argument('--chkpt_file', type=str, required=True,
            help='Path to weights dump of model training.')
    parser.add_argument('--model_args', type=str, required=True,
            help='Path to pickle dump of model args to reinitialize model.')

    # Data.
    parser.add_argument('--test_folder', type=str, required=True,
            help='Location of test set data.')
    parser.add_argument('--prepro_file', type=str, required=True,
            help='Path to dumped preprocessed stats.')

    # Interpolation.
    parser.add_argument('--order', type=int, default=3,
            help='Polynomial order of spline interpolation.')
    parser.add_argument('--mode', type=str, default='reflect',
            help='Method of extrapolation for spline interpolation.')

    # Generation.
    parser.add_argument('--stride', type=int, default=64,
            help='Stride at which to take sample crops from inpute image.')
    parser.add_argument('--batch_size', type=int, default=8,
            help='Batch size of crops to load into model.')

    # Segmentation.
    parser.add_argument('--threshold', type=float, default=0.5,
            help='Threshold at which to create mask from probabilities.')
    
    args = parser.parse_args()

    # Set model initialization arguments.
    args.model_args = json.load(args.model_args)

    return args
