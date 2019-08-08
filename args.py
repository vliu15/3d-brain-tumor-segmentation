import argparse
import shutil
import pickle
import copy
import os
import numpy as np


class BaseArgParser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def namespace_to_dict(self, args):
        """Turns a nested Namespace object to a nested dictionary"""
        args_dict = vars(copy.deepcopy(args))

        for arg in args_dict:
            obj = args_dict[arg]
            if isinstance(obj, argparse.Namespace):
                args_dict[arg] = self.namespace_to_dict(obj)

        return args_dict

    def fix_nested_namespaces(self, args):
        """Makes sure that nested Namespace work. Supports only one level of nesting."""
        group_name_keys = []

        for key in args.__dict__:
            if '.' in key:
                group, name = key.split('.')
                group_name_keys.append((group, name, key))

        for group, name, key in group_name_keys:
            if group not in args:
                args.__dict__[group] = argparse.Namespace()

            args.__dict__[group].__dict__[name] = args.__dict__[key]
            del args.__dict__[key]

    def parse_args(self):
        args = self.parser.parse_args()
        args = self.namespace_to_dict(args)
        self.fix_nested_namespaces(args)
        return args


class PreproArgParser(BaseArgParser):
    def __init__(self):
        super(PreproArgParser, self).__init__()

        self.parser.add_argument('--in_locs', type=str, required=True,
                help='Comma-separated list of paths to all data folders.')
        self.parser.add_argument('--modalities', type=str, required=True,
                help='Comma-separated list of all input modalities to use.')
        self.parser.add_argument('--truth', type=str, required=True,
                help='Truth label pattern to use.')

        self.parser.add_argument('--create_val', action='store_true', default=False,
                help='Whether to create validation set.')
        self.parser.add_argument('--out_loc', type=str, default='./data',
                help='Location to write preprocessed data.')

    def parse_args(self):
        args = self.parser.parse_args()

        # Create list of all input datasets.
        args.in_locs = args.in_locs.split(',')

        # Create list of all accepted modalities.
        args.modalities = args.modalities.split(',')

        # Create output directory if it doesn't already exist.
        if os.path.isdir(args.out_loc):
            shutil.rmtree(args.out_loc)

        args.train_loc = os.path.join(args.out_loc, 'train')
        args.val_loc = os.path.join(args.out_loc, 'val')

        os.mkdir(args.out_loc)
        os.mkdir(args.train_loc)
        os.mkdir(args.val_loc)

        return args


class TrainArgParser(BaseArgParser):
    def __init__(self):
        super(TrainArgParser, self).__init__()

        # Data args.
        self.parser.add_argument('--train_loc', type=str, required=True,
                help='Location of .tfrecords training data.')
        self.parser.add_argument('--prepro_loc', type=str, required=True,
                help='Location of preprocessed dump.')
        self.parser.add_argument('--val_loc', type=str, default='',
                help='Location of .tfrecords validation data.')

        # Checkpoint args.
        self.parser.add_argument('--save_folder', type=str, default='',
                help='Output folder to save checkpoints, logs, and configs.')
        self.parser.add_argument('--load_folder', type=str, default='',
                help='Input folder to load checkpoints and configs to resume training.')

        # Training args.
        self.parser.add_argument('--lr', type=float, default=1e-4,
                help='Initial learning rate for training.')
        self.parser.add_argument('--batch_size', type=int, default=1,
                help='Batch size to use in training.')
        self.parser.add_argument('--patience', type=int, default=-1,
                help='Number of epochs without validation improvement to stop training.')
        self.parser.add_argument('--n_epochs', type=int, default=300,
                help='Number of epochs to train for.')
        self.parser.add_argument('--gpu', action='store_true', default=False,
                help='Whether to train using GPU.')

        # Augmentation args.
        self.parser.add_argument('--crop_size', type=str, default='128,128,128',
                help='Crop size of image (comma-separated h,w,d).')

        # Model args.
        self.parser.add_argument('--data_format', type=str, dest='model_args.data_format',
                default='channels_first', choices=['channels_last', 'channels_first'],
                help='Data format to be passed through the model.')
        self.parser.add_argument('--base_filters', type=int, dest='model_args.base_filters', default=32,
                help='Number of filters in the base convolutional layer.')
        self.parser.add_argument('--depth', type=int, dest='model_args.depth', default=4,
                help='Number of spatial levels through the model.')
        self.parser.add_argument('--l2_scale', type=float, dest='model_args.l2_scale', default=1e-5,
                help='Scale of L2 regularization applied to all kernels.')
        self.parser.add_argument('--dropout', type=float, dest='model_args.dropout', default=0.2,
                help='Dropout ratio to apply to input data.')
        self.parser.add_argument('--groups', type=int, dest='model_args.groups', default=8,
                help='Number of groups in group normalization.')
        self.parser.add_argument('--reduction', type=int, dest='model_args.reduction', default=8,
                help='Size of reduction ratio in squeeze-excitation layers.')
        self.parser.add_argument('--downsampling', type=str, dest='model_args.downsampling',
                default='conv', choices=['conv', 'max', 'avg'],
                help='Type of downsampling method.')
        self.parser.add_argument('--upsampling', type=str, dest='model_args.upsampling',
                default='conv', choices=['conv', 'linear'],
                help='Type of upsampling method.')
        self.parser.add_argument('--out_ch', type=int, dest='model_args.out_ch', default=3,
                help='Number of output classes.')

    def parse_args(self):
        args = self.parser.parse_args()

        # Fix nested Namespaces.
        self.fix_nested_namespaces(args)
        args.data_format = args.model_args.data_format

        # Check data format and GPU compatibility.
        args.device = '/device:GPU:0' if args.gpu else '/cpu:0'
        if not args.gpu:
            assert args.model_args.data_format == 'channels_last', \
                'tf.keras.layers.Conv3D only supports `channels_last` input for CPU.'

        # Convert model args to dictionaries.
        args.model_args = self.namespace_to_dict(args.model_args)

        # Set crop size.
        args.crop_size = args.crop_size.split(',')
        args.crop_size = [int(s) for s in args.crop_size]

        # Load preprocessed stats.
        prepro = np.load(args.prepro_loc).item()
        args.prepro_size = [prepro['size']['h'], prepro['size']['w'], prepro['size']['d'], prepro['size']['c']]

        # Check that sizes work out.
        assert (args.model_args['base_filters'] / 2) % args.model_args['groups'] == 0, \
            'Base filters must be a multiple of {} for group normalization at lowest spatial level.'.format(args.model_args['groups'] * 2)

        assert args.model_args['base_filters'] % args.model_args['reduction'] == 0, \
            'Base filters must be a multiple of {} for squeeze-excitation reduction.'.format(args.model_args['reduction'])

        # Add args.model_args.in_ch for output size of variational autoencoder.
        args.model_args['in_ch'] = prepro['size']['c']

        # Check for checkpointing option.
        if args.load_folder:
            with open(os.path.join(args.load_folder, 'train_args.pkl'), 'rb') as f:
                chkpt_args = pickle.load(f)
                args.model_args = chkpt_args['model_args']
                args.crop_size = chkpt_args.crop_size
                assert isinstance(args.model_args, dict)
            args.save_folder = args.load_folder

        # Create checkpoint folder if necessary.
        if not os.path.isdir(args.save_folder):
            os.mkdir(args.save_folder)

        # Save training args.
        with open(os.path.join(args.save_folder, 'train_args.pkl'), 'wb') as f:
            pickle.dump(self.namespace_to_dict(args), f)
        
        return args


class TestArgParser(BaseArgParser):
    def __init__(self):
        super(TestArgParser, self).__init__()

        # Data.
        self.parser.add_argument('--in_locs', type=str, required=True,
                help='Comma-separated paths of test data.')
        self.parser.add_argument('--modalities', type=str, required=True,
                help='Comma-separated modalities to be used as input')
        self.parser.add_argument('--truth', type=str, default='',
                help='Truth label pattern to use (optional).')

        # Training and preprocessing stats.
        self.parser.add_argument('--tumor_prepro', type=str, required=True,
                help='Path to Numpy preprocessing dump for tumor segmentation.')
        self.parser.add_argument('--skull_prepro', type=str, default='',
                help='Path to Numpy preprocessing dump for skull segmentation.')
        self.parser.add_argument('--tumor_model', type=str, required=True,
                help='Path to checkpoint folder for tumor segmentation.')
        self.parser.add_argument('--skull_model', type=str, default='',
                help='Path to checkpoint folder for skull-stripping segmentation.')

        # Input normalization parameters.
        self.parser.add_argument('--order', type=int, default=3,
                help='Order of interpolation function to be used in voxel resizing.')
        self.parser.add_argument('--mode', type=str, default='reflect',
                help='Method of handling image edges in interpolation.')

        # Test time augmentation and segmentation.
        self.parser.add_argument('--spatial_tta', action='store_true', default=True,
                help='Whether to apply spatial augmentation on all spatial axes.')
        self.parser.add_argument('--channel_tta', type=int, default=0,
                help='Additional intensity shifting samples to take.')
        self.parser.add_argument('--threshold', type=float, default=0.5,
                help='Threshold at which to create mask from probabilities.')
        self.parser.add_argument('--gpu', action='store_true', default=False,
                help='Whether to evaluate on GPU.')

    def parse_args(self):
        args = self.parser.parse_args()
        args.modalities = args.modalities.split(',')
        args.in_locs = args.in_locs.split(',')

        # Assert proper combination of inputs.
        assert args.threshold > 0 and args.threshold < 1, \
            'Threshold must be a probability between (0, 1).'
        if args.skull_model:
            assert args.skull_prepro, 'Need skull preprocessing stats if model is provided.'
        args.skull_strip = bool(args.skull_model)

        # Load model args.
        with open(os.path.join(args.tumor_model, 'train_args.pkl'), 'rb') as f:
            train_args = pickle.load(f)
            args.tumor_model_args = train_args['model_args']
            args.tumor_spatial_res = 2 ** args.tumor_model_args['depth']
            args.tumor_crop_size = train_args['crop_size']
        if args.skull_model:
            with open(os.path.join(args.skull_model, 'train_args.pkl'), 'rb') as f:
                train_args = pickle.load(f)
                args.skull_model_args = train_args['model_args']
                args.skull_spatial_res = 2 ** args.skull_model_args['depth']
                args.skull_crop_size = train_args['crop_size']

        # Load prepro stats.
        args.tumor_prepro = np.load(args.tumor_prepro).item()
        if args.skull_prepro:
            args.skull_prepro = np.load(args.skull_prepro).item()

        # Check data format and GPU compatibility.
        args.device = '/device:GPU:0' if args.gpu else '/cpu:0'
        if not args.gpu:
            assert args.tumor_model_args['data_format'] == 'channels_last' and args.skull_model_args['data_format'], \
                'tf.keras.layers.Conv3D only supports `channels_last` input for CPU.'

        return args
