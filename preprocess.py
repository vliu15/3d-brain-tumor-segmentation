'''
    Tools for converting, normalizing, and fixing the brats data.
    Taken from https://github.com/ellisdg/3DUnetCNN/blob/master/brats/preprocess.py.
'''
import glob
import os
import random

import numpy as np
import nibabel as nib
from tqdm import tqdm
import tensorflow as tf

from utils.arg_parser import prepro_parser


def get_npy_image(subject_folder, name):
    """Returns np.array from .nii.gz files."""
    file_card = glob.glob(os.path.join(subject_folder, '*' + name + '.nii.gz'))[0]
    return np.array(nib.load(file_card).dataobj)


def create_dataset(brats_folder, data_format='channels_last'):
    """Returns lists of data inputs and labels.

        Args:
            brats_folder: path to the BraTS unprocessed data folder.
            data_format: format of the target preprocessed data, either
                `channels_first` or `channels_last`.

        Returns:
            X: a list of examples, of shapes
                if data_format == 'channels_first': shape=(4, 240, 240, 155)
                if data_format == 'channels_last': shape=(240, 240, 155, 4)
            y: a list of corresponding labels, of shapes
                if data_format == 'channels_first': shape=(1, 240, 240, 155)
                if data_format == 'channels_last': shape=(240, 240, 155, 1)
    """
    ALL_MODALITIES = ['t1', 't1ce', 'flair', 't2']
    TRUTH = 'seg'

    X = []
    y = []
    # Loop through each folder with `.nii.gz` files.
    for subject_folder in tqdm(glob.glob(os.path.join(brats_folder, '*', '*')), leave=False):

        # Create corresponding output folder.
        if os.path.isdir(subject_folder):
            if data_format == 'channels_last':
                X_example = np.stack(
                    [get_npy_image(subject_folder, name) for name in ALL_MODALITIES], axis=-1)
                y_example = np.expand_dims(get_npy_image(subject_folder, TRUTH), axis=-1)
            elif data_format == 'channels_first':
                X_example = np.stack(
                    [get_npy_image(subject_folder, name) for name in ALL_MODALITIES], axis=0)
                y_example = np.expand_dims(get_npy_image(subject_folder, TRUTH), axis=0)
        
        X.append(X_example)
        y.append(y_example)

    if data_format == 'channels_last':
        n_channels = X[0].shape[-1]
    elif data_format == 'channels_first':
        n_channels = X[0].shape[0]

    return X, y, n_channels


def create_splits(X, y):
    """Creates 10:1 train:val split with the data.
    
        Args:
            X: list of inputs (np.ndarray's)
            y: list of labels (np.ndarray's)

        Returns:
    """
    data = list(zip(X, y))
    random.shuffle(data)
    X, y = list(zip(*data))
    split = len(data) // 11

    del data

    X = np.stack(X, axis=0)
    y = np.stack(y, axis=0)

    return X[split:], y[split:], X[:split], y[:split]


def compute_train_norm(X_train, data_format='channels_last'):
    """Returns mean and standard deviation per channel for training inputs."""
    if data_format == 'channels_last':
        X_train = np.reshape(X_train, newshape=(X_train.shape[-1], -1))
    elif data_format == 'channels_first':
        X_train = np.reshape(X_train, newshape=(X_train.shape[0], -1))

    # Compute mean and std for each channel
    voxel_mean = np.zeros(X_train.shape[0])
    voxel_std = np.zeros(X_train.shape[0])
    for i, channel in tqdm(enumerate(X_train), leave=False):
        voxel_mean[i] = np.mean(channel[channel != 0])
        voxel_std[i] = np.std(channel[channel != 0])

    return voxel_mean, voxel_std


def normalize(voxel_mean, voxel_std, X, shard_size, data_format='channels_last'):
    """Normalizes an array of features X given voxel-wise mean and std."""
    # Reshape mean and std into broadcastable with X, then loop per channel to avoid OOM.
    if data_format == 'channels_last':
        voxel_mean = np.reshape(voxel_mean, newshape=(1, 1, 1, 1, voxel_mean.shape[-1]))
        voxel_std = np.reshape(voxel_std, newshape=(1, 1, 1, 1, voxel_std.shape[-1]))
    elif data_format == 'channels_first':
        voxel_mean = np.reshape(voxel_mean, newshape=(1, voxel_mean.shape[-1], 1, 1, 1))
        voxel_std = np.reshape(voxel_std, newshape=(1, voxel_std.shape[-1], 1, 1, 1))

    num_shards = X.shape[0] // shard_size + 1
    for shard in tqdm(range(num_shards)):
        X_shard = X[shard*shard_size:(shard+1)*shard_size, ...]
        X_shard = (X_shard - voxel_mean) / voxel_std
        X[shard*shard_size:(shard+1)*shard_size, ...] = X_shard

    return X


def intensify(shifts, scales, X, shard_size, data_format='channels_last'):
    """Applies intensity shifting and scaling on X given shift and scale values."""
    # Reshape shifts and scales into broadcastable with X, then loop per channel to avoid OOM.
    if data_format == 'channels_last':
        shifts = np.reshape(shifts, newshape=(1, 1, 1, 1, shifts.shape[-1]))
        scales = np.reshape(scales, newshape=(1, 1, 1, 1, scales.shape[-1]))
    elif data_format == 'channels_first':
        shifts = np.reshape(shifts, newshape=(1, shifts.shape[-1], 1, 1, 1))
        scales = np.reshape(scales, newshape=(1, scales.shape[-1], 1, 1, 1))

    num_shards = X.shape[0] // shard_size + 1
    for shard in tqdm(range(num_shards)):
        X_shard = X[shard*shard_size:(shard+1)*shard_size, ...]
        X_shard = (X_shard + shifts) * scales
        X[shard*shard_size:(shard+1)*shard_size, ...] = X_shard
    
    return X


def example_to_tfrecords(X, y, writer):
    """Writes one (X, y) example to tf.TFRecord file."""
    example = {
        'X': tf.train.Feature(float_list=tf.train.FloatList(value=tf.reshape(X, shape=[-1]))),
        'y': tf.train.Feature(float_list=tf.train.FloatList(value=tf.reshape(X, shape=[-1])))
    }
    example = tf.train.Features(feature=example)
    examplle = tf.train.Example(features=example)
    writer.write(example.SerializeToString())


def sample_crop(X, y, data_format='channels_last'):
    """Returns the crop of one (X, y) example."""
    if data_format == 'channels_last':
        size = (160, 192, 128, 5)
        value = np.concatenate([X, y], axis=-1)
        value = tf.image.random_crop(value, size)
        X = value[..., :4]
        y = value[..., 4]
    elif data_format == 'channels_first':
        size = (5, 160, 192, 128)
        value = np.concatenate([X, y], axis=0)
        value = tf.image.random_crop(value, size)
        X = value[:4, ...]
        y = value[4, ...]
    
    return X, y


def main(args):
    # Convert .nii.gz data files to a list Tensors.
    print('Convert data to Numpy arrays.')
    X_train, y_train, n_channels = create_dataset(args.brats_folder, data_format=args.data_format)

    # Create dataset splits (if necessary).
    if args.create_val:
        print('Create train / val splits.')
        X_train, y_train, X_val, y_val = create_splits(X_train, y_train)
        print(f'X_train shape: {X_train.shape}.')
        print(f'y_train shape: {y_train.shape}.')
        print(f'X_val shape: {X_val.shape}.')
        print(f'y_val shape: {y_val.shape}.')
    else:
        X_train = np.stack(X_train, axis=0)
        y_train = np.stack(y_train, axis=0)
        print(f'X_train shape: {X_train.shape}.')
        print(f'y_train shape: {y_train.shape}.')

    # Compute mean and std for normalization.
    print('Calculate voxel-wise mean and std per channel of training set.')
    voxel_mean, voxel_std = compute_train_norm(X_train, data_format=args.data_format)
    print(f'Voxel-wise mean per channel: {voxel_mean}.')
    print(f'Voxel-wise std per channel: {voxel_std}.')

    # Normalize training and validation data.
    print('Apply per channel normalization.')
    X_train = normalize(voxel_mean, voxel_std, X_train, args.shard_size, data_format=args.data_format)
    if args.create_val:
        X_val = normalize(voxel_mean, voxel_std, X_val, args.shard_size, data_format=args.data_format)

    # Compute shifts and scales for intensification.
    print('Calculate intensity shifts and scales per channel.')
    shifts = np.random.uniform(low=0.0-args.intensity_shift,
                            high=0.0+args.intensity_shift,
                            size=(n_channels,))
    shifts = shifts * voxel_std
    scales = np.random.uniform(low=1.0-args.intensity_scale,
                            high=1.0+args.intensity_scale,
                            size=(n_channels,))
    print(f'Intensity shifts per channel: {shifts}.')
    print(f'Intensity scales per channel: {scales}.')

    # Apply intensity shifts and scales.
    print('Apply per channel intensity shifts and scales.')
    X_train = intensify(shifts, scales, X_train, args.shard_size, data_format=args.data_format)
    if args.create_val:
        X_val = intensify(shifts, scales, X_val, args.shard_size, data_format=args.data_format)

        writer = tf.io.TFRecordWriter('./data/val.tfrecords')
        for X, y in tqdm(zip(X_val, y_val)):
            for _ in range(args.n_crops):
                X_crop, y_crop = sample_crop(X, y, data_format=args.data_format)
                example_to_tfrecords(X_crop, y_crop, writer)
        del X_val
        del y_val

    # Randomly flip for data augmentation.
    print('Randomly augment training data.')
    writer = tf.io.TFRecordWriter('./data/train.tfrecords')

    for X, y in tqdm(zip(X_train, y_train)):
        # Augment.
        if np.random.uniform() < args.mirror_prob:
            if args.data_format == 'channels_last':
                X_augment = np.flip(X, axis=(1, 2, 3))
                y_augment = np.flip(y, axis=(1, 2, 3))
            elif args.data_format == 'channels_first':
                X_augment = np.flip(X, axis=(2, 3, 4))
                y_augment = np.flip(y, axis=(2, 3, 4))

            # Write augmented crops to TFRecord.
            for _ in range(args.n_crops):
                X_crop, y_crop = sample_crop(X_augment, y_augment, data_format=args.data_format)
                example_to_tfrecords(X_crop, y_crop, writer)

        # Write original crop to TFRecord.
        for _ in range(args.ncrops):
            X_crop, y_crop = sample_crop(X_augment, y_augment, data_format=args.data_format)
            example_to_tfrecords(X_crop, y_crop, writer)


if __name__ == '__main__':
    args = prepro_parser()

    main(args)
