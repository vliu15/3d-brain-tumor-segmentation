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
from utils.constants import *


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

    return X, y


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
        X_train = X_train.transpose(4, 0, 1, 2, 3).reshape(IN_CH, -1)
    elif data_format == 'channels_first':
        X_train = X_train.transpose(1, 0, 2, 3, 4).reshape(IN_CH, -1)

    # Compute mean and std for each channel
    voxel_mean = np.zeros(IN_CH)
    voxel_std = np.zeros(IN_CH)
    for i, channel in tqdm(enumerate(X_train), leave=False):
        voxel_mean[i] = np.mean(channel[channel != 0])
        voxel_std[i] = np.std(channel[channel != 0])

    return voxel_mean, voxel_std


def normalize(voxel_mean, voxel_std, X, shard_size, data_format='channels_last'):
    """Normalizes an array of features X given voxel-wise mean and std."""
    # Reshape mean and std into broadcastable with X, then loop per channel to avoid OOM.
    if data_format == 'channels_last':
        voxel_mean = voxel_mean.reshape(1, 1, 1, 1, IN_CH)
        voxel_std = voxel_std.reshape(1, 1, 1, 1, IN_CH)
    elif data_format == 'channels_first':
        voxel_mean = voxel_mean.reshape(1, IN_CH, 1, 1, 1)
        voxel_std = voxel_std.reshape(1, IN_CH, 1, 1, 1)

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
        shifts = shifts.reshape(1, 1, 1, 1, IN_CH)
        scales = scales.reshape(1, 1, 1, 1, IN_CH)
    elif data_format == 'channels_first':
        shifts = shifts.reshape(1, IN_CH, 1, 1, 1)
        scales = scales.reshape(1, IN_CH, 1, 1, 1)

    num_shards = X.shape[0] // shard_size + 1
    for shard in tqdm(range(num_shards)):
        X_shard = X[shard*shard_size:(shard+1)*shard_size, ...]
        X_shard = (X_shard + shifts) * scales
        X[shard*shard_size:(shard+1)*shard_size, ...] = X_shard
    
    return X


def example_to_tfrecords(X, y, writer):
    """Writes one (X, y) example to tf.TFRecord file."""
    example = {
        'X': tf.train.Feature(float_list=tf.train.FloatList(value=X.flatten())),
        'y': tf.train.Feature(float_list=tf.train.FloatList(value=y.flatten()))
    }
    example = tf.train.Features(feature=example)
    example = tf.train.Example(features=example)
    writer.write(example.SerializeToString())


def sample_crop(X, y, data_format='channels_last'):
    """Returns the crop of one (X, y) example."""
    def choose_corner(dim_len, dim_size):
        return np.random.randint(dim_size, dim_len)

    h = np.random.randint(H, RAW_H)
    w = np.random.randint(W, RAW_W)
    d = np.random.randint(D, RAW_D)

    if data_format == 'channels_last':
        X = X[h-H:h, w-W:w, d-D:d, :]
        y = y[h-H:h, w-W:w, d-D:d, :]
    elif data_format == 'channels_first':
        X = X[:, h-H:h, w-W:w, d-D:d]
        y = y[:, h-H:h, w-W:w, d-D:d]
        
    return X, y


def main(args):
    # Convert .nii.gz data files to a list Tensors.
    print('Convert data to Numpy arrays.')
    X_train, y_train = create_dataset(args.brats_folder, data_format=args.data_format)

    # Create dataset splits (if necessary).
    if args.create_val:
        print('Create train / val splits.')
        X_train, y_train, X_val, y_val = create_splits(X_train, y_train)
        print('X_train shape: {}.'.format(X_train.shape))
        print('y_train shape: {}.'.format(y_train.shape))
        print('X_val shape: {}.'.format(X_val.shape))
        print('y_val shape: {}.'.format(y_val.shape))
    else:
        X_train = np.stack(X_train, axis=0)
        y_train = np.stack(y_train, axis=0)
        print('X_train shape: {}.'.format(X_train.shape))
        print('y_train shape: {}.'.format(y_train.shape))

    # Compute mean and std for normalization.
    print('Calculate voxel-wise mean and std per channel of training set.')
    voxel_mean, voxel_std = compute_train_norm(X_train, data_format=args.data_format)
    print('Voxel-wise mean per channel: {}.'.format(voxel_mean))
    print('Voxel-wise std per channel: {}.'.format(voxel_std))

    # Normalize training and validation data.
    print('Apply per channel normalization.')
    X_train = normalize(voxel_mean, voxel_std, X_train, args.shard_size, data_format=args.data_format)
    if args.create_val:
        X_val = normalize(voxel_mean, voxel_std, X_val, args.shard_size, data_format=args.data_format)

        writer = tf.io.TFRecordWriter('./data/val.tfrecords')
        for X, y in tqdm(zip(X_val, y_val)):
            for _ in range(args.n_crops):
                X_crop, y_crop = sample_crop(X, y, data_format=args.data_format)
                example_to_tfrecords(X_crop, y_crop, writer)
        del X_val
        del y_val

    if args.intensify:
        # Compute shifts and scales for intensification.
        print('Calculate intensity shifts and scales per channel.')
        shifts = np.random.uniform(low=0.0-args.intensity_shift,
                                high=0.0+args.intensity_shift,
                                size=(IN_CH,))
        shifts = shifts * voxel_std
        scales = np.random.uniform(low=1.0-args.intensity_scale,
                                high=1.0+args.intensity_scale,
                                size=(IN_CH,))
        print('Intensity shifts per channel: {}.'.format(shifts))
        print('Intensity scales per channel: {}.'.format(scales))

        # Apply intensity shifts and scales.
        print('Apply per channel intensity shifts and scales.')
        X_train = intensify(shifts, scales, X_train, args.shard_size, data_format=args.data_format)

    # Randomly flip for data augmentation.
    print('Randomly augment training data, crop, and save.')
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
        for _ in range(args.n_crops):
            X_crop, y_crop = sample_crop(X, y, data_format=args.data_format)
            example_to_tfrecords(X_crop, y_crop, writer)


if __name__ == '__main__':
    args = prepro_parser()

    main(args)
