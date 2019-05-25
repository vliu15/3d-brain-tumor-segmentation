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
    try:
        file_card = glob.glob(os.path.join(subject_folder, '*' + name + '.nii.gz'))[0]
    except:
        file_card = glob.glob(os.path.join(subject_folder, '*' + name + '.nii'))[0]
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
    for subject_folder in tqdm(glob.glob(os.path.join(brats_folder, '*', '*')), desc='Load data    '):

        # Create corresponding output folder.
        if os.path.isdir(subject_folder):
            axis = -1 if data_format == 'channels_last' else 0

            # Stack modalities in channel dimension.
            X_example = np.stack(
                [get_npy_image(subject_folder, name) for name in BRATS_MODALITIES], axis=axis)

            # Replace label 4 with 3 for softmax labels at output time.
            y_example = np.expand_dims(get_npy_image(subject_folder, TRUTH), axis=axis)
            np.place(y_example, y_example >= 4, [3])

        X.append(X_example)
        y.append(y_example)

    return X, y


def create_splits(X, y):
    data = list(zip(X, y))
    random.shuffle(data)
    X, y = list(zip(*data))
    split = len(data) // 11

    del data

    X = np.stack(X, axis=0)
    y = np.stack(y, axis=0)

    return X[split:], y[split:], X[:split], y[:split]


def image_norm(X_train, data_format='channels_last'):
    """Returns mean and standard deviation per channel for training inputs."""
    transpose_order = (4, 0, 1, 2, 3) if data_format == 'channels_last' else (1, 0, 2, 3, 4)
    X_train = X_train.transpose(*transpose_order).reshape(IN_CH, -1)

    # Compute mean and std for each channel.
    voxel_mean = np.zeros(IN_CH)
    voxel_std = np.zeros(IN_CH)
    for i, channel in tqdm(enumerate(X_train), leave=False):
        voxel_mean[i] = np.mean(channel[channel != 0])
        voxel_std[i] = np.std(channel[channel != 0])

    voxel_mean = voxel_mean.reshape(1, 1, 1, voxel_mean.shape[0]).repeat(*X_train[0].shape)
    voxel_std = voxel_std.reshape(1, 1, 1, voxel_std.shape[0]).repeat(*X_train[0].shape)

    return voxel_mean, voxel_std


def pixel_norm(X_train, data_format='channels_last'):
    if data_format == 'channels_last':
        get_mean = lambda X, h, w, d, c: X[:, h, w, d, c].mean()
        get_std = lambda X, h, w, d, c: X[:, h, w, d, c].std()
    elif data_format == 'channels_first':
        get_mean = lambda X, h, w, d, c: X[:, c, h, w, d].mean()
        get_std = lambda X, h, w, d, c: X[:, c, h, w, d].std()

    # Compute mean and std for each position.
    voxel_mean = np.zeros_like(X_train[0])
    voxel_std = np.zeros_like(X_train[0])
    for h in range(H):
        for w in range(W):
            for d in range(D):
                for c in range(IN_CH):
                    voxel_mean[h, w, d, c] = get_mean(X_train, h, w, d, c)
                    voxel_std[h, w, d, c] = get_std(X_train, h, w, d, c)

    return voxel_mean, voxel_std


def normalize(voxel_mean, voxel_std, X):
    """Normalizes an array of features X given voxel-wise mean and std."""
    for i in tqdm(range(X.shape[0]), desc='Normalize    '):
        X[i] = (X[i] - voxel_mean) / (voxel_std + 1e-8)

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
    # Sample corner points.
    h = np.random.randint(H, RAW_H)
    w = np.random.randint(W, RAW_W)
    d = np.random.randint(D, RAW_D)

    if data_format == 'channels_last':
        X = X[h-H:h, w-W:w, d-D:d, :]
        y = y[h-H:h, w-W:w, d-D:d, :]
    else:
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

    # Normalize training and validation data.
    if args.norm == 'image':
        print('Apply image-wise normalization per channel.')
        voxel_mean, voxel_std = image_norm(X_train, data_format=args.data_format)
        np.save(os.path.join(args.out_folder, 'image_mean_std.npy'),
                {'mean': voxel_mean, 'std': voxel_std})
    elif args.norm == 'pixel':
        print('Apply pixel-wise normalization per channel.')
        voxel_mean, voxel_std = pixel_norm(X_train, data_format=args.data_format)
        np.save(os.path.join(args.out_folder, 'pixel_mean_std.npy'),
                {'mean': voxel_mean, 'std': voxel_std})

    X_train = normalize(voxel_mean, voxel_std, X_train)
    if args.create_val:
        X_val = normalize(voxel_mean, voxel_std, X_val)

        writer = tf.io.TFRecordWriter(os.path.join(args.out_folder, 'val.tfrecords'))
        for X, y in tqdm(zip(X_val, y_val)):
            for _ in range(args.n_crops):
                X_crop, y_crop = sample_crop(X, y, data_format=args.data_format)
                example_to_tfrecords(X_crop, y_crop, writer)
        del X_val
        del y_val

    # Randomly flip for data augmentation.
    print('Randomly augment training data, crop, and save.')
    writer = tf.io.TFRecordWriter(os.path.join(args.out_folder, 'train.tfrecords'))

    for X, y in tqdm(zip(X_train, y_train)):
        # Augment.
        if np.random.uniform() < args.mirror_prob:
            axis = (1, 2, 3) if args.data_format == 'channels_last' else (2, 3, 4)
            X_augment = np.flip(X, axis=axis)
            y_augment = np.flip(y, axis=axis)

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
