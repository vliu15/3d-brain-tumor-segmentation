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

from utils.arg_parser import prepro_parser


def get_npy_image(subject_folder, name):
    file_card = glob.glob(os.path.join(subject_folder, '*' + name + '.nii.gz'))[0]
    return nib.load(file_card).dataobj


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
        else:
            raise RuntimeError
        
        X.append(X_example)
        y.append(y_example)

    if data_format == 'channels_last':
        n_channels = X[0].shape[-1]
    elif data_format == 'channels_first':
        n_channels = X[0][0]

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

    X_train = np.stack(X[split:], axis=0)
    y_train = np.stack(y[split:], axis=0)
    X_val = np.stack(X[:split], axis=0)
    y_val = np.stack(y[:split], axis=0)

    return X_train, y_train, X_val, y_val


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


def normalize(voxel_mean, voxel_std, X, data_format='channels_last'):
    """Normalizes an array of features X given voxel-wise mean and std."""
    # Reshape mean and std into broadcastable with X, then loop per channel to avoid OOM.
    if data_format == 'channels_last':
        voxel_mean = np.reshape(voxel_mean, newshape=(1, 1, 1, voxel_mean.shape[-1]))
        voxel_std = np.reshape(voxel_std, newshape=(1, 1, 1, voxel_std.shape[-1]))
        for b in tqdm(range(X.shape[0]), leave=False):
            X[b] = (X[b] - voxel_mean) / voxel_std
    elif data_format == 'channels_first':
        voxel_mean = np.reshape(voxel_mean, newshape=(voxel_mean.shape[-1], 1, 1, 1))
        voxel_std = np.reshape(voxel_std, newshape=(voxel_std.shape[-1], 1, 1, 1))
        for b in tqdm(range(X.shape[0]), leave=False):
            X[b] = (X[b] - voxel_mean) / voxel_std

    return X


def intensify(shifts, scales, X, data_format='channels_last'):
    """Applies intensity shifting and scaling on X given shift and scale values."""
    # Reshape shifts and scales into broadcastable with X, then loop per channel to avoid OOM.
    if data_format == 'channels_last':
        shifts = np.reshape(shifts, newshape=(1, 1, 1, shifts.shape[-1]))
        scales = np.reshape(scales, newshape=(1, 1, 1, scales.shape[-1]))
        for b in tqdm(range(X.shape[0]), leave=False):
            X[b] = (X[b] - shifts) * scales
    elif data_format == 'channels_first':
        shifts = np.reshape(shifts, newshape=(shifts.shape[-1], 1, 1, 1))
        scales = np.reshape(scales, newshape=(scales.shape[-1], 1, 1, 1))
        for b in tqdm(range(X.shape[0]), leave=False):
            X[b] = (X[b] - shifts) * scales

    return X


def main(args):
    # Convert .nii.gz data files to a list Numpy arrays.
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
    np.save(f'./{args.out_folder}/voxel_mean_std.npy', {'mean': voxel_mean, 'std': voxel_std})
    print(f'Voxel-wise mean per channel: {voxel_mean}.')
    print(f'Voxel-wise std per channel: {voxel_std}.')

    # Normalize training and validation data.
    print('Apply per channel normalization.')
    num_shards = X_train.shape[0] // args.shard_size + 1
    for shard in tqdm(range(num_shards)):
        X_shard = normalize(voxel_mean,
                            voxel_std,
                            X_train[shard*args.shard_size:(shard+1)*args.shard_size, ...],
                            data_format=args.data_format)
        y_shard = y_train[shard*args.shard_size:(shard+1)*args.shard_size, ...]
        X_train[shard*args.shard_size:(shard+1)*args.shard_size, ...] = X_shard
    if args.create_val:
        num_shards = X_val.shape[0] // args.shard_size + 1
        for shard in tqdm(range(num_shards)):
            X_shard = normalize(voxel_mean,
                                voxel_std,
                                X_val[shard*args.shard_size:(shard+1)*args.shard_size, ...],
                                data_format=args.data_format)
            y_shard = y_val[shard*args.shard_size:(shard+1)*args.shard_size, ...]
            X_val[shard*args.shard_size:(shard+1)*args.shard_size, ...] = X_shard

    # Compute shifts and scales for intensification.
    print('Calculate intensity shifts and scales per channel.')
    shifts = np.random.uniform(low=0.0-args.intensity_shift, high=0.0+args.intensity_shift, size=(n_channels,))
    shifts = shifts * voxel_std
    scales = np.random.uniform(low=1.0-args.intensity_scale, high=1.0+args.intensity_scale, size=(n_channels,))
    np.save(f'./{args.out_folder}/intensity_shifts_scales.npy', {'shifts': shifts, 'scales': scales})
    print(f'Intensity shifts per channel: {shifts}.')
    print(f'Intensity scales per channel: {scales}.')

    # Apply intensity shifts and scales.
    print('Apply per channel intensity shifts and scales.')
    num_shards = X_train.shape[0] // args.shard_size + 1
    for shard in tqdm(range(num_shards)):
        X_shard = intensify(shifts,
                            scales,
                            X_train[shard*args.shard_size:(shard+1)*args.shard_size, ...],
                            data_format=args.data_format)
        y_shard = y_train[shard*args.shard_size:(shard+1)*args.shard_size, ...]
        X_train[shard*args.shard_size:(shard+1)*args.shard_size, ...] = X_shard
    if args.create_val:
        print('Save validation data.')
        num_shards = X_val.shape[0] // args.shard_size + 1
        for shard in tqdm(range(num_shards)):
            X_shard = intensify(shifts,
                                scales,
                                X_val[shard*args.shard_size:(shard+1)*args.shard_size, ...],
                                data_format=args.data_format)
            y_shard = y_val[shard*args.shard_size:(shard+1)*args.shard_size, ...]
            np.save(f'./{args.out_folder}/val_shard_{shard}', {'X': X_shard, 'y': y_shard})

    # Free memory.
    if args.create_val:
        del X_val
        del y_val

    # Randomly flip for data augmentation.
    print('Randomly augment training data.')
    augmented_X = []
    augmented_y = []
    for example, label in tqdm(zip(X_train, y_train)):
        if np.random.uniform() < args.mirror_prob:
            if args.data_format == 'channels_last':
                X_augment = np.flip(example, axis=(1, 2, 3))
                y_augment = np.flip(label, axis=(1, 2, 3))
            elif args.data_format == 'channels_first':
                X_augment = np.flip(example, axis=(2, 3, 4))
                y_augment = np.flip(label, axis=(2, 3, 4))
            augmented_X.append(X_augment)
            augmented_y.append(y_augment)

    print('Save training data.')
    num_shards = X_train.shape[0] // args.shard_size + 1
    for shard in tqdm(range(num_shards)):
        X_shard = X_train[shard*args.shard_size:(shard+1)*args.shard_size, ...]
        y_shard = y_train[shard*args.shard_size:(shard+1)*args.shard_size, ...]
        np.save(f'./{args.out_folder}/train_shard_{shard}.npy', {'X': X_shard, 'y': y_shard})

    # Free memory.
    del X_train
    del y_train

    print('Save augmented training data.')
    num_shards = len(augmented_X) // args.shard_size + 1
    for shard in tqdm(range(num_shards)):
        X_shard = np.stack(augmented_X[shard*args.shard_size:(shard+1)*args.shard_size], axis=0)
        y_shard = np.stack(augmented_y[shard*args.shard_size:(shard+1)*args.shard_size], axis=0)
        np.save(f'./{args.out_folder}/augment_shard_{shard}.npy', {'X': X_shard, 'y': y_shard})


if __name__ == '__main__':
    args = prepro_parser()
    main(args)
