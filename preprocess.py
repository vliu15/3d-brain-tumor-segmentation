"""Contains preprocessing script."""
import glob
import os
import random
import numpy as np
import nibabel as nib
from tqdm import tqdm
import tensorflow as tf

from utils.arg_parser import prepro_parser
from utils.constants import BRATS_MODALITIES, TRUTH, IN_CH, RAW_H, RAW_W, RAW_D


def get_npy_image(subject_folder, name):
    try:
        file_card = glob.glob(os.path.join(subject_folder, '*' + name + '.nii.gz'))[0]
    except:
        file_card = glob.glob(os.path.join(subject_folder, '*' + name + '.nii'))[0]
    return np.array(nib.load(file_card).dataobj).astype(np.float32)


def create_dataset(brats_folder):
    """Returns lists of data inputs and labels."""
    X = []
    y = []

    h_min, h_max = RAW_H - 1, 0
    w_min, w_max = RAW_W - 1, 0
    d_min, d_max = RAW_D - 1, 0
    # Loop through each folder with `.nii.gz` files.
    for subject_folder in tqdm(glob.glob(os.path.join(brats_folder, '*', '*')), desc='Load data    '):

        # Create corresponding output folder.
        if os.path.isdir(subject_folder):
            # Stack modalities in channel dimension.
            X_ = np.stack(
                [get_npy_image(subject_folder, name) for name in BRATS_MODALITIES], axis=-1)

            # Replace label 4 with 3 for softmax labels at output time.
            y_ = np.expand_dims(get_npy_image(subject_folder, TRUTH), axis=-1)
            np.place(y_, y_ >= 4, [3])

            # Determine smallest bounding box.
            h_min_, h_max_ = np.where(np.any(X_, axis=(1, 2, -1)))[0][[0, -1]]
            h_min = min(h_min, h_min_)
            h_max = max(h_max, h_max_)

            w_min_, w_max_ = np.where(np.any(X_, axis=(0, 2, -1)))[0][[0, -1]]
            w_min = min(w_min, w_min_)
            w_max = max(w_max, w_max_)

            d_min_, d_max_ = np.where(np.any(X_, axis=(0, 1, -1)))[0][[0, -1]]
            d_min = min(d_min, d_min_)
            d_max = max(d_max, d_max_)

            X.append(X_)
            y.append(y_)

    # Crop to minimal size for all examples.
    X = [X_[h_min:h_max, w_min:w_max, d_min:d_max, :] for X_ in X]
    y = [y_[h_min:h_max, w_min:w_max, d_min:d_max, :] for y_ in y]

    print('Maximal crop size: [{}, {}, {}, 2]'.format(h_max-h_min-1, w_max-w_min-1, d_max-d_min-1))

    crop_indices = {'h_min': h_min, 'h_max': h_max, 
                    'w_min': w_min, 'w_max': w_max,
                    'd_min': d_min, 'd_max': d_max}

    return X, y, crop_indices


def create_splits(X, y):
    """Creates 10:1 train:val split."""
    data = list(zip(X, y))
    random.shuffle(data)
    X, y = list(zip(*data))
    split = len(data) // 11

    del data

    X = np.stack(X, axis=0)
    y = np.stack(y, axis=0)

    return (X[split:], y[split:]), (X[:split], y[:split])


def image_norm(X_train):
    """Returns image-wise mean and standard deviation per channel."""
    X_train = X_train.transpose(4, 0, 1, 2, 3).reshape(IN_CH, -1)

    # Compute mean and std for each channel.
    voxel_mean = np.zeros((1, 1, 1, IN_CH))
    voxel_std = np.zeros((1, 1, 1, IN_CH))
    for i, channel in tqdm(enumerate(X_train)):
        voxel_mean[..., i] = np.mean(channel[channel != 0])
        voxel_std[..., i] = np.std(channel[channel != 0])

    return voxel_mean, voxel_std


def pixel_norm(X_train):
    """Returns pixel-wise mean and standard deviation per channel."""
    # Compute mean and std for each position.
    voxel_mean = np.zeros_like(X_train[0])
    voxel_std = np.zeros_like(X_train[0])
    H_, W_, D_, C_ = X_train[0].shape
    for h in range(H_):
        for w in range(W_):
            for d in range(D_):
                for c in range(C_):
                    voxel_mean[h, w, d, c] = np.mean(X[:, h, w, d, c])
                    voxel_std[h, w, d, c] = np.std(X[:, h, w, d, c])

    return voxel_mean, voxel_std


def scale_norm(X_train):
    """Returns maximum and minimum number per channel."""
    voxel_mean = float('inf') * np.ones((1, 1, 1, IN_CH))
    voxel_std = -float('inf') * np.ones((1, 1, 1, IN_CH))

    for i in range(X_train.shape[0]):
        voxel_mean = np.minimum(voxel_mean, np.amin(X_train[i], axis=(0, 1, 2)))
        voxel_std = np.maximum(voxel_std, np.amax(X_train[i], axis=(0, 1, 2)))

    return voxel_mean, voxel_std


def normalize(voxel_mean, voxel_std, X):
    """Normalizes an array of features X given voxel-wise mean and std."""
    for i in tqdm(range(X.shape[0]), desc='Normalize    '):
        np.place(voxel_std, voxel_std <= 0, [1])
        X[i] = (X[i] - voxel_mean) / (voxel_std + 1e-7)

    return X


def example_to_tfrecords(X, y, writer):
    """Writes one (X, y) example to tf.TFRecord file."""
    example = {
        'X': tf.train.Feature(float_list=tf.train.FloatList(value=X.flatten())),
        'y': tf.train.Feature(float_list=tf.train.FloatList(value=y.flatten()))}

    example = tf.train.Features(feature=example)
    example = tf.train.Example(features=example)
    writer.write(example.SerializeToString())


def main(args):
    # Convert .nii.gz data files to a list Tensors.
    print('Convert data to Numpy arrays.')
    X_train, y_train, prepro_stats = create_dataset(args.brats_folder)

    # Create dataset splits (if necessary).
    if args.create_val:
        print('Create train / val splits.')
        (X_train, y_train), (X_val, y_val) = create_splits(X_train, y_train)
    else:
        X_train = np.stack(X_train, axis=0)
        y_train = np.stack(y_train, axis=0)

    # Normalize training and validation data.
    if args.norm == 'image':
        print('Apply image-wise normalization per channel.')
        voxel_mean, voxel_std = image_norm(X_train)
    elif args.norm == 'pixel':
        print('Apply pixel-wise normalization per channel.')
        voxel_mean, voxel_std = pixel_norm(X_train)
    elif args.norm == 'scale':
        print('Normalize all values to [0, 1] per channel.')
        voxel_mean, voxel_std = scale_norm(X_train)

    prepro_stats.update({'mean': voxel_mean, 'std': voxel_std})
    np.save(os.path.join(args.out_folder, '{}_mean_std.npy'.format(args.norm)), prepro_stats)

    # Save normalized training data.
    print('Save training set.')
    X_train = normalize(voxel_mean, voxel_std, X_train)
    writer = tf.io.TFRecordWriter(os.path.join(args.out_folder, 'train.{}_wise.tfrecords'.format(args.norm)))
    for X, y in tqdm(zip(X_train, y_train)):
        example_to_tfrecords(X, y, writer)

    # Save normalized validation data.
    if args.create_val:
        print('Save validation data.')
        X_val = normalize(voxel_mean, voxel_std, X_val)
        writer = tf.io.TFRecordWriter(os.path.join(args.out_folder, 'val.{}_wise.tfrecords'.format(args.norm)))
        for X, y in tqdm(zip(X_val, y_val)):
            example_to_tfrecords(X, y, writer)


if __name__ == '__main__':
    args = prepro_parser()
    print('Preprocess args: {}'.format(args))
    main(args)
