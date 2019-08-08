import glob
import os
import random
import numpy as np
import nibabel as nib
from tqdm import tqdm
import tensorflow as tf

from args import PreproArgParser


def get_npy_image(path, name):
    file_card = glob.glob(os.path.join(path, '*' + name + '.nii' + '*'))[0]
    return np.array(nib.load(file_card).dataobj).astype(np.float32)


def create_dataset(locs, modalities, truth):
    """Returns lists of data inputs and labels."""
    x = []
    y = []

    h_min, h_max = float('inf'), -float('inf')
    w_min, w_max = float('inf'), -float('inf')
    d_min, d_max = float('inf'), -float('inf')

    # Loop through each folder with `.nii.gz` files.
    for i, loc in enumerate(locs, 1):
        for path in tqdm(glob.glob(os.path.join(loc, '*')), desc='Load data ({}/{})'.format(i, len(locs))):

            # Stack modalities in channel dimension.
            x_ = np.stack(
                [get_npy_image(path, name) for name in modalities], axis=-1)

            # Replace label 4 with 3 for softmax labels at output time.
            y_ = np.expand_dims(get_npy_image(path, truth), axis=-1)
            np.place(y_, y_ >= 4, [3])

            # Determine smallest bounding box.
            h_min_, h_max_ = np.where(np.any(x_, axis=(1, 2, -1)))[0][[0, -1]]
            h_min = min(h_min, h_min_)
            h_max = max(h_max, h_max_)

            w_min_, w_max_ = np.where(np.any(x_, axis=(0, 2, -1)))[0][[0, -1]]
            w_min = min(w_min, w_min_)
            w_max = max(w_max, w_max_)

            d_min_, d_max_ = np.where(np.any(x_, axis=(0, 1, -1)))[0][[0, -1]]
            d_min = min(d_min, d_min_)
            d_max = max(d_max, d_max_)

            x.append(x_)
            y.append(y_)

    # Crop to minimal size for all examples.
    x = [x_[h_min:h_max, w_min:w_max, d_min:d_max, :] for x_ in x]
    y = [y_[h_min:h_max, w_min:w_max, d_min:d_max, :] for y_ in y]

    print('Maximal crop size: [{}, {}, {}, {}]'.format(h_max-h_min, w_max-w_min, d_max-d_min, len(modalities)))

    size = {'h': h_max - h_min,
            'w': w_max - w_min,
            'd': d_max - d_min,
            'c': len(modalities)}

    return x, y, size


def compute_norm(x_train, in_ch):
    """Returns image-wise mean and standard deviation per channel."""
    mean = np.zeros((1, 1, 1, in_ch))
    std = np.zeros((1, 1, 1, in_ch))
    n = np.zeros((1, 1, 1, in_ch))

    # Compute mean.
    for x in tqdm(x_train, desc='Compute mean'):
        mean += np.sum(x, axis=(0, 1, 2), keepdims=True)
        n += np.sum(x > 0, axis=(0, 1, 2), keepdims=True)
    mean /= n

    # Compute std.
    for x in tqdm(x_train, desc='Compute std'):
        std += np.sum((x - mean) ** 2, axis=(0, 1, 2), keepdims=True)
    std = (std / n) ** 0.5

    return mean, std


def example_to_tfrecords(x, y, writer):
    """Writes one (x, y) example to tf.TFRecord file."""
    example = {
        'x': tf.train.Feature(float_list=tf.train.FloatList(value=x.flatten())),
        'y': tf.train.Feature(float_list=tf.train.FloatList(value=y.flatten()))}

    example = tf.train.Features(feature=example)
    example = tf.train.Example(features=example)
    writer.write(example.SerializeToString())


def main(args):
    # Load and convert data to Numpy.
    x_train, y_train, size = create_dataset(args.in_locs, args.modalities, args.truth)

    # Create train/val split.
    if args.create_val:
        indices = list(range(len(x_train)))
        random.shuffle(indices)
        split = len(indices) // 11
        x_val = x_train[:split]
        y_val = y_train[:split]
        print('{} validation examples.'.format(len(x_val)))

        x_train = x_train[split:]
        y_train = y_train[split:]
    print('{} training examples.'.format(len(x_train)))

    # Compute and save normalization stats.
    mean, std = compute_norm(x_train, len(args.modalities))
    np.save(os.path.join(args.out_loc, 'prepro.npy'),
            {'size': size, 'norm': {'mean': mean, 'std': std}})

    # Normalize and save training data.
    for i, (x, y) in tqdm(enumerate(list(zip(x_train, y_train)), 1), desc='Normalize and save training data'):
        writer = tf.io.TFRecordWriter(os.path.join(args.train_loc, '{}.tfrecord'.format(i)))
        example_to_tfrecords((x - mean) / std, y, writer)

    # Save normalized validation data.
    if args.create_val:
        for i, (x, y) in tqdm(enumerate(list(zip(x_val, y_val)), 1), desc='Normalize and save validation data'):
            writer = tf.io.TFRecordWriter(os.path.join(args.val_loc, '{}.tfrecord'.format(i)))
            example_to_tfrecords((x - mean) / std, y, writer)


if __name__ == '__main__':
    parser = PreproArgParser()
    args = parser.parse_args()
    print('Preprocess args: {}'.format(args))
    main(args)
