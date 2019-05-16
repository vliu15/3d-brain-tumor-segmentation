import os
import glob
import tensorflow as tf
import numpy as np

from utils.arg_parser import test_parser
from utils.constants import *
from model.volumetric_cnn import VolumetricCNN


def get_npy_image(subject_folder, name):
    """Returns np.array from .nii.gz files."""
    file_card = glob.glob(os.path.join(subject_folder, '*' + name + '_proc.nii'))[0]
    return np.array(nib.load(file_card).dataobj)


def prepare_image(X, voxel_mean, voxel_std, data_format):
    # Expand batch dimension and normalize.
    X = (tf.expand_dims(X, axis=0) - voxel_mean) / voxel_std

    # Crop to size.
    h = np.random.randint(H, RAW_H)
    w = np.random.randint(W, RAW_W)
    d = np.random.randint(D, RAW_D)

    if data_format == 'channels_last':
        X = X[h-H:h, w-W:w, d-D:d, :]
    elif data_format == 'channels_first':
        X = X[:, h-H:h, w-W:w, d-D:d]
        
    return X


def main(args):
    # Initialize.
    with np.load(args.prepro_file) as data:
        voxel_mean = data.item()['mean']
        voxel_std = data.item()['std']
        if args.data_format == 'channels_last':
            voxel_mean = np.reshape(voxel_mean, newshape=(1, 1, 1, 1, -1))
            voxel_std = np.reshape(voxel_std, newshape=(1, 1, 1, 1, -1))
        elif args.data_format == 'channels_first':
            voxel_mean = np.reshape(voxel_mean, newshape=(1, -1, 1, 1, 1))
            voxel_std = np.reshape(voxel_std, newshape=(1, -1, 1, 1, 1))
    
    model = VolumetricCNN(
                        data_format=args.data_format,
                        kernel_size=args.conv_kernel_size,
                        groups=args.gn_groups,
                        dropout=args.dropout,
                        kernel_regularizer=tf.keras.regularizers.l2(l=args.l2_scale))
    try:
        model.load_weights(args.chkpt_file)
    except:
        print('NEED TO INITIALIZE MODEL WITH IDENTICAL PARAMETERS AS CHKPT.')
        raise RuntimeError

    # Loop through each patient.
    for subject_folder in tqdm(glob.glob(os.path.join(args.test_folder, '*'))):

        # Extract raw images from each.
        if args.data_format == 'channels_last':
            X = np.stack(
                [get_npy_image(subject_folder, name) for name in RTOG_MODALITIES], axis=-1)
        elif args.data_format == 'channels_first':
            X = np.stack(
                [get_npy_image(subject_folder, name) for name in RTOG_MODALITIES], axis=0)

        # Prepare input image.
        X = prepare_image(X, voxel_mean, voxel_std, args.data_format)

        # Forward pass.
        with tf.device(args.device):
            y_pred, _, _, _ = model(X, training=False)


if __name__ == '__main__':
    args = test_parser()
    main(args)
