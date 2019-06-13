"""Contains inference script for generating segmentation mask."""
import os
import glob
import tensorflow as tf
import numpy as np
import scipy
import nibabel as nib
from tqdm import tqdm
import random

from utils.arg_parser import test_parser
from utils.constants import *
from model.volumetric_cnn import VolumetricCNN


class Interpolator(object):
    """Depthwise interpolator for shorter images."""
    def __init__(self, data_format='channels_last'):
        self.data_format = data_format

    def __call__(self, image):
        def interpolate(slice1, slice2):
            step = (slice2 - slice1) / self._ratio
            return tf.stack([slice1 + i*step for i in range(self._ratio)], axis=d_axis)

        self.image_shape = image.shape
        d_axis = 2 if self.data_format == 'channels_last' else -1
        d = image.shape[d_axis]

        # Determine upscaling ratio.
        self._ratio = (D // (d-1)) + 1

        # Check if image is sufficiently sized.
        if self._ratio == 1:
            return image

        # Loop through and fill in slices.
        output = tf.concat([interpolate(image[:, :, :, i], image[:, :, :, i+1])
                                for i in range(d-1)], axis=d_axis)

        # Add on end slice.
        end = image[:, :, -1, :] if self.data_format == 'channels_last' else image[:, :, :, -1]
        output = tf.concat([output, tf.expand_dims(end, axis=d_axis)], axis=d_axis)

        return output

    def reverse(self, output):
        """Compresses interpolated output to original size."""
        if self._ratio == 1:
            return output

        d_axis = 2 if self.data_format == 'channels_last' else -1
        d = output.shape[d_axis]
        window = self._ratio // 2

        # Each depth is the average of the window of values around it.
        if self.data_format == 'channels_last':
            output = tf.stack([tf.reduce_mean(output[:, :, max(i-window, 0):min(i+window, d), :], axis=d_axis)
                        for i in range(0, d, self._ratio)], axis=d_axis)
        else:
            output = tf.stack([tf.reduce_mean(output[:, :, :, max(i-window, 0):min(i+window, d)], axis=d_axis)
                        for i in range(0, d, self._ratio)], axis=d_axis)

        return output


class Generator(object):
    def __init__(self, batch_size, stride=32,
                 data_format='channels_last'):
        self.batch_size = batch_size
        self.stride = stride
        self.data_format = data_format

    def __call__(self, image):
        """Generates crops with stride to create mask."""
        def _get_crop(h, w, d):
            if self.data_format == 'channels_last':
                return image[h-H:h, w-W:w, d-D:d, :]
            else:
                return image[:, h-H:h, w-W:w, d-D:d]

        iH, iW, iD = image.shape[:-1] if self.data_format == 'channels_last' else image.shape[1:]
        batch, idxs = [], []

        # Loop over all crops.
        for h in range(H, iH + self.stride - 1, self.stride):
            for w in range(W, iW + self.stride - 1, self.stride):
                for d in range(D, iD + self.stride - 1, self.stride):
                    h = min(h, iH)
                    w = min(w, iW)
                    d = min(d, iD)

                    batch.append(_get_crop(h, w, d))
                    idxs.append((h, w, d))

                    if len(batch) == self.batch_size:
                        yield (tf.stack(batch, axis=0), idxs)
                        batch, idxs = [], []

        if len(batch) > 0:
            yield (tf.stack(batch, axis=0), idxs)


class Segmentor(object):
    def __init__(self, model, threshold=0.5,
                 data_format='channels_last', **kwargs):
        self.model = model
        self.threshold = threshold
        self.data_format = data_format

    def __call__(self, image, generator):
        if self.data_format == 'channels_last':
            output = np.zeros(shape=list(image.shape[:-1]) + [OUT_CH], dtype=np.float32)
        else:
            output = np.zeros(shape=[OUT_CH] + list(image.shape[1:]), dtype=np.float32)

        for batch, idxs in generator(image):
            # Generate predictions.
            y, *_ = self.model(batch, training=False, inference=True)
            y = tf.clip_by_value(y, 1e-7, 1 - 1e-7)

            # Accumulate predicted probabilities.
            for p, (h, w, d) in zip(y.numpy(), idxs):
                if self.data_format == 'channels_last':
                    output[h-H:h, w-W:w, d-D:d, :] = np.add(output[h-H:h, w-W:w, d-D:d, :], p)
                else:
                    output[:, h-H:h, w-W:w, d-D:d] = np.add(output[:, h-H:h, w-W:w, d-D:d], p)

        # Normalize probabilities.
        # output /= tf.reduce_sum(output, axis=-1 if self.data_format == 'channels_last' else 0)

        return output

    def create_mask(self, output):
        axis = -1 if self.data_format == 'channels_last' else 0

        # Mask out values that correspond to values < threshold.
        mask = tf.reduce_max(output, axis=axis)
        mask = tf.cast(mask > self.threshold, tf.int32)

        # Take the argmax to determine label.
        output = tf.argmax(output, axis=axis, output_type=tf.int32)

        # Mask out values that correspond to <0.5 probability.
        output *= mask

        # Convert [0, 1, 2] to [1, 2, 3] for label consistency.
        output += 1

        # Replace label `3` with `4` for consistency with data.
        output = output.numpy()
        np.place(output, output >= 3, [4])

        return output


def get_npy_image(subject_folder, name):
    try:
        file_card = glob.glob(os.path.join(subject_folder, '*' + name + '.nii' + '*'))[0]
    except:
        file_card = glob.glob(os.path.join(subject_folder, '*' + name + '_proc.nii'))[0]
    return np.array(nib.load(file_card).dataobj).astype(np.float32)


def convert_to_nii(save_folder, mask):
    affine = np.array([[-0.977, 0., 0., 91.1309967],
                       [0., -0.977, 0., 149.34700012],
                       [0., 0., 1., -74.67500305],
                       [0., 0., 0., 1.]])
    img = nib.Nifti1Image(mask, affine)
    nib.save(img, os.path.join(save_folder, 'mask.nii'))
    

def dice_coefficient(y_true, y_pred, eps=1.0):
    """Returns dice coefficient of one prediction."""
    # Calculate mask for label 1.
    true_ones = tf.cast(tf.equal(y_true, 1), tf.float32)
    pred_ones = tf.cast(tf.equal(y_pred, 1), tf.float32)

    # Calculate mask for label 2.
    true_twos = tf.cast(tf.equal(y_true, 2), tf.float32)
    pred_twos = tf.cast(tf.equal(y_pred, 2), tf.float32)

    # Calculate mask for label 4.
    true_fours = tf.cast(tf.equal(y_true, 4), tf.float32)
    pred_fours = tf.cast(tf.equal(y_pred, 4), tf.float32)

    # Calculate mask for all positives.
    true_all = tf.cast(y_true > 0, tf.float32)
    pred_all = tf.cast(y_pred > 0, tf.float32)

    # Calculate per-label dice scores.
    dice_ones = (tf.reduce_sum(true_ones * pred_ones) + eps) / (tf.reduce_sum(true_ones + pred_ones) + eps)
    dice_twos = (tf.reduce_sum(true_twos * pred_twos) + eps) / (tf.reduce_sum(true_twos + pred_twos) + eps)
    dice_fours = (tf.reduce_sum(true_fours * pred_fours) + eps) / (tf.reduce_sum(true_fours + pred_fours) + eps)

    # Calculated macro- and micro- dice scores.
    print([dice_ones, dice_twos, dice_fours])
    dice_macro = tf.reduce_mean([dice_ones, dice_twos, dice_fours])
    dice_micro = (tf.reduce_sum(true_all * pred_all) + eps) / (tf.reduce_sum(true_all + pred_all) + eps)
    
    return dice_macro, dice_micro


def prepare_image(X, voxel_mean, voxel_std, data_format):
    """Normalize image and sample crops to represent whole input."""
    # Expand batch dimension and normalize.
    X = (X - voxel_mean) / voxel_std

    # Crop to size.
    if data_format == 'channels_last':
        crop1 = X[:H, :W, :D, :]
        crop2 = X[-H:, :W, :D, :]
        crop3 = X[:H, -W:, :D, :]
        crop4 = X[-H:, -W:, :D, :]
        crop5 = X[:H, :W, -D:, :]
        crop6 = X[-H:, :W, -D:, :]
        crop7 = X[:H, -W:, -D:, :]
        crop8 = X[-H:, -W:, -D:, :]
    elif data_format == 'channels_first':
        crop1 = X[:, :H, :W, :D]
        crop2 = X[:, -H:, :W, :D]
        crop3 = X[:, :H, -W:, :D]
        crop4 = X[:, -H:, -W:, :D]
        crop5 = X[:, :H, :W, -D:]
        crop6 = X[:, -H:, :W, -D:]
        crop7 = X[:, :H, -W:, -D:]
        crop8 = X[:, -H:, -W:, -D:]
        
    return np.stack([crop1, crop2, crop3, crop4, crop5, crop6, crop7, crop8], axis=0)


def create_mask(y, data_format):
    def merge_h(h1, h2):
        if data_format == 'channels_last':
            return tf.concat([h1[:-h_overlap, :, :, :],
                              tf.minimum(h1[-h_overlap:, :, :, :], h2[:h_overlap, :, :, :]),
                              h2[h_overlap:, :, :, :]], axis=0)
        elif data_format == 'channels_first':
            return tf.concat([h1[:, :-h_overlap, :, :],
                              tf.minimum(h1[:, -h_overlap:, :, :], h2[:, :h_overlap, :, :]),
                              h2[:, h_overlap:, :, :]], axis=1)

    def merge_w(w1, w2):
        if data_format == 'channels_last':
            return tf.concat([w1[:, :-w_overlap, :, :],
                              tf.minimum(w1[:, -w_overlap:, :, :], w2[:, :w_overlap, :, :]),
                              w2[:, w_overlap:, :, :]], axis=1)
        elif data_format == 'channels_first':
            return tf.concat([w1[:, :, :-w_overlap, :],
                              tf.minimum(w1[:, :, -w_overlap:, :], w2[:, :, :w_overlap, :]),
                              w2[:, :, w_overlap:, :]], axis=2)

    def merge_d(d1, d2):
        if data_format == 'channels_last':
            return tf.concat([d1[:, :, :-d_overlap, :],
                              tf.minimum(d1[:, :, -d_overlap:, :], d2[:, :, :d_overlap, :]),
                              d2[:, :, d_overlap:, :]], axis=2)
        elif data_format == 'channels_first':
            return tf.concat([d1[:, :, :, :-d_overlap],
                              tf.minimum(d1[:, :, :, -d_overlap:], d2[:, :, :, :d_overlap]),
                              d2[:, :, :, d_overlap:]], axis=-1)

    crop1, crop2, crop3, crop4, crop5, crop6, crop7, crop8 = [y[i, ...] for i in range(8)]

    # Calculate region of overlap.
    h_overlap = 2 * H - RAW_H
    w_overlap = 2 * W - RAW_W
    d_overlap = 2 * D - RAW_D

    h1 = merge_h(crop1, crop2)
    h2 = merge_h(crop3, crop4)
    h3 = merge_h(crop5, crop6)
    h4 = merge_h(crop7, crop8)

    w1 = merge_w(h1, h2)
    w2 = merge_w(h3, h4)

    d1 = merge_d(w1, w2)

    axis = -1 if data_format == 'channels_last' else 0

    # Mask out values that correspond to values < 0.5.
    mask = tf.reduce_max(d1, axis=axis)
    mask = tf.cast(mask > 0.5, tf.int32)

    # Take the argmax to determine label.
    seg_mask = tf.argmax(d1, axis=axis, output_type=tf.int32)

    # Convert [0, 1, 2] to [1, 2, 3] for consistency.
    seg_mask += 1

    # Mask out values that correspond to <0.5 probability.
    seg_mask *= mask

    return seg_mask


def main(args):
    # Initialize.
    data = np.load(args.prepro_file)
    voxel_mean = tf.convert_to_tensor(data.item()['mean'], dtype=tf.float32)
    voxel_std = tf.convert_to_tensor(data.item()['std'], dtype=tf.float32)
    
    # Initialize model.
    model = VolumetricCNN(
                        data_format=args.data_format,
                        kernel_size=args.conv_kernel_size,
                        groups=args.gn_groups,
                        reduction=args.se_reduction,
                        kernel_regularizer=tf.keras.regularizers.l2(l=args.l2_scale),
                        kernel_initializer=args.kernel_init,
                        downsampling=args.downsamp,
                        upsampling=args.upsamp,
                        normalization=args.norm)

    # Build model with initial forward pass.
    _ = model(tf.zeros(shape=(1, H, W, D, IN_CH) if args.data_format == 'channels_last' \
                                    else (1, IN_CH, H, W, D), dtype=tf.float32))
    # Load weights.
    model.load_weights(args.chkpt_file)

    # Initialize interpolator, generator, and segmentor.
    interpolator = Interpolator(data_format=args.data_format)
    generator = Generator(args.batch_size, stride=args.stride, data_format=args.data_format)
    segmentor = Segmentor(model, threshold=args.threshold, data_format=args.data_format)

    # Loop through each patient.
    for subject_folder in tqdm(glob.glob(os.path.join(args.test_folder, '*'))):

        with tf.device(args.device):
            axis = -1 if args.data_format == 'channels_last' else 0

            # Extract raw images from each.
            try:
                X = tf.stack(
                    [get_npy_image(subject_folder, name) for name in BRATS_MODALITIES], axis=axis)
            except:
                X = tf.stack(
                    [get_npy_image(subject_folder, name) for name in RTOG_MODALITIES], axis=axis)

            # If data is labeled, extract label.
            try:
                y = tf.expand_dims(get_npy_image(subject_folder, TRUTH), axis=axis)
            except:
                y = None

            # Normalize input image.
            X = (X - voxel_mean) / voxel_std

            # Interpolate depth (if necessary).
            X = interpolator(X)

            # Generate probabilities.
            y_pred = segmentor(X, generator)

            # Compress interpolation (if necessary).
            y_pred = interpolator.reverse(y_pred)

            # Create mask.
            y_pred = segmentor.create_mask(y_pred)

            # X = prepare_image(X, voxel_mean, voxel_std, args.data_format)
            # y_pred, *_ = model(X, training=False, inference=True)
            # y_pred = create_mask(y_pred, args.data_format).numpy()
            # np.place(y_pred, y_pred >= 3, [4])

            # If label is available, score the prediction.
            if y is not None:
                macro, micro = dice_coefficient(y, y_pred, eps=1.0)
                print('{}. Macro: {ma: 1.4f}. Micro: {mi:1.4f}'
                        .format(subject_folder.split('/')[-1], ma=macro, mi=micro), flush=True)

        # Save as Nifti.
        convert_to_nii(os.path.join(args.test_folder, subject_folder), y_pred)


if __name__ == '__main__':
    args = test_parser()
    print('Test args: {}'.format(args))
    main(args)
