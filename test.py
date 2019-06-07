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


class Generator(object):
    def __init__(self, batch_size, stride=32, data_format='channels_last'):
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

        if self.data_format == 'channels_last':
            iH, iW, iD, _ = image.shape
        else:
            _, iH, iW, iD = image.shape

        self._frequency = np.zeros(shape=(iH, iW, iD), dtype=np.float32)

        batch = []
        idxs = []

        # Loop over all crops.
        for h in range(H, iH + self.stride - 1, self.stride):
            for w in range(W, iW + self.stride - 1, self.stride):
                for d in range(D, iD + self.stride - 1, self.stride):
                    h = min(h, iH)
                    w = min(w, iW)
                    d = min(d, iD)

                    # Add crop to batch.
                    crop = _get_crop(h, w, d)
                    batch.append(crop)

                    # Track indices to overlay crop later.
                    idxs.append((h, w, d))

                    # Update frequency.
                    self._frequency[h-H:h, w-W:w, d-D:d] += 1.0

                    if len(batch) == self.batch_size:
                        yield (np.stack(batch, axis=0), idxs)
                        batch = []
                        idxs = []

        if len(batch) > 0:
            yield (np.stack(batch, axis=0), idxs)

    @property
    def frequency(self):
        if self.data_format == 'channels_last':
            return np.expand_dims(self._frequency, axis=-1)
        else:
            return np.expand_dims(self._frequency, axis=0)


class Segmentor(object):
    def __init__(self, model, device,
                 threshold=0.5, data_format='channels_last', **kwargs):
        self.model = model
        self.device = device
        self.threshold = threshold
        self.data_format = data_format

    def __call__(self, image, generator):
        if self.data_format == 'channels_last':
            axis = -1
            output = np.zeros(shape=list(image.shape[:-1]) + [OUT_CH-1], dtype=np.float32)
        else:
            axis = 0
            output = np.zeros(shape=[OUT_CH-1] + list(image.shape[1:]), dtype=np.float32)

        for batch, idxs in generator(image):
            # Generate predictions.
            with tf.device(self.device):
                y, *_ = self.model(batch, training=False)

            # Accumulate predicted probabilities, average later.
            for p, (h, w, d) in zip(y.numpy(), idxs):
                if self.data_format == 'channels_last':
                    output[h-H:h, w-W:w, d-D:d, :] += p
                else:
                    output[:, h-H:h, w-W:w, d-D:d] += p

        output = tf.convert_to_tensor(output)

        # Average predicted probabilities.
        output /= generator.frequency

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
    """Returns np.array from .nii files."""
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


def main(args):
    # Initialize.
    data = np.load(args.prepro_file)
    voxel_mean = data.item()['mean'].astype(np.float32)
    voxel_std = data.item()['std'].astype(np.float32)
    
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
                                    else (1, IN_CH, H, W, D)))
    # Load weights.
    model.load_weights(args.chkpt_file)

    # Initialize generator and segmentor.
    generator = Generator(args.batch_size, stride=args.stride, data_format=args.data_format)
    segmentor = Segmentor(model, args.device, threshold=0.5, data_format=args.data_format)

    # Loop through each patient.
    for subject_folder in tqdm(glob.glob(os.path.join(args.test_folder, '*'))):

        # Extract raw images from each.
        if args.data_format == 'channels_last':
            X = np.stack(
                [get_npy_image(subject_folder, name) for name in BRATS_MODALITIES], axis=-1)
        elif args.data_format == 'channels_first':
            X = np.stack(
                [get_npy_image(subject_folder, name) for name in BRATS_MODALITIES], axis=0)

        # Prepare input image.
        X = (X - voxel_mean) / voxel_std

        # Generate segmentation.
        mask = segmentor(X, generator)

        # Save as Numpy.
        np.save(os.path.join(args.test_folder, subject_folder, 'mask.npy'), mask)

        # Save as Nifti.
        convert_to_nii(os.path.join(args.test_folder, subject_folder), mask)


if __name__ == '__main__':
    args = test_parser()
    print('Test args: {}'.format(args))
    main(args)
