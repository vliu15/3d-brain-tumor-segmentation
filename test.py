"""Contains inference script for generating segmentation mask."""
import os
import glob
import random
import scipy
import numpy as np
import nibabel as nib
import tensorflow as tf
from tqdm import tqdm
from scipy.ndimage import zoom

from args import TestArgParser
from util import DiceCoefficient
from model import Model


class Interpolator(object):
    def __init__(self, data_format, order=3, mode='reflect'):
        self.data_format = data_format
        self.order = order
        self.mode = mode

    def __call__(self, path, modalities):
        """Extracts Numpy image and normalizes it to 1 mm^3."""
        # Extract raw images from each.
        image = []
        pixdim = []
        affine = []
        
        for name in modalities:
            file_card = glob.glob(os.path.join(path, '*' + name + '*' + '.nii' + '*'))[0]
            img = nib.load(file_card)

            array.append(np.array(img.dataobj).astype(np.float32))
            pixdim.append(img.header['pixdim'][:4])
            affine.append(np.stack([img.header['srow_x'],
                                    img.header['srow_y'],
                                    img.header['srow_z'],
                                    np.array([0., 0., 0., 1.])], axis=0))
            
            axis = -1 if self.data_format == 'channels_last' else 0

            # Prepare image.
            image = np.stack(image, axis=axis)
            self.pixdim = np.mean(pixdim, axis=0)
            self.affine = np.mean(affine, axis=0)

            # Scale voxel size.
            zoomdim = list(pixdim[:-1])
            zoomdim.insert(axis, 1.0)
            image = zoom(image, zoomdim, order=self.order, mode=self.mode)

            # Interpolate depth.
            depthzoom = np.ones_like(self.pixdim)
            depthzoom[axis] = self.pixdim[-1]
            image = zoom(image, list(depthzoom), order=self.order, mode=self.mode)

            return image

    def reverse(self, output):
        """Reverses the interpolation performed in __call__."""
        output = output.numpy()
        axis = -1 if self.data_format == 'channels_last' else 0

        # Reverse depth interpolation.
        depthzoom = np.ones_like(self.pixdim)
        depthzoom[axis] = 1.0 / self.pixdim[-1]
        output = zoom(output, list(depthzoom), order=self.order, mode=self.mode)

        # Reverse voxel scaling.
        zoomdim = list(1.0 / pixdim[:-1])
        output = zoom(output, zoomdim, order=self.order, mode=self.mode)

        return output
        

class Generator(object):
    def __init__(self, batch_size, data_format, stride=32):
        self.batch_size = batch_size
        self.data_format = data_format
        self.stride = stride

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
    def __init__(self, model, data_format, out_ch, threshold=0.5, **kwargs):
        self.model = model
        self.data_format = data_format
        self.threshold = threshold
        self.out_ch = out_ch

    def __call__(self, image, generator):
        if self.data_format == 'channels_last':
            output = np.zeros(shape=list(image.shape[:-1]) + [self.out_ch], dtype=np.float32)
        else:
            output = np.zeros(shape=[self.out_ch] + list(image.shape[1:]), dtype=np.float32)

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


def main(args):
    # Initialize.
    mean = tf.convert_to_tensor(args.prepro_stats['norm']['mean'])
    std = tf.convert_to_tensor(args.prepro_stats['norm']['std'])
    
    # Initialize model(s) and load weights.
    in_ch = len(args.modalities)
    tumor_model = Model(**args.tumor_model_args)
    _ = tumor_model(tf.zeros(shape=[1] + args.crop_size + [in_ch] if args.data_format == 'channels_last' \
                                    else [1, in_ch] + args.crop_size, dtype=tf.float32))
    tumor_model.load_weights(os.path.join(args.tumor_model, 'chkpt.hdf5'))

    if args.skull_model:
        skull_model = Model(**args.skull_model_args)
        _ = model(tf.zeros(shape=[1] + args.crop_size + [in_ch] if args.data_format == 'channels_last' \
                                    else [1, in_ch] + args.crop_size, dtype=tf.float32))
        skull_model.load_weights(os.path.join(args.skull_model, 'chkpt.hdf5'))

    # Initialize interpolator, generator, and segmentor.
    interpolator = Interpolator(order=args.order, mode=args.mode)
    skull_generator = Generator(args.skull_model_args['data_format'], args.batch_size, stride=args.stride)
    skull_segmentor = Segmentor(model, args.skull_model_args['data_format'], threshold=args.threshold)
    tumor_generator = Generator(args.tumor_model_args['data_format'], args.batch_size, stride=args.stride)
    tumor_segmentor = Segmentor(model, args.tumor_model_args['data_format'], threshold=args.threshold)

    # Loop through each patient.
    for subject_folder in tqdm(glob.glob(os.path.join(args.test_folder, '*'))):
        with tf.device(args.device):
            # Rescale and interpolate input image.
            X = interpolator(subject_folder)

            # If data is labeled, extract label.
            try:
                file_card = glob.glob(os.path.join(subject_folder, '*' + name + '*' + '.nii' + '*'))[0]
                y = np.array(nib.load(file_card).dataobj).astype(np.float32)
                y = tf.expand_dims(y, axis=axis)
            except:
                y = None

            # Normalize input image.
            X = (X - voxel_mean) / voxel_std

            # Generate probabilities.
            y_pred = segmentor(X, generator)

            # Compress interpolation (if necessary).
            y_pred = interpolator.reverse(y_pred)

            # Create mask.
            y_pred = segmentor.create_mask(y_pred)

            # If label is available, score the prediction.
            if y is not None:
                macro, micro = dice_coefficient(y, y_pred, eps=1.0)
                print('{}. Macro: {ma: 1.4f}. Micro: {mi:1.4f}'
                        .format(subject_folder.split('/')[-1], ma=macro, mi=micro), flush=True)

        # Save as Nifti.
        nib.save(nib.Nifti1Image(y_pred, affine),
                 os.path.join(args.test_folder, subject_folder, 'mask.nii'))


if __name__ == '__main__':
    args = test_parser()
    print('Test args: {}'.format(args))
    main(args)
