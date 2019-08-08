import os
import glob
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
    def __init__(self, modalities, order=3, mode='reflect'):
        self.modalities = modalities
        self.order = order
        self.mode = mode

    def __call__(self, path):
        """Extracts Numpy image and normalizes it to 1 mm^3."""
        # Extract raw images from each.
        image = []
        pixdim = []
        affine = []
        
        for name in self.modalities:
            file_card = glob.glob(os.path.join(path, '*' + name + '*' + '.nii' + '*'))[0]
            img = nib.load(file_card)

            image.append(np.array(img.dataobj).astype(np.float32))
            pixdim.append(img.header['pixdim'][:4])
            affine.append(np.stack([img.header['srow_x'],
                                    img.header['srow_y'],
                                    img.header['srow_z'],
                                    np.array([0., 0., 0., 1.])], axis=0))

        # Prepare image.
        image = np.stack(image, axis=-1)
        self.pixdim = np.mean(pixdim, axis=0, dtype=np.float32)
        self.affine = np.mean(affine, axis=0, dtype=np.float32)

        # Rescale and interpolate voxels spatially.
        if np.any(self.pixdim[:-1] != 1.0):
            image = zoom(image, self.pixdim[:-1] + [1.0], order=self.order, mode=self.mode)
        
        # Rescale and interpolate voxels depthwise (along time).
        if self.pixdim[-1] != 1.0:
            image = zoom(image, [1.0, 1.0, self.pixdim[-1], 1.0], order=self.order, mode=self.mode)

        # Mask out background voxels.
        mask = np.max(image, axis=-1, keepdims=True)
        mask = (mask > 0).astype(np.float32)

        return image, mask

    def reverse(self, output, path):
        """Reverses the interpolation performed in __call__."""
        # Scale back spatial voxel interpolation.
        if np.any(self.pixdim[:-1] != 1.0):
            output = zoom(output, 1.0 / self.pixdim[:-1], order=self.order, mode=self.mode)

        # Scale back depthwise voxel interpolation.
        if self.pixdim[-1] != 1.0:
            output = zoom(output, [1.0, 1.0, 1.0 / self.pixdim[-1]], order=self.order, mode=self.mode)

        # Save file.
        nib.save(nib.Nifti1Image(output, self.affine),
                    os.path.join(path, 'mask.nii'))

        return output


class TestTimeAugmentor(object):
    """Handles full inference on input with test-time augmentation."""
    def __init__(self,
                 mean,
                 std,
                 model,
                 model_data_format,
                 spatial_tta=True,
                 channel_tta=0,
                 threshold=0.5):
        self.mean = mean
        self.std = std
        self.model = model
        self.model_data_format = model_data_format
        self.channel_tta = channel_tta
        self.threshold = threshold

        self.channel_axis = -1 if self.model_data_format == 'channels_last' else 1
        self.spatial_axes = [1, 2, 3] if self.model_data_format == 'channels_last' else [2, 3, 4]

        if spatial_tta:
            self.augment_axes = [self.spatial_axes, []]
            for axis in self.spatial_axes:
                pairs = self.spatial_axes.copy()
                pairs.remove(axis)
                self.augment_axes.append([axis])
                self.augment_axes.append(pairs)
        else:
            self.augment_axes = [[]]

    def __call__(self, x, bmask):
        # Normalize and prepare input (assumes input data format of 'channels_last').
        x = (x - self.mean) / self.std

        # Transpose to channels_first data format if required by model.
        if self.model_data_format == 'channels_first':
            x = tf.transpose(x, (3, 0, 1, 2))
            bmask = tf.transpose(bmask, (3, 0, 1, 2))
        x = tf.expand_dims(x, axis=0)
        bmask = tf.expand_dims(bmask, axis=0)

        # Initialize list of inputs to feed model.
        y = []

        # Create shape for intensity shifting.
        shape = [1, 1, 1]
        shape.insert(self.channel_axis, x.shape[self.channel_axis])

        if self.channel_tta:
            _, var = tf.nn.moments(x, axes=self.spatial_axes, keepdims=True)
            std = tf.sqrt(var)

        # Apply spatial augmentation.
        for flip in self.augment_axes:

            # Run inference on spatially augmented input.
            aug = tf.reverse(x, axis=flip)

            aug, *_ = self.model(aug, training=False, inference=True)
            y.append(tf.reverse(aug, axis=flip))

            for _ in range(self.channel_tta):
                shift = tf.random.uniform(shape, -0.1, 0.1)
                scale = tf.random.uniform(shape, 0.9, 1.1)

                # Run inference on channel augmented input.
                aug = (aug + shift * std) * scale
                aug = self.model(aug, training=False, inference=True)
                aug = tf.reverse(aug, axis=flip)
                y.append(aug)

        # Aggregate outputs.
        y = tf.concat(y, axis=0)
        y = tf.reduce_mean(y, axis=0, keepdims=True)

        # Mask out zero-valued voxels.
        y *= bmask

        # Take the argmax to determine label.
        # y = tf.argmax(y, axis=self.channel_axis, output_type=tf.int32)

        # Transpose back to channels_last data format.
        y = tf.squeeze(y, axis=0)
        if self.model_data_format == 'channels_first':
            y = tf.transpose(y, (1, 2, 3, 0))

        return y


def pad_to_spatial_res(res, x, mask):
    # Assumes that x and mask are channels_last data format.
    res = tf.convert_to_tensor([res])
    shape = tf.convert_to_tensor(x.shape[:-1], dtype=tf.int32)
    shape = res - (shape % res)
    pad = [[0, shape[0]],
           [0, shape[1]],
           [0, shape[2]],
           [0, 0]]

    orig_shape = list(x.shape[:-1])
    x = tf.pad(x, pad, mode='CONSTANT', constant_values=0.0)
    mask = tf.pad(mask, pad, mode='CONSTANT', constant_values=0.0)

    return x, mask, orig_shape
    

def main(args):
    in_ch = len(args.modalities)

    # Initialize model(s) and load weights / preprocessing stats.
    tumor_model = Model(**args.tumor_model_args)
    tumor_crop_size = args.tumor_model_args['crop_size']
    _ = tumor_model(tf.zeros(shape=[1] + tumor_crop_size + [in_ch] if args.tumor_model_args['data_format'] == 'channels_last' \
                                    else [1, in_ch] + tumor_crop_size, dtype=tf.float32))
    tumor_model.load_weights(os.path.join(args.tumor_model, 'chkpt.hdf5'))
    tumor_mean = tf.convert_to_tensor(args.tumor_prepro['norm']['mean'], dtype=tf.float32)
    tumor_std = tf.convert_to_tensor(args.tumor_prepro['norm']['std'], dtype=tf.float32)

    if args.skull_strip:
        skull_model = Model(**args.skull_model_args)
        skull_crop_size = args.skull_model_args['crop_size']
        _ = model(tf.zeros(shape=[1] + skull_crop_size + [in_ch] if args.skull_model_args['data_format'] == 'channels_last' \
                                    else [1, in_ch] + skull_crop_size, dtype=tf.float32))
        skull_model.load_weights(os.path.join(args.skull_model, 'chkpt.hdf5'))
        skull_mean = tf.convert_to_tensor(args.skull_prepro['norm']['mean'], dtype=tf.float32)
        skull_std = tf.convert_to_tensor(args.skull_prepro['norm']['std'], dtype=tf.float32)

    # Initialize helper classes for inference and evaluation (optional).
    dice_fn = DiceCoefficient(data_format='channels_last')
    interpolator = Interpolator(args.modalities, order=args.order, mode=args.mode)
    tumor_ttaugmentor = TestTimeAugmentor(
                                tumor_mean,
                                tumor_std,
                                tumor_model,
                                args.tumor_model_args['data_format'],
                                spatial_tta=args.spatial_tta,
                                channel_tta=args.channel_tta,
                                threshold=args.threshold)
    if args.skull_strip:
        skull_ttaugmentor = TestTimeAugmentor(
                                    skull_mean,
                                    skull_std,
                                    skull_model,
                                    args.skull_model_args['data_format'],
                                    spatial_tta=args.spatial_tta,
                                    channel_tta=args.channel_tta,
                                    threshold=args.threshold)

    for loc in args.in_locs:
        for path in tqdm(glob.glob(os.path.join(loc, '*'))):
            with tf.device(args.device):
                # If data is labeled, extract label.
                try:
                    file_card = glob.glob(os.path.join(path, '*' + args.truth + '*' + '.nii' + '*'))[0]
                    y = np.array(nib.load(file_card).dataobj).astype(np.float32)
                    y = tf.expand_dims(y, axis=-1)
                except:
                    y = None

                # Rescale and interpolate input image.
                x, mask = interpolator(path)

                # Strip MRI brain of skull and eye sockets.
                if args.skull_strip:
                    x, pad_mask, pad = pad_to_spatial_res(          # Pad to spatial resolution
                                        args.skull_spatial_res,
                                        x,
                                        mask)
                    skull_mask = skull_ttaugmentor(x, pad_mask)     # Inference with test time augmentation.
                    skull_mask = 1.0 - skull_mask                   # Convert skull positives into negatives.
                    x *= skull_mask                                 # Mask out skull voxels.
                    x = tf.slice(x,                                 # Remove padding.
                                 [0, 0, 0, 0],
                                 pad + [-1])

                # Label brain tumor categories per voxel.
                x, pad_mask, pad = pad_to_spatial_res(              # Pad to spatial resolution.
                                        args.tumor_spatial_res,
                                        x,
                                        mask)
                tumor_mask = tumor_ttaugmentor(x, pad_mask)         # Inference with test time augmentation.
                tumor_mask = tf.slice(tumor_mask,                   # Remove padding.
                                      [0, 0, 0, 0],
                                      pad + [-1])
                tumor_mask += 1                                     # Convert [0,1,2] to [1,2,3] for label consistency.
                tumor_mask = tumor_mask.numpy()
                np.place(tumor_mask, tumor_mask >= 3, [4])          # Replace label `3` with `4` for label consistency.

                # Reverse interpolation and save as .nii.
                y_pred = interpolator.reverse(tumor_mask, path)
                
                # If label is available, score the prediction.
                if y is not None:
                    macro, micro = dice_fn(y, y_pred)
                    print('{}. Macro: {ma: 1.4f}. Micro: {mi: 1.4f}'
                            .format(path.split('/')[-1], ma=macro, mi=micro), flush=True)


if __name__ == '__main__':
    parser = TestArgParser()
    args = parser.parse_args()
    print('Test args: {}'.format(args))
    main(args)
