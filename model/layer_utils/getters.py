"""Contains getter functions for retrieving different options for layers."""
import tensorflow as tf


def get_normalization(normalization):
    from model.layer_utils.group_norm import GroupNormalization

    if normalization == 'group':
        return GroupNormalization
    elif normalization == 'batch':
        return tf.keras.layers.BatchNormalization


def get_downsampling(downsampling):
    from model.layer_utils.downsample import MaxDownsample, AvgDownsample, ConvDownsample

    if downsampling == 'max':
        return MaxDownsample
    elif downsampling == 'avg':
        return AvgDownsample
    else:
        return ConvDownsample


def get_upsampling(upsampling):
    from model.layer_utils.upsample import LinearUpsample, ConvUpsample

    if upsampling == 'linear':
        return LinearUpsample
    else:
        return ConvUpsample
