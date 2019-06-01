"""Contains getter functions for retrieving different options for layers."""
import tensorflow as tf


def get_normalization(normalization):
    from model.layer_utils.group_norm import GroupNormalization
    from model.layer_utils.layer_norm import LayerNormalization

    if normalization == 'group':
        return GroupNormalization
    elif normalization == 'batch':
        return tf.keras.layers.BatchNormalization
    elif normalization == 'layer':
        return LayerNormalization


def get_downsampling(downsampling):
    from model.layer_utils.downsample import MaxDownsample, AvgDownsample, ConvDownsample

    if downsampling == 'max':
        return MaxDownsample
    elif downsampling == 'avg':
        return AvgDownsample
    elif downsampling == 'conv':
        return ConvDownsample


def get_upsampling(upsampling):
    from model.layer_utils.upsample import LinearUpsample, ConvUpsample

    if upsampling == 'linear':
        return LinearUpsample
    elif upsampling == 'conv':
        return ConvUpsample
