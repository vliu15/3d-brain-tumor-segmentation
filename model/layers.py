import tensorflow as tf
from model.group_norm import GroupNormalization


class ConvLayer(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size=3,
                 data_format='channels_last',
                 groups=8,
                 kernel_regularizer=None):
        """Initializes one convolutional layer.

            Each convolutional layer is comprised of a group normalization,
            followed by a ReLU activation and a 3D-convolution.

            Args:
                filters: int
                    The number of filters to use in the 3D convolutional
                    block. The output layer of this green block will have
                    this many number of channels.
                kernel_size: kernel_size: (int, int, int), optional
                    The size of all convolutional kernels. Defaults to 3,
                    as used in the paper.
                data_format: str, optional
                    The format of the input data. Must be either 'channels_last'
                    or 'channels_first'. Defaults to `channels_last` for CPU
                    development. 'channels_first is used in the paper.
                groups: int, optional
                    The size of each group for GroupNormalization. Defaults to
                    8, as used in the paper.
        """

        super(ConvLayer, self).__init__()
        self.groupnorm = GroupNormalization(
                            groups=groups,
                            axis=-1 if data_format == 'channels_last' else 1)
        self.relu = tf.keras.layers.Activation('relu')
        self.conv3d = tf.keras.layers.Conv3D(
                            filters=filters,
                            kernel_size=kernel_size,
                            strides=1,
                            padding='same',
                            data_format=data_format,
                            kernel_regularizer=kernel_regularizer)

    def call(self, x):
        """Returns the forward pass of the ConvLayer.

            { GroupNormalization -> ReLU -> Conv3D }
        """
        x = self.groupnorm(x)
        x = self.relu(x)
        x = self.conv3d(x)
        return x


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size=3,
                 data_format='channels_last',
                 groups=8,
                 kernel_regularizer=None):
        """Initializes one convolutional block.

            A convolutional block (green block in the paper) consists of a pointwise
            3D-convolution (for residual connection) followed by 2 convolutional
            layers and a residual connection.

            Args:
                filters: int
                    The number of filters to use in the 3D convolutional
                    block. The output layer of this green block will have
                    this many number of channels.
                kernel_size: kernel_size: (int, int, int), optional
                    The size of all convolutional kernels. Defaults to 3,
                    as used in the paper.
                data_format: str, optional
                    The format of the input data. Must be either 'channels_last'
                    or 'channels_first'. Defaults to `channels_last` for CPU
                    development. 'channels_first is used in the paper.
                groups: int, optional
                    The size of each group for GroupNormalization. Defaults to
                    8, as used in the paper.
        """
        super(ConvBlock, self).__init__()
        self.conv3d_ptwise = tf.keras.layers.Conv3D(
                                filters=filters,
                                kernel_size=1,
                                strides=1,
                                padding='same',
                                data_format=data_format,
                                kernel_regularizer=kernel_regularizer)
        self.conv_layer1 = ConvLayer(
                                filters=filters,
                                kernel_size=kernel_size,
                                data_format=data_format,
                                groups=groups,
                                kernel_regularizer=kernel_regularizer)
        self.conv_layer2 = ConvLayer(
                                filters=filters,
                                kernel_size=kernel_size,
                                data_format=data_format,
                                groups=groups,
                                kernel_regularizer=kernel_regularizer)
        self.residual = tf.keras.layers.Add()

    def call(self, x):
        """Returns the forward pass of the ConvBlock.

            { Conv3D_pointwise -> ConvLayer -> ConvLayer -> Residual }
        """
        res = self.conv3d_ptwise(x)
        x = self.conv_layer1(res)
        x = self.conv_layer2(x)
        x = self.residual([res, x])
        return x
