import tensorflow as tf
from model.group_norm import GroupNormalization


class SqueezeExcitationLayer(tf.keras.layers.Layer):
    def __init__(self,
                 reduction=4,
                 data_format='channels_last'):
        """Initializes one squeeze-excitation layer.
        
            See https://arxiv.org/pdf/1709.01507.pdf for details.
            This layer is meant to be applied to each residual at
            the beginning of each ConvBlock (which are ResNet blocks).

            Args:
                reduction: int, optional
                    Reduction ratio for excitation size.
                data_format: str, optional
                    The format of the input data. Must be either 'channels_last'
                    or 'channels_first'. Defaults to `channels_last` for CPU
                    development. 'channels_first is used in the paper.
        """
        super(SqueezeExcitationLayer, self).__init__()
        self.data_format = data_format
        self.reduction = reduction

        self.global_pool = tf.keras.layers.GlobalAveragePooling3D(
                                data_format=data_format)

        # Initialize reduction and excitation in self.build().

        self.scale = tf.keras.layers.Multiply()

    def build(input_shape):
        """Builds dense layers at call-time."""
        channels = input_shape[-1] self.data_format == 'channels_last' else input_shape[1]
        if channels % self.reduction != 0:
            raise ValueError(
                'Reduction ratio, {}, must be a factor of number of channels, {}.'
                .format(self.reduction, channels))

        self.dense_relu = tf.keras.layers.Dense(channels / self.reduction, activation='relu')
        self.dense_sigmoid = tf.keras.layers.Dense(channels, activation='relu')

    def call(x):
        """Returns the forward pass of one SqueezeExcitationLayer.
        
            { GlobalAveragePool -> Dense+ReLu -> Dense+Sigmoid -> Scale }
        """
        se = self.global_pool(x)
        se = self.dense_relu(x)
        se = self.dense_sigmoid(x)
        x = self.scale(se, x)
        return x


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
                kernel_regularizer: tf.keras.regularizer callable, optional
                    Kernel regularizer for convolutional operations.
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
                 reduction=4,
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
                reduction: int, optional
                    Reduction ratio for excitation size in squeeze-excitation layer.
                kernel_regularizer: tf.keras.regularizer callable, optional
                    Kernel regularizer for convolutional operations.
        """
        super(ConvBlock, self).__init__()
        self.conv3d_ptwise = tf.keras.layers.Conv3D(
                                filters=filters,
                                kernel_size=1,
                                strides=1,
                                padding='same',
                                data_format=data_format,
                                kernel_regularizer=kernel_regularizer)
        self.se_layer = SqueezeExcitationLayer(
                                reduction=reduction,
                                data_format=data_format)
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
        x = self.conv3d_ptwise(x)
        res = self.se_layer(x)
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.residual([res, x])
        return x
