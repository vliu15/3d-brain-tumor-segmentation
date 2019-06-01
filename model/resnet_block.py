import tensorflow as tf
from model.layer_utils.group_norm import GroupNormalization
from model.layer_utils.squeeze_excitation import SqueezeExcitation


class ConvLayer(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size=3,
                 data_format='channels_last',
                 groups=8,
                 kernel_regularizer=tf.keras.regularizers.l2(l=1e-5),
                 kernel_initializer='he_normal',
                 **kwargs):
        """Initializes one convolutional layer.

            Each convolutional layer is comprised of a group normalization,
            followed by a ReLU activation and a 3D-convolution.

            Args:
                filters: int
                    The number of filters to use in the 3D convolutional
                    block. The output layer of this green block will have
                    this many number of channels.
                kernel_size: kernel_size: int, optional
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
                kernel_initializer: tf.keras.initializers callable, optional
                    Kernel initializer for convolutional operations.
        """
        super(ConvLayer, self).__init__()
        # Set up config for self.get_config() to serialize later.
        self.config = super(ConvLayer, self).get_config()
        self.config.update({'filters': filters,
                            'kernel_size': kernel_size,
                            'data_format': data_format,
                            'groups': groups,
                            'kernel_regularizer': tf.keras.regularizers.serialize(kernel_regularizer),
                            'kernel_initializer': kernel_initializer})

        self.conv3d = tf.keras.layers.Conv3D(
                            filters=filters,
                            kernel_size=kernel_size,
                            strides=1,
                            padding='same',
                            data_format=data_format,
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer)
        self.groupnorm = GroupNormalization(
                            groups=groups,
                            axis=-1 if data_format == 'channels_last' else 1,
                            **kwargs)
        self.relu = tf.keras.layers.Activation('relu')

    def call(self, inputs, training=False):
        """Returns the forward pass of the ConvLayer.

            { Conv3D -> GroupNorm -> ReLU }
        """
        inputs = self.conv3d(inputs)
        inputs = self.groupnorm(inputs, training=training)
        inputs = self.relu(inputs)
        return inputs

    def get_config(self):
        return self.config


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size=3,
                 data_format='channels_last',
                 groups=8,
                 reduction=2,
                 kernel_regularizer=tf.keras.regularizers.l2(l=1e-5),
                 kernel_initializer='he_normal',
                 use_se=False,
                 **kwargs):
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
                use_se: bool, optional
                    Whether to apply a squeeze-excitation layer to the residual.
        """
        super(ConvBlock, self).__init__()
        # Set up config for self.get_config() to serialize later.
        self.config = super(ConvBlock, self).get_config()
        self.config.update({'filters': filters,
                            'kernel_size': kernel_size,
                            'data_format': data_format,
                            'groups': groups,
                            'reduction': reduction,
                            'kernel_regularizer': tf.keras.regularizers.serialize(kernel_regularizer),
                            'kernel_initializer': kernel_initializer,
                            'use_se': use_se})

        self.conv3d_ptwise = tf.keras.layers.Conv3D(
                                filters=filters,
                                kernel_size=1,
                                strides=1,
                                padding='same',
                                data_format=data_format,
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer,
                                **kwargs)
        self.use_se = use_se
        if self.use_se:
            self.se_layer = SqueezeExcitation(
                                reduction=reduction,
                                data_format=data_format,
                                **kwargs)
            self.scale = tf.keras.layers.Multiply()
        self.conv_layer1 = ConvLayer(
                                filters=filters,
                                kernel_size=kernel_size,
                                groups=groups,
                                reduction=reduction,
                                data_format=data_format,
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer,
                                **kwargs)
        self.conv_layer2 = ConvLayer(
                                filters=filters,
                                kernel_size=kernel_size,
                                groups=groups,
                                reduction=reduction,
                                data_format=data_format,
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer,
                                **kwargs)
        self.residual = tf.keras.layers.Add()

    def call(self, inputs, training=False):
        """Returns the forward pass of the ConvBlock.

            { Conv3D_pointwise -> ConvLayer -> ConvLayer -> Residual }
        """
        if self.use_se:
            inputs = self.conv3d_ptwise(inputs)
            res = self.scale([self.se_layer(inputs, training=training), inputs])
        else:
            res = self.conv3d_ptwise(inputs)
        inputs = self.conv_layer1(inputs, training=training)
        inputs = self.conv_layer2(inputs, training=training)
        inputs = self.residual([res, inputs])
        return inputs

    def get_config(self):
        return self.config
