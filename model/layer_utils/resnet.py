"""Contains convolutional sublayer and ResNet block classes."""
import tensorflow as tf

from model.layer_utils.group_norm import GroupNormalization


class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 data_format='channels_last',
                 groups=8,
                 reduction=2,
                 l2_scale=1e-5):
        """ Initializes one SENet block. Builds on basic ResNet block
            structure, but applies squeeze-and-excitation to the residual.

            References:
                - [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
        """
        super(ResnetBlock, self).__init__()
        # Set up config for self.get_config() to serialize later.
        self.config = super(ResnetBlock, self).get_config()
        self.config.update({'filters': filters,
                            'data_format': data_format,
                            'reduction': reduction,
                            'l2_scale': l2_scale,
                            'groups': groups})

        # Pointwise convolution.
        self.conv3d_ptwise = tf.keras.layers.Conv3D(
                                filters=filters,
                                kernel_size=1,
                                strides=1,
                                padding='same',
                                data_format=data_format,
                                kernel_regularizer=tf.keras.regularizers.l2(l=l2_scale),
                                kernel_initializer='he_normal')

        if filters % reduction != 0:
            raise ValueError(
                'Reduction ratio, {}, must be a factor of number of channels, {}.'
                .format(reduction, filters))

        # Channel squeeze excitation layers.
        self.squeeze = tf.keras.layers.GlobalAveragePooling3D(
                                data_format=data_format)
        self.dense_relu = tf.keras.layers.Dense(
                                units=filters // reduction,
                                kernel_regularizer=tf.keras.regularizers.l2(l=l2_scale),
                                kernel_initializer='he_normal',
                                use_bias=False,
                                activation='relu')
        self.dense_sigmoid = tf.keras.layers.Dense(
                                units=filters,
                                kernel_regularizer=tf.keras.regularizers.l2(l=l2_scale),
                                kernel_initializer='glorot_normal',
                                use_bias=False,
                                activation='sigmoid')
        self.reshape = tf.keras.layers.Reshape(
                                (1, 1, 1, -1) if data_format == 'channels_last'
                                              else (-1, 1, 1, 1))

        # Spatial squeeze excitation layers.
        self.spatial = tf.keras.layers.Conv3D(
                                filters=1,
                                kernel_size=1,
                                strides=1,
                                padding='same',
                                data_format=data_format,
                                kernel_initializer='glorot_normal',
                                kernel_regularizer=tf.keras.regularizers.l2(l=l2_scale),
                                activation='sigmoid')

        self.scale = tf.keras.layers.Multiply()
        self.add = tf.keras.layers.Add()

        # Convolutional layers.
        self.convs = []
        self.convs.append([tf.keras.layers.Conv3D(
                                filters=filters,
                                kernel_size=3,
                                strides=1,
                                padding='same',
                                data_format=data_format,
                                kernel_regularizer=tf.keras.regularizers.l2(l=l2_scale),
                                kernel_initializer='he_normal'),
                           GroupNormalization(
                                groups=groups,
                                axis=-1 if data_format == 'channels_last' else 1,
                                beta_initializer='zeros',
                                gamma_initializer='ones'),
                           tf.keras.layers.Activation('relu')])
        self.convs.append([tf.keras.layers.Conv3D(
                                filters=filters,
                                kernel_size=3,
                                strides=1,
                                padding='same',
                                data_format=data_format,
                                kernel_regularizer=tf.keras.regularizers.l2(l=l2_scale),
                                kernel_initializer='he_normal'),
                           GroupNormalization(
                                groups=groups,
                                axis=-1 if data_format == 'channels_last' else 1,
                                beta_initializer='zeros',
                                gamma_initializer='zeros'),
                           tf.keras.layers.Activation('relu')])

        self.residual = tf.keras.layers.Add()


    def call(self, inputs, training=None):
        # Pointwise input convolution.
        res = self.conv3d_ptwise(inputs)

        # Channel squeeze & excitation.
        chse = self.squeeze(res)
        chse = self.dense_relu(chse)
        chse = self.dense_sigmoid(chse)
        chse = self.reshape(chse)

        # Spatial squeeze & excitation.
        spse = self.spatial(res)

        # Scale residual.
        res = self.scale([res, self.add([spse, chse])])

        # Convolutional layers.
        for conv, norm, relu in self.convs:
            inputs = conv(inputs)
            inputs = norm(inputs, training=training)
            inputs = relu(inputs)
        inputs = self.residual([res, inputs])
        return inputs

    def get_config(self):
        return self.config
