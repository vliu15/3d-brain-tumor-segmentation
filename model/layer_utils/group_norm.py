"""
    Contains Keras group normalization class from
    https://github.com/titu1994/Keras-Group-Normalization/blob/master/group_norm.py
"""
import tensorflow as tf
from tensorflow.keras import initializers, constraints, regularizers


class GroupNormalization(tf.keras.layers.Layer):
    def __init__(self,
                 groups=8,
                 axis=-1,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        """[Group Normalization](https://arxiv.org/abs/1803.08494)

            Args:
                groups: int, optional
                    Number of groups for Group Normalization.
                axis: int, optional
                    The axis that should be normalized (typically the
                    features axis).
                center: bool, optional
                    Whether to add an offset parameter.
                scale: bool, optional
                    Whether to add a scale parameter.
                epsilon: float, optional
                    Epsilon for calculating variance.
                gamma_initializer: str / tf.keras.initializer, optional
                    Initializer for the gamma weight.
                beta_initializer: str / tf.keras.initializer, optional
                    Initializer for the beta weight.
                gamma_regularizer: str / tf.keras.regularizer, optional
                    Regularizer for the gamma weight.
                beta_regularizer: str / tf.keras.regularizer, optional
                    Regularizer for the beta weight.
                gamma_constraint: str / tf.keras.constraint, optional
                    Constraint for the gamma weight.
                beta_constraint: str / tf.keras.constraint, optional
                    Constraint for the beta weight.
        """
        super(GroupNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]

        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        if dim < self.groups:
            raise ValueError('Number of groups (' + str(self.groups) + ') cannot be '
                             'more than the number of channels (' +
                             str(dim) + ').')

        if dim % self.groups != 0:
            raise ValueError('Number of groups (' + str(self.groups) + ') must be a '
                             'multiple of the number of channels (' +
                             str(dim) + ').')

        self.input_spec = tf.keras.layers.InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None, **kwargs):
        input_shape = list(inputs.shape)

        # Prepare broadcasting shape.
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(1, self.groups)

        group_axes = [input_shape[i] for i in range(len(input_shape))]
        group_axes[self.axis] = input_shape[self.axis] // self.groups
        group_axes.insert(1, self.groups)

        # Reshape inputs to new group shape.
        group_shape = [group_axes[0], self.groups] + group_axes[2:]
        group_shape = tf.stack(group_shape)
        inputs = tf.reshape(inputs, group_shape)

        group_reduction_axes = list(range(len(group_axes)))
        group_reduction_axes = group_reduction_axes[2:]

        mean, variance = tf.nn.moments(inputs, axes=group_reduction_axes, keepdims=True)

        inputs = (inputs - mean) / (tf.math.sqrt(variance + self.epsilon))

        # Prepare broadcast shape.
        inputs = tf.reshape(inputs, group_shape)
        outputs = inputs

        # In this case we must explicitly broadcast all parameters.
        if self.scale:
            broadcast_gamma = tf.reshape(self.gamma, broadcast_shape)
            outputs = outputs * broadcast_gamma

        if self.center:
            broadcast_beta = tf.reshape(self.beta, broadcast_shape)
            outputs = outputs + broadcast_beta

        outputs = tf.reshape(outputs, input_shape)

        return outputs

    def get_config(self):
        config = {
            'groups': self.groups,
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(GroupNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
