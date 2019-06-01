"""
    Contains Keras implementation of layer normalization:
    https://github.com/CyberZHG/keras-layer-normalization/blob/master/keras_layer_normalization/layer_normalization.py
"""
import tensorflow as tf
from tensorflow.keras import initializers, constraints, regularizers


class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self,
                 center=True,
                 scale=True,
                 epsilon=1e-5,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 gamma_constraint=None,
                 beta_constraint=None,
                 **kwargs):
        """[Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)

            Args:
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
        super(LayerNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.center = center
        self.scale = scale
        self.epsilon = epsilon
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma, self.beta = None, None

    def build(self, input_shape):
        self.input_spec = keras.engine.InputSpec(shape=input_shape)
        shape = input_shape[-1:]
        if self.scale:
            self.gamma = self.add_weight(
                            shape=shape,
                            initializer=self.gamma_initializer,
                            regularizer=self.gamma_regularizer,
                            constraint=self.gamma_constraint,
                            name='gamma')

        if self.center:
            self.beta = self.add_weight(
                            shape=shape,
                            initializer=self.beta_initializer,
                            regularizer=self.beta_regularizer,
                            constraint=self.beta_constraint,
                            name='beta')

        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[0]

        mean, variance = tf.nn.moments(inputs, axes=reduction_axes, keepdims=True)
        std = tf.math.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_constraint': constraints.serialize(self.gamma_constraint),
            'beta_constraint': constraints.serialize(self.beta_constraint),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask