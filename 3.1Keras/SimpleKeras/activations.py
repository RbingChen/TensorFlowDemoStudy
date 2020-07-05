#coding:utf-8
import  tensorflow as tf
from  tensorflow import keras


def activation_layer(activation, prefix, is_training):
    if isinstance(activation, str):
        if activation == 'dice':
            return Dice(prefix, is_training)
        else:
            return tf.keras.layers.Activation(activation)
    else:
        raise ValueError(
            "Invalid activation,found %s.You should use a str or a Activation Layer Class." % activation)


class Dice(keras.layers.Layer):
    def __init__(self, prefix, is_training, **kwargs):
        self.prefix = prefix
        self.is_training = is_training
        super(Dice, self).__init__(**kwargs)

    def build(self, input_shape):
        self.alphas = tf.get_variable(name='dice_alpha_'+self.prefix, shape=input_shape[1:],
                                      initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        super(Dice, self).build(input_shape)

    def call(self, inputs):
        inputs_normed = tf.layers.batch_normalization(
            inputs=inputs,
            axis=-1,
            epsilon=0.000000001,
            center=False,
            scale=False,
            training=self.is_training)
        x_p = tf.sigmoid(inputs_normed)
        return self.alphas * (1.0 - x_p) * inputs + x_p * inputs