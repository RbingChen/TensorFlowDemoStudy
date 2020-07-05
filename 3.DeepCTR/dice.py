# coding:utf-8

import tensorflow.compat.v1 as tf


def dice(input_var, axis=-1, epsilon=0.000000001, name='dice', training=True):
    alphas = tf.get_variable('alpha_'+name, input_var.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
    inputs_normed = tf.layers.batch_normalization(
        inputs=input_var,
        axis=axis,
        epsilon=epsilon,
        center=False,
        scale=False,
        training=training)
    p = tf.sigmoid(inputs_normed)
    return alphas * (1.0 - p) * input_var + p * input_var


def bn(input_var, axis=-1, epsilon=10e-9, name="bn"):

    x_mean = tf.reduce_mean(input_var)
    x_var = tf.reduce_sum(tf.pow(input_var-x_mean, 2))

    x = (input_var - x_mean)/tf.sqrt(x_var+epsilon)

    alpha = tf.get_variable('alpha_'+name, input_var.get_shape()[-1],
                                     initializer=tf.constant_initializer(0.0),
                                     dtype=tf.float32)
    beta = tf.get_variable('alpha_'+name, input_var.get_shape()[-1],
                           initializer=tf.constant_initializer(0.0),
                           dtype=tf.float32)
    y = alpha * x + beta
    return y






