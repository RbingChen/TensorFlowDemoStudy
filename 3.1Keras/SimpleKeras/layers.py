#coding:utf-8

import tensorflow as tf
from tensorflow import keras
from activations import activation_layer

"""
  
"""
class DNN(keras.layers.Layer):

    def __init__(self,
                 hidden_units,
                 name,
                 activation='relu',
                 kernel_initializer='glorot_uniform',
                 bias_initizlizer='zeros'
                 ):
        super(DNN,self).__init__()

        assert hidden_units is not None
        assert name is not None

        self.hidden_units = hidden_units
        self.activation = activation_layer.get(activation)
        self.kernel_initializer=kernel_initializer
        self.bias_initizlizer=bias_initizlizer

    def _get_fc_name(self, layer, unit):
        return 'fc{0}_{1}'.format(layer, unit)

    def build(self, input_shape):
        input_size = input_shape[-1]
        shape_units = [int(input_size)] + list(self.hidden_units)

        with tf.variable_scope(self.name):
           self.w = []
           self.b = []
           self.activation_layer = []

           for i in range(len(self.hidden_units)):
               with tf.variable_scope(self._get_fc_name(i, self.hidden_units[i])):
                    w = self.add_weight(name='kernel'+i,
                                        shape=[shape_units[i], shape_units[i+1]],
                                        initializer=self.kernel_initializer)
                    b = self.add_weight(name='bias'+i,
                                        shape=[shape_units[i+1]],
                                        initizlizer=self.bias_initizlizer)

                    a_l = activation_layer.get(self.activation)
                    self.w.append(w)
                    self.b.append(b)
                    self.activation_layer.append(a_l)

    def call(self, inputs):

        x = inputs
        for i in range(len(self.hidden_units)):
            x = self.activation_layer[i](tf.matmul(inputs, self.w[i])+self.b[i])
        return x
