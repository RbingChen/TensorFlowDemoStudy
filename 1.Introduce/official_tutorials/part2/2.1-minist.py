# coding:utf-8

import tensorflow as tf
from tensorflow import keras
import numpy as np

(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()
x_train = tf.reshape(x_train,[-1, 28*28])
x_test = tf.reshape(x_test,[-1, 28*8])
dim = x_train.shape[-1]

w = tf.Variable(np.random.random((1,dim)), dtype=tf.float32, name="w")
b= tf.Variable(np.random.random(), dtype=tf.float32, name="bias")












