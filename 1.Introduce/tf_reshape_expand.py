# coding:utf-8

import tensorflow.compat.v1 as tf
import numpy as np

a = tf.constant([[1,2,3,4,5,6]])
b= tf.reshape(a,(2,3))
e = tf.constant([1,2,3,4])
"""
  tf expand_dims 指定某个维度，增加维度。reshape 也可以实现。
"""
c = tf.expand_dims(a, axis=1)
d = tf.expand_dims(a, axis=2)
f = tf.expand_dims(e, axis=-1)
g = tf.expand_dims(e, axis =1 )
with tf.Session() as sess:
     print(sess.run(b))
     print(np.shape(sess.run(c)), np.shape(sess.run(d)))
     print(np.shape(sess.run(f)),np.shape(sess.run(g)))