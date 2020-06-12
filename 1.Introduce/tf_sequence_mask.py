# coding:utf-8

import tensorflow.compat.v1 as tf
import numpy as np

"""
  探讨一下各种乘法
  sequence_mask(
    lengths,
    maxlen=None,
    dtype=tf.bool,
    name=None）
  dtype:可以指定是输出bool 还是 数值类型的 1、0
  maxlne:最大长度，注意lengths大于maxlen时，则lengths=maxlen
"""

key_mask = tf.sequence_mask(10,maxlen=5)

query_mask = tf.sequence_mask(2,maxlen=5,dtype=tf.float32)
q_m = tf.expand_dims(query_mask,axis=1)
q = tf.cast(q_m<0.5,dtype=tf.int16)

with tf.Session() as sess:
    print(np.shape(sess.run(key_mask)))
    print(np.shape(sess.run(query_mask)))
    print(sess.run(q))
