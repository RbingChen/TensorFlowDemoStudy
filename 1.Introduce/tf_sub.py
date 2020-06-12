# coding:utf-8

import tensorflow.compat.v1 as tf
import numpy as np

#e = tf.constant([2]*12,shape=[1,3,4])
e = tf.constant([2]*12,shape=[3,4])
f = tf.constant([3]*3,shape=[3,1])
h = tf.constant([3]*3,shape=[1,4])
g_1 = tf.constant([3]*3)
g_2 = tf.constant([3]*4)
e_f = e - f
#e_g = e - g_1 # 不行
e_g = e - g_2
e_h = e-h

with tf.Session() as sess:
    print(sess.run(e_f))
    print(sess.run(e_h))


