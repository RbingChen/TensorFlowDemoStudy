# coding:utf-8

import tensorflow.compat.v1 as tf

"""
  测试 tile用法
  https://blog.csdn.net/xwd18280820053/article/details/72867818
  tile(input,multiples,name=None)
  multiples：同一维度重复次数。维度和input一致。
  一般使用在，复制tensor，再进行reshape
"""

a = tf.constant([1, 2], name='a')
b = tf.tile(a,[3])

c = tf.constant([[3,4]],name='c')
d = tf.tile(c,[2,3])

with tf.Session() as sess:
    print(sess.run(b))
    print(sess.run(d))