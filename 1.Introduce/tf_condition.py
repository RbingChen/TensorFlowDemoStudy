# coding:utf-8

import tensorflow.compat.v1 as tf
import numpy as np

"""
  tf cast 
  tf where
  tf cond
"""

"""
  tf.where(
    condition, x=None, y=None, name=None
  )
  比较难用。
  1. 当condition 为1-d时，此时维度长度要与 x的 axis=0 的维度长度一致。
  2. 别搞 BroadCast。condition维度直接tile成和x、y维度一样。
"""
a = tf.where([True, False, False])
n = tf.constant([2]*24,shape=[2,3,4])
m = tf.constant([3]*12,shape=[2,3,4])

b1 = tf.where([True,False],n,m)

cond1 = tf.constant([True,False])  #第一维度
cond2 = tf.constant([[True,False,True,False],[True,False,False,False],[True,False,True,False]])
cond2 = tf.tile(tf.expand_dims(cond2,axis=0),[2,1,1]) #全选
# 3 、 4 不可行
cond3 = tf.constant([[True,False,True],[True,False,False]])
cond4 = tf.constant([[True,False,True,False]])
cond4 = tf.tile(tf.expand_dims(cond4,axis=0),[2,1,1])
b2 = tf.where(cond2,n,m)


with tf.Session() as sess:
    print(np.shape(sess.run(a)))
    print(sess.run(b1))
    print(sess.run(b2))
    print(np.shape(sess.run(cond2)))