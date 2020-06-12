# coding:utf-8

import tensorflow.compat.v1 as tf
import numpy as np

"""
  探讨一下各种乘法
  tf.matmul : 矩阵乘
  tf.multiply ：元素乘法。不是点乘，点乘需要求和。
"""

a = tf.constant([[1,2],[3,4]])
b = tf.constant([[1,2],[3,4]])
c = tf.multiply(a,b)
d = a*b
"""
  1.表量和向量、矩阵进行元素乘法。
  2.两个维度一模一样的矩阵和向量进行元素乘法。
  3.不同维度进行相乘?
  a: Tensor of type float16, float32, float64, int32, complex64, complex128 and rank > 1.
  b: Tensor with same type and rank as a.
  tensor的rank取决于倒数两维度。
"""
e = tf.constant([2]*12,shape=[1,3,4])
f = tf.constant([3]*3,shape=[3,1]) #shape[1,3] 也不行
h = tf.constant([3]*4,shape=[1,4])
g = tf.constant([3]*3) #不能和 e相乘。rank不一样。
e_f = e*f
#e_g = tf.matmul(e,g)
e_h =e * h
"""
 矩阵乘法：在多维（三维、四维）矩阵的相乘中，需要最后两维满足匹配原则。
 问题:二维和三维能否矩阵乘法？可以的。
 比如 对于（2，2，4）来说，视为2个（2，4）矩阵。
     对于（2，2，2，4）来说，视为2*2个（2，4）矩阵。
 小结：最后两维度做矩阵乘法
"""
b_2d = tf.constant([2]*12,shape=[3,4])
c_3d = tf.constant([3]*24,shape=[2,4,3])
d_4d = tf.constant([4]*48,shape=[2,2,4,3])
b_c = tf.matmul(b_2d,c_3d)
b_d = tf.matmul(b_2d,d_4d)
with tf.Session() as sess:
    print(sess.run(c))
    print(sess.run(d))
    print(sess.run(e_f))
    print(sess.run(e_h))
    #print(sess.run(e_g))
    print(" matmul >>>>>>>>>>")
    print(sess.run(b_c))
    print(sess.run(b_d))
