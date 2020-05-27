#coding:utf-8

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

a = tf.constant(2)
b = tf.constant(3)
c = tf.add(a,b)

with tf.compat.v1.Session() as sess:
    print("a[0]={},a[1]={}".format(a.eval(),b.eval()))