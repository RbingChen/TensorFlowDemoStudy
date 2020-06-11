# coding:utf-8

import tensorflow.compat.v1 as tf
from sklearn import datasets
import numpy as np
import os



a=[1,2,3,4,5,6,7,8,9,10]
b=[ list(set(list(np.random.randint(1,10,np.random.randint(3,7))))) for i in range(len(a)) ]

c=[1,2,3,4,5,6,7,8,9,10]
d=[ list(set(list(np.random.randint(1,10,np.random.randint(3,7))))) for i in range(len(c)) ]
target=[np.random.randint(0,1) for i in range(len(c))]

print(a)
print(b)

emb_len = 3
emb_a = tf.Variable(shape=(len(a),emb_len),dtype=tf.float16)
emc_c = tf.Variable(shape=(len(a),emb_len),dtype=tf.float16)


# 2
dim1 = 2
conv1 = tf.Variable(shape=(len(dim1,emb_len)))

dim2 = 2

conv2_1 = tf.Variable(shape=(len(dim2,emb_len)))
conv2_2 = tf.Variable(shape=(len(dim2,emb_len)))

A = tf.placeholder(None, dtype=int)
B = tf.placeholder(None, dtype=int)
C = tf.placeholder(None, dtype=int)

input = [ b[i,np.random.randn(0,len(a),2)] for i in range(len(a))]










