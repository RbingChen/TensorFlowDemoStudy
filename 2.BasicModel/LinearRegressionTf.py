# coding:utf-8


import tensorflow.compat.v1 as tf

from sklearn import datasets
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
"""
  this code must be run in tensorflow v1.x
"""

data_X,data_Y = datasets.load_diabetes(return_X_y=True)
data_X = data_X[:, :1]
train_X = data_X[:-20]
test_X = data_X[-20:]

train_Y = data_Y[:-20]
test_Y = data_Y[-20:]

dim = np.shape(data_X)[1]

with tf.name_scope("X"):
    X = tf.placeholder("float", shape=[None, dim], name="X")
with tf.name_scope("Y"):
    Y = tf.placeholder("float", shape=None,name="Y")

with tf.name_scope("linear_regression"):
    #变量使用variable_scope
    with tf.variable_scope("w"):
        W = tf.Variable(tf.random_normal([1,dim]),name="weight")
    with tf.variable_scope("b"):
        b = tf.Variable(tf.random_normal([1]),name="bias")
    pred_Y = tf.matmul(W,tf.transpose(X))+b
with tf.name_scope("optimizer"):
    cost = tf.reduce_sum(tf.pow(pred_Y-Y,2))/400.0
    optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.9).minimize(cost)

tf.summary.scalar("lost",cost)
#初始化
init = tf.global_variables_initializer()
sess = tf.Session()
if True:
     sess.run(init)

     for epoch in range(1500):
         sess.run(optimizer,feed_dict={X:train_X,Y:train_Y})
         #sess.run(tp,feed_dict={X:test_X[1,:],Y:test_Y[1]})
         eval_cost = sess.run(cost,feed_dict={X:test_X,Y:test_Y})
         train_cost = sess.run(cost,feed_dict={X:train_X,Y:train_Y})
         #print("train_cost={:0.9f},eval_cost={:.9f},w={},b={}".format(train_cost,eval_cost,sess.run(W),sess.run(b)))

## 把图保存本地
writer = tf.summary.FileWriter("./logs/",sess.graph)

#306.72757499 , 153.24279071761313


