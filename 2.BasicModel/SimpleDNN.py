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
print(np.shape(data_X))
"""
   使用类来定义模型
"""
class SimpleDNN():
    def __init__(self,hidden_units,learning_rate=0.1):
        self.hidden_units = hidden_units
        assert len(hidden_units)>0
        self.learning_rate = learning_rate
        self.build()
    def build(self):
        self.layers = []
        for h in self.hidden_units:
            self.layers.append(tf.layers.Dense(h, activation='relu'))
        self.layers.append(tf.layers.Dense(1,activation='relu'))

    def __call__(self,inputs):
        assert len(self.layers) >0
        x = inputs
        for layers in self.layers:
            x = layers(x)
        return x



if __name__ == "__main__":



    with tf.name_scope("X"):
        X = tf.placeholder("float", shape=[None, dim], name="X")
    with tf.name_scope("Y"):
        Y = tf.placeholder("float", shape=None,name="Y")

    dnn = SimpleDNN(hidden_units=[3,2])
    pred_y = dnn(X)
    loss = tf.reduce_sum(tf.sqrt(tf.pow(pred_y-Y,2)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)


    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    t = sess.run(optimizer,feed_dict={X:train_X,Y:train_Y})
    loss_ = sess.run(loss,feed_dict={X:train_X,Y:train_Y})
    print(loss_)
    writer = tf.summary.FileWriter("./logs/",sess.graph)


