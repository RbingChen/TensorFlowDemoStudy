#coding-utf-8
import  tensorflow as tf
from  tensorflow import keras

def kernel_initializers(initializers):

    return tf.keras.initializers.Initializer(initializers)

def bias_initializers(initializers):

    return tf.keras.