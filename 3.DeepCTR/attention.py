# coding:utf-8

import tensorflow.compat.v1 as tf
from tensorflow.keras  import layers
import numpy as np
"""
  1.https://github.com/zhougr1993/DeepInterestNetwork/blob/master/din/model.py
  2.https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/layers/dense_attention.py#L210-L340
  
"""
def attentionV0(query,key,max_seq_len):
    """
    :param query: [B,H]
    :param key: [B,N,H]
    :param max_seq_len: intV0
    :return:
     花式搞 reshape
    """
    # 判断 dim 是否符合要求
    assert int(key.shape[-1]) == int(query.shape[-1])
    # 该用哪种乘法得到 [B,N]
    seq_len = key.shape[1]
    out_dim = key.shape[-1]
    batch = key.shape[0]

    query = tf.reshape(query,[-1,out_dim,1]) #[B,H,1]
    qk = tf.matmul(key,query) # [B,N,1]

    mask = tf.sequence_mask(max_seq_len, maxlen=seq_len)

    t = tf.ones_like(qk, dtype=tf.float32)*(-10e8)

    mask = tf.tile(tf.reshape(mask,(-1,seq_len,1)), [batch, 1,1])
    mask = tf.Print(mask,["mask",tf.shape(mask)])
    #qk = tf.Print(qk,["qk1",qk])
    #qk = tf.Print(qk,["qk2",qk])
    qk = tf.where(mask, qk, t)
    #qk = tf.Print(qk,["qk2",qk])
    qk_sum = tf.reduce_sum(tf.exp(qk),axis=1,keep_dims=True)  #[B，1，1】

    score = tf.exp(qk)/qk_sum  #[B,N,1]
    score = tf.Print(score, ["score : ", score])

    score = tf.reshape(score, [-1, 1, seq_len])
    output = tf.matmul(score, key)
    return output,qk

def attentionV1(query,key,max_seq_len):
    """
    :param query: [B,H]
    :param key: [B,N,H]
    :param max_seq_len: int
    :return:
     花式搞 reshape
    """
    # 判断 dim 是否符合要求
    assert int(key.shape[-1]) == int(query.shape[-1])
    # 该用哪种乘法得到 [B,N]
    seq_len = key.shape[1]
    out_dim = key.shape[-1]
    batch = key.shape[0]

    query = tf.reshape(query,[-1,1,out_dim]) #[B,1,H]
    qk = tf.reduce_sum(key * query,axis=-1)# [B,N]

    mask = tf.sequence_mask(max_seq_len, maxlen=seq_len)

    t = tf.ones_like(qk, dtype=tf.float32)*(-10e8)
    mask = tf.tile(tf.reshape(mask,(-1,seq_len)), [batch, 1])
    #qk = tf.Print(qk,["qk2",qk])

    qk = tf.where(mask, qk, t)

    #qk = tf.Print(qk,["qk2",qk])
    qk_sum = tf.reduce_sum(tf.exp(qk),axis=-1,keepdims=True)
    score = tf.exp(qk)/qk_sum
    score = tf.Print(score, ["score : ", score])

    score = tf.reshape(score, [-1, 1, seq_len])  # [B,1,N]
    output = tf.matmul(score, key)  # [B,N,H]
    return output,qk

def attentionV2(query,key,max_seq_len):
    """

    :param query: [B,H]
    :param key: [B,N,H]
    :param max_seq_len: int
    :return:
      灵活使用 星号乘法
    """
    dim = query.shape[-1]
    seq_len = key.shape[1]

    query = tf.tile(query, [seq_len, 1])  # N,B,H
    query = tf.reshape(query, [-1, seq_len, dim])  # B,N,H
    key_mask = tf.sequence_mask(max_seq_len, maxlen=seq_len, dtype=tf.float32)

    output = key * query * tf.expand_dims(key_mask,axis=1)# B,N,H
    output = tf.reduce_sum(output, axis=-1)  # B,N
    t_cast = 10e9*tf.cast(key_mask < 0.5, dtype=tf.float32)

    output -= t_cast

    score = tf.nn.softmax(output)
    score = tf.expand_dims(score,axis=2)
    f = tf.reduce_sum(key * score,axis=1,keepdims=True)
    return f,output


def test_mask(max_seq_len, key_len):
     mask1 = tf.sequence_mask(max_seq_len, key_len,dtype=tf.float32)
     mask1 = tf.expand_dims(mask1, axis=1)
     mask1 = tf.Print(mask1, ["mask : ", mask1.shape])

     return mask1


sess = tf.Session()
q = tf.constant([2.0]*8,shape=[2,4],dtype=tf.float32)
k = tf.constant([3.0]*24,shape=[2,3,4],dtype=tf.float32)

init = tf.global_variables_initializer()
sess.run(init)
at_v0,qk_v0= attentionV0(q, k, 2)
#print(sess.run(q))
print(sess.run(qk_v0))
print(sess.run(at_v0))

at_v1,qk_v1= attentionV1(q, k, 2)
#print("v1 q :",sess.run(q))
print(sess.run(qk_v1))
print(sess.run(at_v1))

at_v2,qk_v2= attentionV2(q, k, 2)
#print("v2 q :",sess.run(q))
print(sess.run(qk_v2))
print(np.shape(sess.run(at_v2)))

sess.close()


