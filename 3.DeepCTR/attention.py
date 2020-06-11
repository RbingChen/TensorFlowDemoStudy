# coding:utf-8

import tensorflow.compat.v1 as tf



def attention(query,key,max_seq_len):
    """

    :param query: [B,N,H]
    :param key: [B,H]
    :param max_seq_len: int
    :return:
    """
    # 判断 dim 是否符合要求
    assert len(tf.shape(query))== 3
    assert len(tf.shape(key))== 2
    assert int(tf.shape(key)[-1])== int(tf.shape(query)[-1])
    query * key

