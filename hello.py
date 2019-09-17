#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-03-20 07:15:37
# @Author  : Anurag Roy (anu15roy@gmail.com)
# @Link    : ranarag.github.io
# @Version : 1.0.0

import os
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
def cosine_sim(x1, x2,name = 'Cosine_loss'):
    with tf.name_scope(name):
        x1_val = tf.sqrt(tf.reduce_sum(tf.matmul(x1,tf.transpose(x1)),axis=1))
        x2_val = tf.sqrt(tf.reduce_sum(tf.matmul(x2,tf.transpose(x2)),axis=1))
        denom =  tf.multiply(x1_val,x2_val)
        print (denom.shape)
        num = tf.reduce_sum(tf.multiply(x1,x2),axis=1)
        print (num.shape)
        return tf.div(num,denom)


X = tf.placeholder(tf.float32,[None,400])
Y = tf.placeholder(tf.float32,[None, 400])
sim = cosine_sim(X,Y)

#####################for Testing###################
sess = tf.Session()
sess.run(tf.global_variables_initializer())
def cmp(X_ = np.random.randn(2,400),Y_ = np.random.randn(2,400)):
    sk_sim = cosine_similarity(X_,Y_)[0][0]
    tf_sim = sess.run(sim,feed_dict={X:X_,Y:Y_})
    print ("TF SIM: ", tf_sim)
    tf_sim = np.mean(tf_sim)
    print ("TF SIM: ", tf_sim)
    print ("SK SIM: ", sk_sim)
    if abs(sk_sim - tf_sim) < 1e-3:
        print ("success")

cmp()
