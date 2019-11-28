# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:24:40 2019

@author: Affinity
"""

import numpy as np

p=np.random.random((32,100,128))


p

from tensorflow.contrib.rnn import GRUCell
import tensorflow as tf



init = tf.global_variables_initializer()

a=GRUCell(128)
b=GRUCell(128)

with tf.Session() as sess:
    sess.run(init)
    
    outputs, states = sess.run(tf.nn.bidirectional_dynamic_rnn(a, b,inputs=p, dtype=tf.float32))