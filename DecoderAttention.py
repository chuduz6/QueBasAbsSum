import tensorflow as tf
fc_layer = tf.contrib.layers.fully_connected
from utils import *

class DecoderAttention(object):
    def __init__(self, encoder_states, state_size, scope):
        self.context = encoder_states
        self.hidden_size = state_size
        self.scope = scope 

        
    def __call__(self, h):
        with tf.variable_scope(self.scope):
            gamma_h = tf.expand_dims(two_linear_layer_net(h, self.hidden_size, 'lin_in', tf.nn.selu), 2)   # batch * size * 1
            print ("GAMMA H: ", gamma_h)
            weights = tf.squeeze(tf.matmul(self.context, gamma_h), 2)   # batch * time
            print ("weights: ", weights)
            
            weights = tf.nn.softmax(weights/tf.sqrt(float(self.hidden_size)))   # batch * time
            print ("weights: ", weights)
            c_t = tf.squeeze(tf.matmul(tf.expand_dims(weights, 1), self.context), 1) # batch * size
            print ("c_t: ", c_t)
            output = fc_layer(tf.concat([c_t, h], 1), self.hidden_size, scope='lin_out', activation_fn=tf.nn.selu)
            print ("output: ", output)
            return output, weights