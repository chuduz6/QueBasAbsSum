import tensorflow as tf
import numpy as np
from Unit import *
fc_layer = tf.contrib.layers.fully_connected
from utils import *

class InceptionUnit(object):

    def __init__(self, type, hidden_size, is_training, drop_conv=0.9):
        self.type = type
        self.drop_conv = drop_conv
        self.hidden_size = hidden_size    
        self.is_training = is_training
        #self.linear_layer = OutputUnit(3*self.hidden_size, self.hidden_size, "inception_"+self.type)
        
    def __call__(self, inputs):
        is_training = self.is_training
        with tf.variable_scope(self.type):
            '''
            sw1 = conv1d(inputs, 1, self.hidden_size, tf.nn.relu, self.is_training, self.drop_conv, self.type+'1_1')
            #print ("SW1: ", sw1)
            
            sw13 = conv1d(inputs, 1, self.hidden_size, tf.nn.relu, self.is_training, self.drop_conv, self.type+'13_1')
            sw13 = conv1d(sw13, 3, self.hidden_size, tf.nn.relu, self.is_training, self.drop_conv, self.type+'13_2')
            #print ("SW13: ", sw13)

            sw133 = conv1d(inputs, 1, self.hidden_size, tf.nn.relu, self.is_training, self.drop_conv, self.type+'133_1')
            sw133 = conv1d(sw133, 3, self.hidden_size, tf.nn.relu, self.is_training, self.drop_conv, self.type+'133_2')
            sw133 = conv1d(sw133, 3, self.hidden_size, tf.nn.relu, self.is_training, self.drop_conv, self.type+'133_3')
            #print ("SW133: ", sw133)
            
            #result = tf.reshape(tf.concat([sw1, sw13, sw133], 2), [-1, 3*self.hidden_size])
            #result = tf.reshape(self.linear_layer(result), tf.shape(inputs))            
            result = fc_layer(tf.concat([sw1, sw13, sw133], 2), int(self.hidden_size))
            return result
            '''
            #inputs = tf.transpose(inputs, [0, 2, 1])
            sw1 = conv1d(inputs, 1, self.hidden_size, tf.nn.relu, self.is_training, self.drop_conv, self.type+'1_1')
            sw1= tf.layers.batch_normalization(sw1, training=is_training, name=self.type+'1_1', reuse=tf.AUTO_REUSE)
            
            sw13 = conv1d(inputs, 1, self.hidden_size, tf.nn.relu, self.is_training, self.drop_conv, self.type+'13_1')
            sw13= tf.layers.batch_normalization(sw13, training=is_training, name=self.type+'13_1', reuse=tf.AUTO_REUSE)
            sw13 = conv1d(sw13, 3, self.hidden_size, tf.nn.relu, self.is_training, self.drop_conv, self.type+'13_2')
            sw13= tf.layers.batch_normalization(sw13, training=is_training, name=self.type+'13_1', reuse=tf.AUTO_REUSE)

            sw133 = conv1d(inputs, 1, self.hidden_size, tf.nn.relu, self.is_training, self.drop_conv, self.type+'133_1')
            sw133= tf.layers.batch_normalization(sw133, training=is_training, name=self.type+'133_1', reuse=tf.AUTO_REUSE)
            sw133 = conv1d(sw133, 3, self.hidden_size, tf.nn.relu, self.is_training, self.drop_conv, self.type+'133_2')
            sw133= tf.layers.batch_normalization(sw133, training=is_training, name=self.type+'133_1', reuse=tf.AUTO_REUSE)
            sw133 = conv1d(sw133, 3, self.hidden_size, tf.nn.relu, self.is_training, self.drop_conv, self.type+'133_3')
            sw133= tf.layers.batch_normalization(sw133, training=is_training, name=self.type+'133_1', reuse=tf.AUTO_REUSE)
            #print ("SW133: ", sw133)           
                    
            conv = fc_layer(tf.concat([sw1, sw13, sw133], 2), self.hidden_size, scope='lin_incp', activation_fn=None)
            return conv