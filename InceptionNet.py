import tensorflow as tf
import numpy as np
fc_layer = tf.contrib.layers.fully_connected
from InceptionUnit import *

class InceptionNet:
    def __init__(self, hidden_size, scope, is_training, keep_prob):
        self.hidden_size = hidden_size
        self.scope = scope     
        self.is_training = is_training
        self.keep_prob = keep_prob
        
    def __call__(self, inputs):
        with tf.variable_scope(self.scope):
            scope = self.scope
            #get query Features using CNN Inception Net
            inc_unit1 = InceptionUnit(scope+'1', self.hidden_size, self.is_training)
            en_outputs_inception1 = inc_unit1(inputs)
            # residual conncection
            en_outputs_inception1 = resnet_Add(inputs, en_outputs_inception1)
            
            inc_unit2 = InceptionUnit(scope+'2', self.hidden_size/2, self.is_training)
            en_outputs_inception2 = inc_unit2(en_outputs_inception1)
            # residual conncection
            en_outputs_inception2 =  resnet_Add(en_outputs_inception2, en_outputs_inception1)
            
            inc_unit3 = InceptionUnit(scope+'3', self.hidden_size, self.is_training)
            en_outputs_inception3 = inc_unit3(en_outputs_inception2)
            # residual conncection
            en_outputs_inception3 = resnet_Add(en_outputs_inception2, en_outputs_inception3)
            
            inc_unit4 = InceptionUnit(scope+'4', self.hidden_size*2, self.is_training)
            en_outputs_inception4 = inc_unit4(en_outputs_inception3)
            # residual conncection
            en_outputs_inception4 = resnet_Add(en_outputs_inception3, en_outputs_inception4)
            
            inc_unit5 = InceptionUnit(scope+'5', self.hidden_size, self.is_training)
            en_outputs_inception5 = inc_unit5(en_outputs_inception4)
            # residual conncection
            en_outputs_inception5 = resnet_Add(en_outputs_inception5, en_outputs_inception4)
            
            inception_all_query = tf.concat(([en_outputs_inception1, en_outputs_inception2, en_outputs_inception3, en_outputs_inception4, en_outputs_inception5]), 2)
            
            final_outputs_inception = fc_layer(inception_all_query, self.hidden_size)
            
            final_outputs_inception = batch_norm(final_outputs_inception, self.is_training, self.scope)      
            
            return final_outputs_inception
    