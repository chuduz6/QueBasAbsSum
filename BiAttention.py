import tensorflow as tf
from Unit import *

class BiAttention(object):
    def __init__(self, doc, query, hidden_size, scope):
        self.hidden_size = hidden_size
        self.X = doc
        self.keys = query
        self.linear_layer = OutputUnit(4*self.hidden_size, self.hidden_size, scope)
        self.scope = scope
        
    
    def __call__(self, scaled_=True, masked_=False):
        scope = self.scope
        with tf.variable_scope(scope):
            x= self.X
            keys = self.keys
           
            dist_matrix = tf.matmul(x, keys, transpose_b=True)  
            #print ("DIST MATRIX: ", dist_matrix)
            if scaled_:
                d_k = tf.cast(tf.shape(keys)[-1], dtype=tf.float32)
                dist_matrix = tf.divide(dist_matrix, tf.sqrt(d_k)) 

            if masked_:
                raise NotImplementedError
            query_probs = tf.nn.softmax(dist_matrix)
            select_query = tf.matmul(query_probs, keys)  # (batch, x_words, q_dim)
            #print ("select_query: ", select_query)
            
            context_dist = tf.reduce_max(dist_matrix, axis=2)  # (batch, x_word``s)
            context_probs = tf.nn.softmax(context_dist)  # (batch, x_words)
            select_context = tf.einsum("ai,aik->ak", context_probs, x)  # (batch, x_dim)
            select_context = tf.expand_dims(select_context, 1)
            #print ("select_context: ", select_context)
            result = tf.reshape(tf.concat([x, select_query, x * select_query, x * select_context], axis=2 ), [-1, 4*self.hidden_size])       
            result = tf.reshape(self.linear_layer(result), tf.shape(x))
            #print ("RESULT: ", result)
            return result
    