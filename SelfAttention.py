import tensorflow as tf
fc_layer = tf.contrib.layers.fully_connected

class SelfAttention(object):
    def __init__(self, inputs, hidden_size, scope):
    
        self.hidden_size = hidden_size
        self.context = inputs
        self.scope = scope

        
    def __call__(self, scaled_=True, masked_=False):   
        
        with tf.variable_scope(self.scope):
            gamma_enc = fc_layer(self.context, self.hidden_size, scope='lin_self_attn', activation_fn=tf.nn.selu) # Batch_size * Length * Hidden_size
            gamma_h = tf.transpose(gamma_enc, [0, 2, 1]) # Batch_size * Hidden_size * Length
            weights = tf.matmul(gamma_enc, gamma_h) # Batch_size * Length * Length
            weights = tf.nn.softmax(weights/tf.sqrt(float(self.hidden_size)))
            c_t = tf.matmul(weights, gamma_enc) # Batch_size * Length * Hidden_size
            output = fc_layer(tf.concat([gamma_enc, c_t], 2), self.hidden_size, scope='lin_out_self_attn', activation_fn=tf.nn.selu) + self.context # Batch_size * Length * Hidden_size
            return output

        