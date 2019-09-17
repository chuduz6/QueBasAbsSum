import tensorflow as tf
import sys
linear = tf.contrib.layers.fully_connected

class DistractionLSTMCell_soft(tf.contrib.rnn.RNNCell):  
  def __init__(self, num_units, scope, forget_bias=1.0, input_size=None, activation=tf.tanh): 
    
    self.num_units = num_units
    self.activation = activation
    self.scope = scope
    self.forget_bias = forget_bias

  def __call__(self, inputs, state, reuse=tf.AUTO_REUSE):
    scope = self.scope
    with tf.variable_scope(scope): 

      # Parameters of gates are concatenated into one multiplytiply for efficiency.
      h, c = state
                  
      i = linear(tf.concat((inputs, h), 1), self.num_units, activation_fn=None, scope='lin_dist_lstm_i', reuse=tf.AUTO_REUSE)
      j = linear(tf.concat((inputs, h), 1), self.num_units, activation_fn=None, scope='lin_dist_lstm_j', reuse=tf.AUTO_REUSE)
      f = linear(tf.concat((inputs, h), 1), self.num_units, activation_fn=None, scope='lin_dist_lstm_f', reuse=tf.AUTO_REUSE)
      o = linear(tf.concat((inputs, h), 1), self.num_units, activation_fn=None, scope='lin_dist_lstm_o', reuse=tf.AUTO_REUSE)
      g = linear(tf.concat((inputs, h), 1), self.num_units, activation_fn=None, scope='lin_dist_lstm_g', reuse=tf.AUTO_REUSE)

      #concat = linear([inputs, h], 5 * self.num_units, activation_fn=None, scope='lin_dist_lstm')

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate, g= distract_gate
      #i, j, f, o, g = tf.split(concat, 5, 1)

      new_c = (c * tf.sigmoid(f + self.forget_bias) + tf.sigmoid(i) *
               self.activation(j))
      eps = 1e-13
      temp = tf.div(tf.reduce_sum(tf.multiply(c, new_c),1),tf.reduce_sum(tf.multiply(c,c),1) + eps)

      m = tf.transpose(tf.sigmoid(g))
      t1 = tf.multiply(m , temp)
      t1 = tf.transpose(t1) 
 
      distract_c = new_c  -  c * t1

      new_h = self.activation(distract_c) * tf.sigmoid(o)

      new_state = (new_h, new_c)
       
      return new_h, new_state
