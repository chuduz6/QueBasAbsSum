import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
linear = tf.contrib.layers.fully_connected

class GruUnit(tf.contrib.rnn.RNNCell):
    def __init__(self, num_units, scope, input_size=None, activation=tanh):
        if input_size is not None:
          print("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._activation = activation
        self.scope = scope
    
    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units        

    def __call__(self, inputs, state, reuse=False):
        scope = self.scope
        with vs.variable_scope(scope, reuse=reuse):  # "GRUCell"
          with vs.variable_scope("Gates"):  # Reset gate and update gate.
            # We start with bias of 1.0 to not reset and not update.
            r = linear(tf.concat((inputs, state), 1), self._num_units, activation_fn=None)
            u = linear(tf.concat((inputs, state), 1), self._num_units, activation_fn=None)
            r, u = tf.cast(sigmoid(r), tf.float32), tf.cast(sigmoid(u), tf.float32)
            
          with vs.variable_scope("Candidate"):
            c = self._activation(linear(tf.concat((inputs, r * state), 1), self._num_units, activation_fn=None))
          new_h = u * state + (1 - u) * c
          
        return new_h, new_h