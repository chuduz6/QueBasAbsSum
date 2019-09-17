import pickle
import tensorflow as tf
import io
import json
import os

fc_layer = tf.contrib.layers.fully_connected

def load_text_ms_marco(mode, data_dir):
    print("Loading ", mode, " document from {}.....".format(data_dir))
    with io.open(data_dir, 'r', encoding='ascii', errors='ignore') as input_file:
        data = json.loads(input_file.read())
        docs = []
        summaries = []
        queries = []
        if(mode == 'train'):
            count = 13400
        elif(mode == 'val'):
            count = 0
        elif(mode == 'test'):
            count = 14400
        else:
            raise NotImplementedError
            
        for i in range (count, count + len(data['passages'])):
            docs.append(' '.join([data['passages'][str(i)][j]['passage_text'] for j in range (len(data['passages'][str(i)]))]))
            summaries.append(''.join(data['answers'][str(i)]))
            queries.append(data['query'][str(i)])

    assert     len(docs) == len(queries)
    print ("Number of Data Loaded: ", len(docs))
    return docs, queries, summaries
    
def load_text(mode, data_dir):
    if (mode=='train'):
        cont = os.path.join(data_dir, "train_content")
        title = os.path.join(data_dir, "train_summary")
        query = os.path.join(data_dir, "train_query")
    elif (mode=='val'):
        cont = os.path.join(data_dir, "valid_content")
        title = os.path.join(data_dir, "valid_summary")
        query = os.path.join(data_dir, "valid_query")
    elif (mode=='test'):
        cont = os.path.join(data_dir, "test_content")
        title = os.path.join(data_dir, "test_summary")
        query = os.path.join(data_dir, "test_query")
    else:
        raise NotImplementedError
        
    print("Loading ", mode, " document from {}.....".format(data_dir))
    
    with io.open(cont, 'r', encoding='ascii', errors='ignore') as input_file:
        docs = input_file.readlines()
        
    with io.open(query, 'r', encoding='ascii', errors='ignore') as input_file:
        queries = input_file.readlines()
        
    with io.open(title, 'r', encoding='ascii', errors='ignore') as input_file:
        summaries = input_file.readlines()
        
    assert len(docs) == len(queries)
    assert len(docs) == len(summaries)
    
    print ("MODE: ", mode, " LENGTH OF DOCS: ", len(docs))

    return docs, queries, summaries


def two_linear_layer_net(inputs, size, scope, activation, reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope):
        inputs = fc_layer(inputs, size, scope=scope, activation_fn=activation, reuse=reuse)
        inputs = fc_layer(inputs, size, scope=scope, activation_fn=activation, reuse=reuse)
        return inputs

def one_linear_layer_net(inputs, size, scope, is_training, keep_prob, activation, reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope):
        inputs = fc_layer(inputs, size, scope=scope, activation_fn=activation, reuse=reuse)
        inputs = dropout (inputs, is_training, keep_prob)        
        return inputs
        
def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f) 

def store_pickle(data, file):
    with open(file, "wb") as f:
            pickle.dump(data, f)
            
def highway_layer(inputs, embedding_size, is_training=False, keep=1.0, scope=None):
    with tf.variable_scope(scope or "highway_layer"):
        d = embedding_size
        #print ("HIGHWAY LAYER: ", inputs)
        if is_training:
            inputs = tf.nn.dropout(inputs, keep)
        trans = fc_layer(inputs, d, scope='trans', activation_fn=None)
        trans = tf.nn.relu(trans)
        if is_training:
            inputs = tf.nn.dropout(inputs, keep)
        gate = fc_layer(inputs, d, scope='gate', activation_fn=None)
        gate = tf.nn.sigmoid(gate)
        out = gate * trans + (1 - gate) * inputs
        return out

        
# Resnet add
def resnet_Add(x1, x2):
    """
x1 shape[-1] is small x2 shape[-1]
    """
    if x1.get_shape().as_list()[2] != x2.get_shape().as_list()[2]:
        # Option A:zero-padding
        residual_connection = x2 + tf.pad(x1, [[0, 0], [0, 0],
                                               [0, x2.get_shape().as_list()[2] - x1.get_shape().as_list()[2]]])
    else:
        residual_connection = x2 + x1
        # residual_connection=tf.add(x1,x2)
    return residual_connection
    

def highway_network(inputs, num_layers, embedding_size, is_training=False, keep=1.0, scope=None):
    with tf.variable_scope(scope or "highway_network"):
        prev = inputs
        cur = None
        for layer_idx in range(num_layers):
            cur = highway_layer(prev, embedding_size, scope="layer_{}".format(layer_idx))
            prev = cur
        return cur
        
        
def dropout (inputs, is_training, keep_prob):
    if is_training is True and keep_prob < 1.0:
        inputs = tf.nn.dropout(inputs, keep_prob)
    return inputs
    
    
def batch_norm(inputs, is_training, scope):
    return tf.layers.batch_normalization(inputs, training=is_training, name=scope, reuse=tf.AUTO_REUSE)
    
            
def conv1d(inputs, kernel_size, channels, activation, is_training, keep_prob, scope):
        with tf.variable_scope(scope):
            inputs = dropout(inputs, is_training, keep_prob)
            conv1d_output = tf.layers.conv1d(inputs, filters=channels, kernel_size=kernel_size, activation=activation, padding='same')
            return conv1d_output


def conv1dF(in_, filter_size, height, padding, is_train=None, keep_prob=1.0, scope=None):
        with tf.variable_scope(scope or "conv1d"):
            num_channels = in_.get_shape()[-1]
            filter_ = tf.get_variable("filter", shape=[1, height, num_channels, filter_size], dtype='float')
            bias = tf.get_variable("bias", shape=[filter_size], dtype='float')
            strides = [1, 1, 1, 1]
            in_ = dropout (in_, is_train, keep_prob)
            xxc = tf.nn.conv2d(in_, filter_, strides, padding) + bias  
            out = tf.reduce_max(tf.nn.relu(xxc), 2)  
            return out            