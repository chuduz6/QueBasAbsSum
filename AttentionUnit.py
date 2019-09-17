import tensorflow as tf
import pickle


class AttentionWrapper(object):
    def __init__(self, hidden_size, input_size, hs, scope_name):
        self.hs = tf.transpose(hs, [1,0,2])
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.scope_name = scope_name

        with tf.variable_scope(scope_name):
            self.Wh = tf.get_variable('Wh', [input_size, hidden_size])
            self.bh = tf.get_variable('bh', [hidden_size])
            self.Ws = tf.get_variable('Ws', [input_size, hidden_size])
            self.bs = tf.get_variable('bs', [hidden_size])
            self.Wo = tf.get_variable('Wo', [2*input_size, hidden_size])
            self.bo = tf.get_variable('bo', [hidden_size])
            self.Wh2 = tf.get_variable('Wh2', [input_size, hidden_size])
            self.bh2 = tf.get_variable('bh2', [hidden_size])
        

        hs2d = tf.reshape(self.hs, [-1, input_size])
        phi_hs2d = tf.tanh(tf.nn.xw_plus_b(hs2d, self.Wh, self.bh))
        self.phi_hs = tf.reshape(phi_hs2d, tf.shape(self.hs))

    def __call__(self, x, finished = None):
        gamma_h = tf.tanh(tf.nn.xw_plus_b(x, self.Ws, self.bs))
        '''
        weights = tf.reduce_sum(self.phi_hs * gamma_h, reduction_indices=2, keep_dims=True)
        context = tf.mod(tf.reduce_max(weights * 1000000 + self.hs, reduction_indices=0), 1000000)
        context = tf.stop_gradient(context)
        '''
        weights = tf.reduce_sum(self.phi_hs * gamma_h, reduction_indices=2, keep_dims=True)
        print ("WEIGHTS: ", weights)
        weights = tf.exp(weights - tf.reduce_max(weights, reduction_indices=0, keep_dims=True))
        print ("WEIGHTS: ", weights)

        weights = tf.divide(weights, (1e-6 + tf.reduce_sum(weights, reduction_indices=0, keep_dims=True)))
        print ("WEIGHTS: ", weights)

        context = tf.reduce_sum(self.hs * weights, reduction_indices=0)
        print ("CONTEXT: ", context)

        out = tf.tanh(tf.nn.xw_plus_b(tf.concat([context, x], -1), self.Wo, self.bo))
        print ("OUT ATTN UNIT: ", out)

        if finished is not None:
            out = tf.where(finished, tf.zeros_like(out), out)
        return out, tf.transpose(tf.squeeze(weights, [-1]), [1,0])
        
    def attention_all (self, inputs):
        inputs2 = tf.reshape(inputs, [-1, self.hidden_size])
        weights = tf.reduce_sum(self.phi_hs * inputs2, reduction_indices=2, keep_dims=True)
        weights = tf.exp(weights - tf.reduce_max(weights, reduction_indices=0, keep_dims=True))
        weights = tf.divide(weights, (1e-6 + tf.reduce_sum(weights, reduction_indices=0, keep_dims=True)))
        context = tf.reduce_sum(self.hs * weights, reduction_indices=0)
        out = tf.tanh(tf.nn.xw_plus_b(tf.concat([context, inputs2], -1), self.Wo, self.bo))
        out = tf.reshape(out, tf.shape(inputs))
        return out
        
    def bi_attention(self, inputs1, inputs2):
        #energy score = vt * tanh(W*inputs1 + W*inputs2 + W*(inputs1 . inputs2)
        print ("HELLO")
        
        '''
        bi_att_query2doc_obj = Attention(final_doc_encoder_outputs, final_query_encoder_outputs, self.hidden_size, "bi_attention_query2doc")
        context_vectors_query2doc = bi_att_query2doc_obj()
        
        #Bi-attention
        bi_att_doc2query_obj = Attention(final_query_encoder_outputs, final_doc_encoder_outputs, self.hidden_size, "bi_attention_doc2query")
        context_vectors_doc2query = bi_att_doc2query_obj()
        
        final_context_vectors = tf.concat([final_query_encoder_outputs, final_doc_encoder_outputs, final_query_encoder_outputs*final_doc_encoder_outputs, context_vectors_query2doc*context_vectors_query2doc], 1)
        final_context_vectors = tf.reduce_mean(final_context_vectors, reduction_indices=1, keep_dims=True)
        print ("final_context_vectors: ", final_context_vectors)     
        '''
        
        