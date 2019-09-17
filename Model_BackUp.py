import tensorflow as tf
from Unit import *
import pickle
from AttentionUnit import AttentionWrapper
from InceptionUnit import *
from SelfAttention import *
from BiAttention import *
from char_word_embd_loader import load_doc_word_embedding, load_char_embedding, load_sum_word_embedding
from sent_embd_loader import load_sentence_embeddings
from InceptionNet import *
fc_layer = tf.contrib.layers.fully_connected
from utils import *
from vocabulary import load_dictionaries, sen_map2tok
import time
from distraction_lstm import DistractionLSTMCell_soft
from DecoderAttention import *



class Model(object):
    def __init__(self, args, batch_size, scope_name, name, start_token=3, stop_token=2, max_length=10, weight=0.0001, is_training=True):
        self.batch_size = batch_size
        self.hidden_size = args.size
        self.emb_size = args.embsize
        self.char_embsize = args.char_embsize
        self.grad_clip = args.max_gradient
        self.start_token = start_token
        self.stop_token = stop_token
        self.max_length = max_length
        self.scope_name = scope_name
        self.name = name
        self.args = args
        self.weight = weight
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.doc_dict, self.char_dict = load_dictionaries()
        self.source_vocab_size = min(len(self.doc_dict[0]), args.doc_vocab_size)


        self.encoder_query_inputs = tf.placeholder(tf.int32, [None, None], name='encoder_query_inputs')
        self.encoder_query_c_inputs = tf.placeholder(tf.int32, [None, None, None], name='encoder_query_c_inputs')  
        self.encoder_query_len = tf.placeholder(tf.int32, [None], name='encoder_query_len')
        
        self.encoder_doc_inputs = tf.placeholder(tf.int32, [None, None], name='encoder_doc_inputs')
        self.encoder_doc_c_inputs = tf.placeholder(tf.int32, [None, None, None], name='encoder_doc_c_inputs')
        self.encoder_doc_len = tf.placeholder(tf.int32, [None], name='encoder_doc_len')
        self.enc_padding_mask = tf.placeholder(tf.float32, [None, None], name='enc_padding_mask')

        self.encoder_doc_inputs_ext = tf.placeholder(tf.int32, [None, None], name='encoder_doc_inputs_ext')
        self.max_docs_oovs = tf.placeholder(tf.int32, [], name='max_docs_oovs')

        self.decoder_inputs = tf.placeholder(tf.int32, [None, None], name='dec_inputs')
        self.decoder_outputs = tf.placeholder(tf.int32, [None, None], name='target_ouputs')
        self.pointer_switches = tf.placeholder(tf.int32, [None, None], name='pointer_switches')
        self.dec_padding_mask = tf.placeholder(tf.float32, [None, None], name='dec_padding_mask')
        self.decoder_len = tf.placeholder(tf.int32, [None], name='decoder_len')
        
        self.doc_seq_indices = tf.placeholder(tf.int32, [None, None, None], name='doc_seq_indices')
        self.query_seq_indices = tf.placeholder(tf.int32, [None, None, None], name='query_seq_indices')
        self.feed_previous = tf.placeholder(tf.bool, [], name='feed_previous')
        
        with tf.variable_scope('units'):
            
            self.encoder_query_lstm = LstmUnit(self.hidden_size, self.emb_size, 'encoder_query_lstm')
            self.encoder_doc_lstm = LstmUnit(self.hidden_size, self.emb_size, 'encoder_doc_lstm')
            self.decoder_lstm = LstmUnit(self.hidden_size, self.emb_size, 'decoder_lstm')
            self.decoder_output_unit = OutputUnit(self.hidden_size, self.source_vocab_size, 'decoder_output')
            self.gated_linear = OutputUnit(self.hidden_size+self.emb_size, self.hidden_size, 'gated_linear')
            self.gated_output = OutputUnit(self.hidden_size, 1, 'gated_output')

        with tf.device('/cpu:0'):
            with tf.variable_scope('embeddings'):
            
                self.char_embedding = tf.constant(load_char_embedding(), dtype=tf.float32)
                self.encoder_query_c_embed = tf.nn.embedding_lookup(self.char_embedding, self.encoder_query_c_inputs)                
                self.encoder_doc_c_embed = tf.nn.embedding_lookup(self.char_embedding, self.encoder_doc_c_inputs)
                
                self.word_embeddings_doc = tf.constant(load_doc_word_embedding(), dtype=tf.float32)
                self.encoder_query_embed = tf.nn.embedding_lookup(self.word_embeddings_doc, self.encoder_query_inputs)
                self.encoder_doc_embed = tf.nn.embedding_lookup(self.word_embeddings_doc, self.encoder_doc_inputs)                
                self.decoder_embed = tf.nn.embedding_lookup(self.word_embeddings_doc, self.decoder_inputs)
                
                #self.sentence_embeddings_doc, self.sentence_embeddings_query = load_sentence_embeddings()
                #self.sent_doc_embd_inputs = tf.nn.embedding_lookup(self.sentence_embeddings_doc, self.doc_seq_indices)
                #self.sent_query_embd_inputs = tf.nn.embedding_lookup(self.sentence_embeddings_doc, self.query_seq_indices)
        
        filter_size_conv = 5
        with tf.variable_scope("char_conv"):
            self.encoder_doc_c_embed = conv1dF(self.encoder_doc_c_embed, self.emb_size, 1, "SAME", is_train=is_training, keep_prob=args.doc_encoder_keep_prob, scope='c1doc1')       
            self.encoder_doc_c_embed = conv1d(self.encoder_doc_c_embed, filter_size_conv, self.emb_size, tf.nn.relu, is_training, args.doc_encoder_keep_prob, scope="c3doc1")            
            
            if args.share_cnn_weights:
                tf.get_variable_scope().reuse_variables()
                self.encoder_query_c_embed = conv1dF(self.encoder_query_c_embed, self.emb_size, 1, "SAME", is_train=is_training, keep_prob=args.query_encoder_keep_prob, scope='c1doc1')              
                self.encoder_query_c_embed = conv1d(self.encoder_query_c_embed, filter_size_conv, self.emb_size, tf.nn.relu, is_training, args.query_encoder_keep_prob, scope="c3doc1")
                
            else:
                self.encoder_query_c_embed = conv1dF(self.encoder_query_c_embed, self.emb_size, 1, "SAME", is_train=is_training, keep_prob=args.query_encoder_keep_prob, scope='c1query1')           
                self.encoder_query_c_embed = conv1d(self.encoder_query_c_embed, filter_size_conv, self.emb_size, tf.nn.relu, is_training, args.query_encoder_keep_prob, scope="3query1")
                
            self.encoder_doc_c_embed = batch_norm(self.encoder_doc_c_embed, is_training, 'doc_char_norm')
            self.encoder_query_c_embed = batch_norm(self.encoder_query_c_embed, is_training, 'query_char_norm')
        '''
        with tf.variable_scope("sent_conv"):
            self.sent_doc_embd_inputs = conv1dF(self.sent_doc_embd_inputs, self.emb_size, 1, "SAME", is_train=is_training, keep_prob=args.doc_encoder_keep_prob, scope='s1doc1')       
            self.sent_doc_embd_inputs = conv1d(self.sent_doc_embd_inputs, filter_size_conv, self.emb_size, tf.nn.relu, is_training, args.doc_encoder_keep_prob, scope="s3doc1")
            
            if args.share_cnn_weights:
                tf.get_variable_scope().reuse_variables()
                self.sent_query_embd_inputs = conv1dF(self.sent_query_embd_inputs, self.emb_size, 1, "SAME", is_train=is_training, keep_prob=args.query_encoder_keep_prob, scope='s1doc1')              
                self.sent_query_embd_inputs = conv1d(self.sent_query_embd_inputs, filter_size_conv, self.emb_size, tf.nn.relu, is_training, args.query_encoder_keep_prob, scope="s3doc1")
                
            else:
                self.sent_query_embd_inputs = conv1dF(self.sent_query_embd_inputs, self.emb_size, 1, "SAME", is_train=is_training, keep_prob=args.query_encoder_keep_prob, scope='s1query1')             
                self.sent_query_embd_inputs = conv1d(self.sent_query_embd_inputs, filter_size_conv, self.emb_size, tf.nn.relu, is_training, args.query_encoder_keep_prob, scope="s3query1")
                
            self.sent_doc_embd_inputs = batch_norm(self.sent_doc_embd_inputs, is_training, 'doc_sent_norm')
            self.sent_query_embd_inputs = batch_norm(self.sent_query_embd_inputs, is_training, 'query_sent_norm')
        '''
        with tf.variable_scope("input_concat"):

            self.encoder_query_embed = fc_layer(tf.concat([self.encoder_query_c_embed, self.encoder_query_embed], 2), self.emb_size, scope='c_w_query', activation_fn=None)        
            self.encoder_doc_embed = fc_layer(tf.concat([self.encoder_doc_c_embed, self.encoder_doc_embed], 2), self.emb_size, scope='c_w_doc', activation_fn=None)

            #self.num_sent_query = tf.shape(self.sent_query_embd_inputs)[1]
            #self.num_sent_doc = tf.shape(self.sent_doc_embd_inputs)[1]            
            
            #self.encoder_query_embed = tf.concat([self.sent_query_embd_inputs, self.encoder_query_embed], 1)
            #self.encoder_doc_embed = tf.concat([self.sent_doc_embd_inputs, self.encoder_doc_embed], 1)
            
            #self.encoder_query_embed = BiAttention(self.encoder_query_embed, self.sent_query_embd_inputs, self.emb_size, 's_aware_w_query').__call__()
            #self.encoder_doc_embed = BiAttention(self.encoder_doc_embed, self.sent_doc_embd_inputs, self.emb_size, 's_aware_w_doc').__call__()                

        
        with tf.variable_scope("highway_network"):
            self.encoder_doc_embed = highway_network(self.encoder_doc_embed, args.highway_num_layers, self.emb_size, is_training, args.doc_encoder_keep_prob)
            tf.get_variable_scope().reuse_variables()
            self.encoder_query_embed = highway_network(self.encoder_query_embed, args.highway_num_layers, self.emb_size, is_training, args.query_encoder_keep_prob)
        
        with tf.variable_scope("encoder_query"):
            #QUERY: encoding using RNN-LSTM. Simple encoding should do for query as there aren't many query words
            en_query_outputs, en_query_state = self.non_gated_query_encoder(self.encoder_query_embed, self.encoder_query_len)        
            
        with tf.variable_scope('encoder_doc'):
            #DOCUMENT: gated document encoder
            en_doc_outputs, en_doc_state = self.gated_doc_encoder(self.encoder_doc_embed, self.encoder_doc_len)            
            
            #DOCUMENT: Extracting features using Modified Inception Network
            #en_doc_outputs_inception = InceptionNet(self.hidden_size, "inception_net_doc", is_training, args.doc_encoder_keep_prob).__call__(en_doc_outputs)
            en_doc_outputs_inception = InceptionUnit("inception_net_doc", self.hidden_size, is_training, args.doc_encoder_keep_prob).__call__(en_doc_outputs)

            #DOCUMENT: Gating Mechanism to self-attended extracted features
            en_doc_outputs_inc_attended = SelfAttention(en_doc_outputs_inception, en_doc_outputs, self.hidden_size, "self_attention_doc").__call__()
            del en_doc_outputs_inception
            gate_doc = tf.sigmoid(en_doc_outputs_inc_attended)   
            del en_doc_outputs_inc_attended
            en_doc_outputs = gate_doc * en_doc_outputs
            del gate_doc
        
        with tf.variable_scope('Bi_Directional_Attention'):
            #Bi-Directional Memoryless Attention       
            self.query_aware_docs = BiAttention(en_doc_outputs, en_query_outputs, self.hidden_size, 'query_aware_doc').__call__()
            del en_doc_outputs, en_query_outputs
        
        with tf.variable_scope('decoder_initial'):
            print ("EN DOC STATE: ", en_doc_state)
            decoder_initial_state = fc_layer(tf.concat([en_query_state[1], en_doc_state[1]], 1), self.hidden_size, scope='linear_decoder_initial_state', activation_fn = None)
            decoder_initial_outputs = fc_layer(tf.concat([en_query_state[0], en_doc_state[0]], 1), self.hidden_size, scope='linear_decoder_initial_outputs', activation_fn = None)
            #self.decoder_initial = tf.contrib.rnn.LSTMStateTuple(decoder_initial_outputs, decoder_initial_state)
            self.decoder_initial = (decoder_initial_outputs, decoder_initial_state)
            print ("DECODER INITIAL: ", self.decoder_initial)
        
        with tf.variable_scope('distraction_cell'):
            self.distract_state = self.decoder_initial
            self.distraction_cell = DistractionLSTMCell_soft(self.hidden_size, 'distraction_cell')
            
        with tf.variable_scope('decoder_attention'):
            #self.att_layer_doc_dec = AttentionWrapper(self.hidden_size, self.hidden_size, self.query_aware_docs, "doc_encoder_decoder_attention")
            self.att_layer_doc_dec = DecoderAttention(self.query_aware_docs, self.enc_padding_mask, self.batch_size, self.hidden_size, "doc_encoder_decoder_attention")
        
        training_final_dists, self.training_final_distract_state, self.training_dec_final_state, self.training_attn_dists, self.training_p_ptrs = self.add_decoder_layer(self.distract_state, self.decoder_initial, is_training)

        self.training_logits = tf.identity(tf.transpose(training_final_dists, [1, 0, 2]), name='training_logits')
        self.predicted_ids = tf.identity(tf.arg_max(self.training_logits, 2, output_type=tf.int32), name='predicted_ids')
        print ("PREDICTED IDS: ", self.predicted_ids)
        
        if(args.mode=='decode'):
            #assert len(training_final_dists)==1 # final_dists is a singleton list containing shape (batch_size, extended_vsize)
            d_training_final_dists = tf.transpose(training_final_dists, [1, 0, 2])
            #d_training_final_dists = d_training_final_dists[0]
            topk_probs, self.topk_ids = tf.nn.top_k(d_training_final_dists, self.batch_size*2) # take the k largest probs. note batch_size=beam_size in decode mode
            self.topk_log_probs = tf.log(topk_probs)
        
        with tf.variable_scope('optimization'):
            self.training_loss = self.calc_training_loss(training_final_dists)
            self.pointer_loss = self.calc_pointer_loss(self.training_p_ptrs)
            
            self.overall_loss = 0.8*self.training_loss +  0.2*self.pointer_loss
                        
            self.training_accuracy = self.get_training_accuracy(tf.transpose(training_final_dists, [1, 0, 2]), self.decoder_outputs)


            tvars = tf.trainable_variables()
            optimizer = tf.train.AdamOptimizer(args.learning_rate)
            
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.overall_loss, tvars), self.grad_clip)
            self.updates = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)  
        
            tf.summary.scalar('overall_loss', self.overall_loss)
            tf.summary.scalar('training_loss', self.training_loss)
            tf.summary.scalar('training_accuracy', self.training_accuracy)

        self.saver = tf.train.Saver(max_to_keep=args.max_to_keep)
        self.summary_merge = tf.summary.merge_all()
        
        
    def cosine_sim(self, x1, x2,name = 'Cosine_loss'):
        with tf.name_scope(name):
            x1_val = tf.sqrt(tf.reduce_sum(tf.matmul(x1,tf.transpose(x1)),axis=1))
            x2_val = tf.sqrt(tf.reduce_sum(tf.matmul(x2,tf.transpose(x2)),axis=1))
            denom =  tf.multiply(x1_val,x2_val)
            num = tf.reduce_sum(tf.multiply(x1,x2),axis=1)
            return tf.div(num+1e-13,denom+1e-13)
        
    def calc_pointer_loss(self, training_p_ptrs, ):
        with tf.variable_scope('pointer_loss'):
            float_pointer_switches = tf.to_float(self.pointer_switches)
            training_p_ptrs = tf.squeeze(tf.transpose(training_p_ptrs, [1, 0, 2]), [2])
            pointer_probability_loss = (float_pointer_switches * -tf.log(training_p_ptrs + 1e-9) +  (1. - float_pointer_switches) * -tf.log(1. - training_p_ptrs + 1e-9))
            length_mask = tf.sign(tf.to_float(self.decoder_outputs))
            masked_pointer_loss = length_mask * pointer_probability_loss
            float_lengths = tf.to_float(self.decoder_len)
            pointer_loss_by_doc = tf.reduce_sum(masked_pointer_loss, axis=1) / float_lengths
            pointer_loss = tf.reduce_mean(pointer_loss_by_doc)
        return pointer_loss
    
    def add_decoder_layer (self, distract_state, en_doc_state, is_training):
        with tf.variable_scope('decoder'):
            training_dec_outputs, training_distract_state, training_dec_final_state, training_attn_dists, training_p_ptrs = self.decoder_layer_training(distract_state, en_doc_state, self.decoder_embed, self.decoder_len)
            
            training_vocab_dists = tf.nn.softmax(training_dec_outputs)
            training_final_dists = self.calc_final_dist(training_vocab_dists, training_attn_dists, training_p_ptrs)
            
        '''
        with tf.variable_scope('decoder', reuse=True):
            inference_dec_outputs, inference_dec_final_state, inference_attn_dists, inference_p_ptrs = self.decoder_layer_inference(distract_state, en_doc_state)       
            inference_vocab_dists = tf.nn.softmax(inference_dec_outputs)
            inference_final_dists = self.calc_final_dist(inference_vocab_dists, inference_attn_dists, inference_p_ptrs)            
        '''
        
        return (training_final_dists, training_distract_state, training_dec_final_state, training_attn_dists, training_p_ptrs)
        
    
    def get_training_accuracy(self, logits, targets):
        predictions = tf.cast(tf.argmax(logits, axis=2), tf.int32) 
        accuracy = tf.contrib.metrics.accuracy(predictions, targets)
        return accuracy
        
    def get_inference_accuracy(self, targets, logits):
        targets_shape = tf.shape(targets)[1]
        logits_shape = tf.shape(logits)[1]
        max_seq = tf.cond(targets_shape < logits_shape, lambda: logits_shape, lambda: targets_shape)
        print ("MAX SEQ: ", max_seq, " TARGET SHAPE: ", targets_shape, " LOGITS SHAPE: ", logits_shape)
        targets = tf.cond(max_seq - targets_shape > 0, lambda: tf.pad(targets, [(0,0),(0,max_seq - targets_shape)], 'constant', constant_values=2), lambda: targets)
        logits = tf.cond(max_seq - logits_shape > 0, lambda: tf.pad(logits,[(0,0),(0,max_seq - logits_shape)], 'constant', constant_values=2), lambda: logits) 
        print ("TARGETS: ", targets)
        print ("LOGITS: ", logits)
        accuracy = tf.contrib.metrics.accuracy(logits, targets)
        return accuracy

    def non_gated_query_encoder(self, inputs, inputs_len):
        batch_size = tf.shape(self.encoder_query_inputs)[0]
        max_time = tf.shape(self.encoder_query_inputs)[1] #+ self.num_sent_query
        hidden_size = self.hidden_size

        time = tf.constant(0, dtype=tf.int32)
        h0 = (tf.zeros([batch_size, hidden_size], dtype=tf.float32),
              tf.zeros([batch_size, hidden_size], dtype=tf.float32))
        f0 = tf.zeros([batch_size], dtype=tf.bool)
        inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
        inputs_ta = inputs_ta.unstack(tf.transpose(inputs, [1,0,2]))
        emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)

        def loop_fn(t, x_t, s_t, emit_ta, finished):
            o_t, s_nt = self.encoder_query_lstm(x_t, s_t, finished)
            emit_ta = emit_ta.write(t, o_t)
            finished = tf.greater_equal(t+1, inputs_len)
            x_nt = tf.cond(tf.reduce_all(finished), lambda: tf.zeros([batch_size, self.emb_size], dtype=tf.float32),
                                     lambda: inputs_ta.read(t+1))
            return t+1, x_nt, s_nt, emit_ta, finished

        _, _, state, emit_ta, _ = tf.while_loop(
            cond=lambda _1, _2, _3, _4, finished: tf.logical_not(tf.reduce_all(finished)),
            body=loop_fn,
            loop_vars=(time, inputs_ta.read(0), h0, emit_ta, f0))

        outputs = tf.transpose(emit_ta.stack(), [1,0,2])
        return outputs, state


    def gated_doc_encoder(self, inputs, inputs_len):
        batch_size = tf.shape(self.encoder_doc_inputs)[0]
        max_time = tf.shape(self.encoder_doc_inputs)[1] #+ self.num_sent_doc
        hidden_size = self.hidden_size

        time = tf.constant(0, dtype=tf.int32)
        h0 = (tf.zeros([batch_size, hidden_size], dtype=tf.float32),
              tf.zeros([batch_size, hidden_size], dtype=tf.float32))
        f0 = tf.zeros([batch_size], dtype=tf.bool)
        inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
        inputs_ta = inputs_ta.unstack(tf.transpose(inputs, [1,0,2]))
        emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)

        def loop_fn(t, x_t, s_t, emit_ta, finished):
            o_t, s_nt = self.encoder_doc_lstm(x_t, s_t, finished)
            finished = tf.greater_equal(t+1, inputs_len)
            x_nt = tf.cond(tf.reduce_all(finished), lambda: tf.zeros([batch_size, self.emb_size], dtype=tf.float32),
                                     lambda: inputs_ta.read(t+1))

            h = tf.nn.relu(self.gated_linear(tf.concat([o_t, x_nt], axis=-1)))
            p = tf.sigmoid(self.gated_output(o_t))
            x_nt = x_nt * p
            o_t = o_t * p
            emit_ta = emit_ta.write(t, o_t)

            return t+1, x_nt, s_nt, emit_ta, finished

        _, _, state, emit_ta, _ = tf.while_loop(
            cond=lambda _1, _2, _3, _4, finished: tf.logical_not(tf.reduce_all(finished)),
            body=loop_fn,
            loop_vars=(time, inputs_ta.read(0), h0, emit_ta, f0))

        outputs = tf.transpose(emit_ta.stack(), [1,0,2])
        return outputs, state


    def decoder_layer_training(self, distract_state, initial_state, inputs, inputs_len):
        batch_size = tf.shape(self.decoder_inputs)[0]
        max_time = tf.shape(self.decoder_inputs)[1]

        time = tf.constant(0, dtype=tf.int32)
        h0 = initial_state
        f0 = tf.zeros([batch_size], dtype=tf.bool)
        inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
        inputs_ta = inputs_ta.unstack(tf.transpose(inputs, [1,0,2]))
        x0 = inputs_ta.read(time)
        emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
        emit_ad = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
        emit_pg = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
        c_v_d = tf.zeros([self.batch_size, self.hidden_size])
        ds_attn = distract_state


        def loop_fn(t, x_t, s_t, emit_ta, finished, distract_state, emit_ad, emit_pg, c_v_d, ds_attn):
            s_t_c = fc_layer(tf.concat([x_t, s_t[1]], -1), self.hidden_size, scope='linear_dec_attn_inp', activation_fn=None)
            c_v, att_dist = self.att_layer_doc_dec(s_t_c)
            c_v_d, distract_state = self.distraction_cell(c_v, distract_state)
            #if (self.args.mode=='decode'):
            #    distract_state.set_shape([self.batch_size,self.hidden_size])
            x_t_c = fc_layer(tf.concat([x_t, c_v_d], -1), self.emb_size, scope='linear_inp_dec', activation_fn=None)
            o_t, s_nt = self.decoder_lstm(x_t_c, s_t, finished)             
            #att_dist_d = self.distraction_cell(att_dist, ds_attn)
            p_ptr = tf.sigmoid(fc_layer(tf.concat([x_t, s_nt[1], c_v_d], -1), 1, scope='p_ptr_linear', activation_fn=None))
            o_t_v = fc_layer(tf.concat([o_t, c_v_d], -1), self.hidden_size, scope='linear_out_v', activation_fn=None) 
            o_t_v = self.decoder_output_unit(o_t_v, finished)
            emit_ta = emit_ta.write(t, o_t_v)
            emit_ad = emit_ad.write(t, att_dist)
            emit_pg = emit_pg.write(t, p_ptr)
            
            finished = tf.greater_equal(t+1, inputs_len)
            x_nt = tf.cond(self.feed_previous, 
                            lambda: tf.nn.embedding_lookup(self.word_embeddings_doc, tf.arg_max(o_t_v, 1)), 
                            lambda: tf.cond(tf.reduce_all(finished), 
                                    lambda: tf.zeros([batch_size, self.emb_size], dtype=tf.float32), 
                                    lambda: inputs_ta.read(t+1)))
            
            return t+1, x_nt, s_nt, emit_ta, finished, distract_state, emit_ad, emit_pg, c_v_d, ds_attn

        _, _, state, emit_ta, _, distract_state, emit_ad, emit_pg, c_v_d, ds_attn = tf.while_loop(
            cond=lambda _1, _2, _3, _4, finished, distract_state, emit_ad, emit_pg, c_v_d, ds_attn: tf.logical_not(tf.reduce_all(finished)),
            body=loop_fn,
            loop_vars=(time, x0, h0, emit_ta, f0, distract_state, emit_ad, emit_pg, c_v_d, ds_attn))

        outputs = tf.transpose(emit_ta.stack(), [0,1,2])
        attn_dists = tf.transpose(emit_ad.stack(), [0,1,2])
        p_ptrs = tf.transpose(emit_pg.stack(), [0,1,2])

        return outputs, distract_state, state, attn_dists, p_ptrs


    def decoder_layer_inference(self, distract_state, initial_state):
        batch_size = tf.shape(self.encoder_doc_inputs)[0]

        time = tf.constant(0, dtype=tf.int32)
        h0 = initial_state
        f0 = tf.zeros([batch_size], dtype=tf.bool)
        x0 = tf.nn.embedding_lookup(self.word_embeddings_doc, tf.fill([batch_size], self.start_token))
        emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
        emit_ad = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
        emit_pg = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
        c_v_d = tf.zeros([self.batch_size, self.hidden_size])

        def loop_fn(t, x_t, s_t, emit_ta, finished, distract_state, emit_ad, emit_pg, c_v_d):
            s_t_c = fc_layer(tf.concat([x_t, s_t[1]], -1), self.hidden_size, scope='linear_dec_attn_inp', activation_fn=None)
            c_v, att_dist = self.att_layer_doc_dec(s_t_c)
            c_v_d, distract_state = self.distraction_cell(c_v, distract_state)
            x_t_c = fc_layer(tf.concat([x_t, c_v_d], -1), self.emb_size, scope='linear_inp_dec', activation_fn=None)
            o_t, s_nt = self.decoder_lstm(x_t_c, s_t, finished)             
            #att_dist_d = self.distraction_cell(att_dist, ds_attn)
            p_ptr = tf.sigmoid(fc_layer(tf.concat([x_t, s_nt[1], c_v_d], -1), 1, scope='p_ptr_linear', activation_fn=None))
            o_t_v = fc_layer(tf.concat([s_nt[1], c_v_d], -1), self.hidden_size, scope='linear_out_v', activation_fn=None) 
            o_t_v = self.decoder_output_unit(o_t_v, finished)
            emit_ta = emit_ta.write(t, o_t_v)
            emit_ad = emit_ad.write(t, att_dist)
            emit_pg = emit_pg.write(t, p_ptr)

            next_token = tf.arg_max(o_t_v, 1)
            x_nt = tf.nn.embedding_lookup(self.word_embeddings_doc, next_token)
            finished = tf.logical_or(finished, tf.equal(next_token, self.stop_token))
            finished = tf.logical_or(finished, tf.greater_equal(t+1, self.max_length))
            return t+1, x_nt, s_nt, emit_ta, finished, distract_state, emit_ad, emit_pg, c_v_d

        _, _, state, emit_ta, _, _, emit_ad, emit_pg, c_v_d = tf.while_loop(
            cond=lambda _1, _2, _3, _4, finished, distract_state, emit_ad, emit_pg, c_v_d: tf.logical_not(tf.reduce_all(finished)),
            body=loop_fn,
            loop_vars=(time, x0, h0, emit_ta, f0, distract_state, emit_ad, emit_pg, c_v_d))

        outputs = tf.transpose(emit_ta.stack(), [0,1,2])
        attn_dists = tf.transpose(emit_ad.stack(), [0,1,2])
        p_ptrs = tf.transpose(emit_pg.stack(), [0,1,2])
        pred_ids = tf.arg_max(outputs, 2)
        print ("PRED IDS: ", pred_ids)
        return outputs, state, attn_dists, p_ptrs
    

    def calc_training_loss(self, final_dists):
        loss_per_step = [] # will be list length max_dec_steps containing shape (batch_size)
        batch_nums = tf.range(0, limit=self.batch_size) # shape (batch_size)
        batch_nums = tf.expand_dims(batch_nums, 1) # shape (batch_size, 1)
        
        dec_len = tf.shape(self.decoder_outputs)[1]

        batch_nums = tf.tile(batch_nums, [1, dec_len]) # shape (batch_size, dec_len)

        indices = tf.stack( (batch_nums, self.decoder_outputs), axis=2) # shape (batch_size, dec_len, 2)
        indices = tf.transpose(indices, [1, 0, 2])
        #final_dists = tf.transpose(final_dists, [1, 0, 2])
        gold_probs, _ = tf.map_fn(lambda x: (tf.gather_nd(x[0], x[1]), x[1]), (final_dists, indices), dtype=(final_dists[0].dtype, indices[0].dtype)) # shape (batch_size, dec_len). prob of correct words on this step
        
        gold_probs = tf.transpose(gold_probs, [1,0])
        self.gold_probs = gold_probs  
        eps = 1e-13 
        gold_probs = gold_probs + eps 
        losses = -tf.log(gold_probs) 
        
        loss = self.training_mask_and_avg(losses, self.dec_padding_mask)
        return loss
        
    def training_mask_and_avg(self, values, padding_mask):
        dec_lens = tf.reduce_sum(padding_mask, axis=1) # shape batch_size. float32
        #print ("DEC_LENS: ", dec_lens.dtype)
        values_per_step = values * padding_mask
        #print ("values_per_step: ", values_per_step.dtype)
        values_per_ex, _ = tf.map_fn(lambda x: (tf.reduce_sum(x[0])/x[1], x[1]), (values_per_step, dec_lens), dtype=(values_per_step[0].dtype, dec_lens[0].dtype))
        #values_per_ex = sum(values_per_step)/dec_lens # shape (batch_size); normalized value for each batch member
        #print ("values_per_ex: ", values_per_ex.dtype)

        return tf.reduce_mean(values_per_ex) # overall average
        
    def calc_inference_loss(self, final_dists):
        loss_per_step = [] # will be list length max_dec_steps containing shape (batch_size)
        batch_nums = tf.range(0, limit=self.batch_size) # shape (batch_size)
        batch_nums = tf.expand_dims(batch_nums, 1) # shape (batch_size, 1)
        
        dec_len = tf.shape(self.decoder_outputs)[1]

        batch_nums = tf.tile(batch_nums, [1, dec_len]) # shape (batch_size, dec_len)

        indices = tf.stack( (batch_nums, self.decoder_outputs), axis=2) # shape (batch_size, dec_len, 2)
        indices = tf.transpose(indices, [1, 0, 2])
        #final_dists = tf.transpose(final_dists, [1, 0, 2])
        shape_difference = tf.shape(indices)[0] - tf.shape(final_dists)[0]
        
        final_dists = tf.cond(tf.shape(indices)[0] > tf.shape(final_dists)[0], lambda: tf.concat([final_dists, tf.fill([shape_difference, self.batch_size, tf.shape(final_dists)[2]], 0.01)], 0), lambda: final_dists)
        self.inference_look = tf.shape(final_dists)
        gold_probs, _ = tf.map_fn(lambda x: (tf.gather_nd(x[0], x[1]), x[1]), (final_dists, indices), dtype=(final_dists[0].dtype, indices[0].dtype)) # shape (batch_size, dec_len). prob of correct words on this step
        
        gold_probs = tf.transpose(gold_probs, [1,0])
        self.gold_probs = gold_probs  
        eps = 1e-13 
        gold_probs = gold_probs + eps 
        losses = -tf.log(gold_probs) 
        
        used = tf.sign(gold_probs)
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        dec_padding_mask = tf.sequence_mask(length, dtype=tf.float32)
        
        loss = self.inference_mask_and_avg(losses, dec_padding_mask)
        return loss
        
    def inference_mask_and_avg(self, values, padding_mask):
        dec_lens = tf.reduce_sum(padding_mask, axis=1) # shape batch_size. float32
        print ("DEC_LENS: ", dec_lens.dtype)
        values_per_step = values * padding_mask
        print ("values_per_step: ", values_per_step.dtype)
        values_per_ex, _ = tf.map_fn(lambda x: (tf.reduce_sum(x[0])/x[1], x[1]), (values_per_step, dec_lens), dtype=(values_per_step[0].dtype, dec_lens[0].dtype))
        #values_per_ex = sum(values_per_step)/dec_lens # shape (batch_size); normalized value for each batch member
        print ("values_per_ex: ", values_per_ex.dtype)

        return tf.reduce_mean(values_per_ex) # overall average

    def calc_final_dist(self, vocab_dists, attn_dists, p_ptrs):
            
        with tf.variable_scope('final_distribution'):
          # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
          extended_vsize = self.source_vocab_size + self.max_docs_oovs # the maximum (over the batch) size of the extended vocabulary
          extra_zeros = tf.zeros((self.batch_size, self.max_docs_oovs))
          vocab_dists_extended = tf.map_fn(lambda dist: tf.concat(axis=1, values=[dist, extra_zeros]), vocab_dists)

          # Project the values in the attention distributions onto the appropriate entries in the final distributions
          # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary, then we add 0.1 onto the 500th entry of the final distribution
          # This is done for each decoder timestep.
          # This is fiddly; we use tf.scatter_nd to do the projection
          batch_nums = tf.range(0, limit=self.batch_size) # shape (batch_size)
          batch_nums = tf.expand_dims(batch_nums, 1) # shape (batch_size, 1)
          attn_len = tf.shape(self.encoder_doc_inputs_ext)[1] # number of states we attend over
          batch_nums = tf.tile(batch_nums, [1, attn_len]) # shape (batch_size, attn_len)
          indices = tf.stack( (batch_nums, self.encoder_doc_inputs_ext), axis=2) # shape (batch_size, enc_t, 2)
          shape = [self.batch_size, extended_vsize]
          attn_dists_projected = tf.map_fn(lambda copy_dist: tf.scatter_nd(indices, copy_dist, shape), attn_dists) # list length max_dec_steps (batch_size, extended_vsize)

          # Add the vocab distributions and the copy distributions together to get the final distributions
          # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving the final distribution for that decoder timestep
          # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
          
          vocab_dists_first_dim = tf.shape(vocab_dists_extended)[0]
          vocab_dists_second_dim = tf.shape(vocab_dists_extended)[1]
          
          vocab_dists_extended_reshape = tf.reshape(vocab_dists_extended, [vocab_dists_first_dim*vocab_dists_second_dim, -1])          
          attn_dists_projected_reshape = tf.reshape(attn_dists_projected, [vocab_dists_first_dim*vocab_dists_second_dim, -1])
          p_ptrs_reshape = tf.reshape(p_ptrs, [vocab_dists_first_dim*vocab_dists_second_dim, -1])
          final_dists, _, _ = tf.map_fn(lambda x: (tf.cond(tf.greater(x[2][0], 0.5), lambda: x[1], lambda: x[0]), x[1], x[2]), (vocab_dists_extended_reshape, attn_dists_projected_reshape, p_ptrs_reshape), dtype=(vocab_dists_extended_reshape[0].dtype, attn_dists_projected_reshape[0].dtype, p_ptrs_reshape[0].dtype))          
          final_dists = tf.reshape(final_dists, tf.shape(vocab_dists_extended))
          return final_dists


    def __call__(self, x, sess, feed_previous=False):
        train_feed = self.make_train_feed(x, feed_previous)
        ploss, loss, loss2, _, accuracy, _, summary_merge, global_step = sess.run([self.pointer_loss, self.overall_loss,  self.training_loss, self.predicted_ids, self.training_accuracy, self.updates, self.summary_merge, self.global_step], train_feed)
        return ploss, loss, loss2, accuracy, summary_merge, global_step
        
    def validate(self, x, sess):
        val_feed = self.make_test_feed(x)
        predicted_ids = sess.run([self.predicted_ids], val_feed)
        return predicted_ids

    def generate(self, x, sess):
        test_feed = self.make_test_feed(x)
        predicted_ids = sess.run(self.predicted_ids, test_feed)
        return predicted_ids
        
    def make_train_feed(self, x, feed_previous):
        train_feed = {
               self.encoder_doc_inputs_ext : [x[0][i][4] for i, a in enumerate(x[0])],
               self.max_docs_oovs : x[3],
               self.encoder_doc_inputs: [x[0][i][2] for i, a in enumerate(x[0])], 
               self.encoder_doc_c_inputs: [x[0][i][3] for i, a in enumerate(x[0])], 
               self.encoder_doc_len: [x[0][i][1] for i, a in enumerate(x[0])], 
               self.doc_seq_indices : [x[0][i][0] for i, a in enumerate(x[0])],
               self.enc_padding_mask : [x[0][i][6] for i, a in enumerate(x[0])],
               self.encoder_query_inputs: [x[1][i][2] for i, a in enumerate(x[1])], 
               self.encoder_query_c_inputs: [x[1][i][3] for i, a in enumerate(x[1])], 
               self.encoder_query_len: [x[1][i][1] for i, a in enumerate(x[1])], 
               self.query_seq_indices : [x[1][i][0] for i, a in enumerate(x[1])],
               self.feed_previous : feed_previous,                           
               self.decoder_inputs: [x[2][i][2][0:-1] for i, a in enumerate(x[2])], 
               self.decoder_outputs: [x[2][i][3][1:] for i, a in enumerate(x[2])], 
               self.pointer_switches: [x[2][i][4][1:] for i, a in enumerate(x[2])], 
               self.decoder_len: [x[2][i][1] for i, a in enumerate(x[2])],
               self.dec_padding_mask : [x[2][i][5] for i, a in enumerate(x[2])],

               }
        return train_feed
        
    def make_test_feed(self, x):
        test_feed =  {              
                 self.encoder_doc_inputs_ext : [x[0][i][4] for i, a in enumerate(x[0])],
               self.max_docs_oovs : x[3],
               self.encoder_doc_inputs: [x[0][i][2] for i, a in enumerate(x[0])], 
               self.encoder_doc_c_inputs: [x[0][i][3] for i, a in enumerate(x[0])], 
               self.encoder_doc_len: [x[0][i][1] for i, a in enumerate(x[0])], 
               self.doc_seq_indices : [x[0][i][0] for i, a in enumerate(x[0])],
               self.enc_padding_mask : [x[0][i][6] for i, a in enumerate(x[0])],
               self.encoder_query_inputs: [x[1][i][2] for i, a in enumerate(x[1])], 
               self.encoder_query_c_inputs: [x[1][i][3] for i, a in enumerate(x[1])], 
               self.encoder_query_len: [x[1][i][1] for i, a in enumerate(x[1])], 
               self.query_seq_indices : [x[1][i][0] for i, a in enumerate(x[1])], 
               }
        return test_feed
            
    def run_encoder(self, sess, batch):   
        feed_dict = self.make_test_feed(batch) 
        (enc_states, distract_state, dec_in_state) = sess.run([self.query_aware_docs, self.distract_state, self.decoder_initial], feed_dict) 

        # dec_in_state is LSTMStateTuple shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
        # Given that the batch is a single example repeated, dec_in_state is identical across the batch so we just take the top row.
        #print ("DEC IN STATE 0 SHAPE: ", np.shape(dec_in_state[0]))
        #print ("DEC IN STATE SHAPE: ", np.shape(dec_in_state)) (2, batch_size, hidden_size)
        return enc_states, [a[0] for a in distract_state], [a[0] for a in dec_in_state]
        
    def decode_onestep(self, sess, batch, latest_tokens, enc_states, distract_states, dec_init_states):
    
        beam_size = len(dec_init_states)
        #print ("BEAM SIZE: ", beam_size, " FIRST SIZE: ", np.shape(distract_states), " SECOND SIZE: ", np.shape(dec_init_states))

        # Turn dec_init_states (a list of LSTMStateTuples) into a single LSTMStateTuple for the batch
        hiddens = [np.expand_dims(state[0], axis=0) for state in dec_init_states]
        cells = [np.expand_dims(state[1], axis=0) for state in dec_init_states]
        new_h = np.concatenate(hiddens, axis=0)  # shape [batch_size,hidden_dim]
        new_c = np.concatenate(cells, axis=0)  # shape [batch_size,hidden_dim]
        new_dec_in_state = (new_h, new_c)
        #print ("NEW DEC: ", np.shape(new_dec_in_state))
        
        dhiddens = [np.expand_dims(state[0], axis=0) for state in distract_states]
        dcells = [np.expand_dims(state[1], axis=0) for state in distract_states]
        dnew_h = np.concatenate(dhiddens, axis=0)  # shape [batch_size,hidden_dim]
        dnew_c = np.concatenate(dcells, axis=0)  # shape [batch_size,hidden_dim]
        new_distract_state = (dnew_h, dnew_c)
        #print ("DNEW DEC: ", np.shape(new_distract_state))
        
        to_return = {
          "ids": self.topk_ids,
          "probs": self.topk_log_probs,
          "distract_states":self.training_final_distract_state,
          "states": self.training_dec_final_state,
          "attn_dists": self.training_attn_dists,
          "p_ptrs": self.training_p_ptrs
          }

        feed = {
            self.query_aware_docs: enc_states,
            self.distract_state: new_distract_state,
            self.decoder_initial: new_dec_in_state,
            self.decoder_inputs: np.transpose(np.array([latest_tokens])),
            self.decoder_len: [1]*beam_size,
            self.encoder_doc_inputs_ext : [batch[0][i][4] for i, a in enumerate(batch[0])],
            self.max_docs_oovs : batch[3],
            self.enc_padding_mask : [batch[0][i][6] for i, a in enumerate(batch[0])],
            self.feed_previous: False,
          }
            
        results = sess.run(to_return, feed_dict=feed) # run the decoder step

        # Convert results['states'] (a single LSTMStateTuple) into a list of LSTMStateTuple -- one for each hypothesis
        new_states = [tf.contrib.rnn.LSTMStateTuple(results['states'][0][i, :], results['states'][1][i, :]) for i in range(beam_size)]
        new_distract_states = [tf.contrib.rnn.LSTMStateTuple(results['distract_states'][0][i, :], results['distract_states'][1][i, :]) for i in range(beam_size)]
        #print ("NEW DISTRACT STATES: ", new_distract_states)

        # Convert singleton list containing a tensor to a list of k arrays
        assert len(results['attn_dists'])==1
        attn_dists = results['attn_dists'][0].tolist()

        assert len(results['p_ptrs'])==1
        p_ptrs = results['p_ptrs'][0].tolist()
        
        return results['ids'], results['probs'], new_distract_states, new_states, attn_dists, p_ptrs



            