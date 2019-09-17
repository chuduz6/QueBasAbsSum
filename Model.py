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
        self.dtype = tf.float32
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
        self.trunc_norm_init = 1e-4

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

        with tf.variable_scope("encoder_query"):
            #QUERY: encoding using RNN-LSTM. Simple encoding should do for query as there aren't many query words
            self.en_query_outputs, en_query_state = self.bi_lstm_layer(self.encoder_query_embed, self.encoder_query_len, args.query_encoder_dropout_flag, args.query_encoder_keep_prob, is_training, 'query_encoder')        
            
        with tf.variable_scope('encoder_doc'):
            #DOCUMENT: gated document encoder
            en_doc_outputs, en_doc_state = self.bi_lstm_layer(self.encoder_doc_embed, self.encoder_doc_len, args.doc_encoder_dropout_flag, args.doc_encoder_keep_prob, is_training, 'doc_encoder')  
            
            #DOCUMENT: Extracting features using Modified Inception Network
            en_doc_outputs_inception = InceptionUnit("inception_net_doc", self.hidden_size, is_training, args.doc_encoder_keep_prob).__call__(en_doc_outputs)

            #DOCUMENT: Gating Mechanism to self-attended extracted features
            en_doc_outputs_inc_attended = SelfAttention(en_doc_outputs_inception, self.hidden_size, "self_attention_doc").__call__()
            gate_doc = tf.sigmoid(en_doc_outputs_inc_attended)   
            self.en_doc_outputs = gate_doc * en_doc_outputs       

        
        with tf.variable_scope('decoder_initial'):
            print ("EN DOC STATE: ", en_doc_state)
            decoder_initial_state = fc_layer(tf.concat([en_query_state[1], en_doc_state[1]], 1), self.hidden_size, scope='linear_decoder_initial_state', activation_fn = None)
            decoder_initial_outputs = fc_layer(tf.concat([en_query_state[0], en_doc_state[0]], 1), self.hidden_size, scope='linear_decoder_initial_outputs', activation_fn = None)
            #self.decoder_initial = tf.contrib.rnn.LSTMStateTuple(decoder_initial_outputs, decoder_initial_state)
            self.decoder_initial = (decoder_initial_outputs, decoder_initial_state)
            self.distract_state = self.decoder_initial
            print ("DECODER INITIAL: ", self.decoder_initial)
        
            
        with tf.variable_scope('decoder_attention'):
            self.att_layer_doc_dec = DecoderAttention(self.en_doc_outputs, self.hidden_size, "doc_encoder_decoder_attention")
            self.att_layer_query_dec = DecoderAttention(self.en_query_outputs, self.hidden_size, "query_encoder_decoder_attention")
        
        training_final_dists, self.training_final_distract_state , self.training_dec_final_state, self.training_attn_dists, self.training_p_ptrs = self.add_decoder_layer(self.distract_state, self.decoder_initial, is_training)

        self.training_logits = tf.identity(training_final_dists, name='training_logits')
        self.predicted_ids = tf.identity(tf.arg_max(self.training_logits, 2, output_type=tf.int32), name='predicted_ids')
        print ("PREDICTED IDS: ", self.predicted_ids)
        
        if(args.mode=='decode'):
            #assert len(training_final_dists)==1 # final_dists is a singleton list containing shape (batch_size, extended_vsize)
            #d_training_final_dists = d_training_final_dists[0]
            topk_probs, self.topk_ids = tf.nn.top_k(training_final_dists, self.batch_size*2) # take the k largest probs. note batch_size=beam_size in decode mode
            self.topk_log_probs = tf.log(topk_probs)
        
        with tf.variable_scope('optimization'):
            
            weights = tf.sequence_mask(self.decoder_len, dtype=tf.float32)
            loss_t = tf.contrib.seq2seq.sequence_loss(training_final_dists, self.decoder_outputs, weights, average_across_timesteps=False, average_across_batch=False)
            dec_lens = tf.reduce_sum(self.dec_padding_mask, axis=1)
            loss_b = tf.reduce_sum(loss_t, axis=1)/dec_lens
            self.overall_loss = tf.reduce_mean(loss_b)
            
            self.training_accuracy = self.get_training_accuracy(training_final_dists, self.decoder_outputs)

            tvars = tf.trainable_variables()
            optimizer = tf.train.AdamOptimizer(args.learning_rate)
            
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.overall_loss, tvars), self.grad_clip)
            self.updates = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)  
        
            tf.summary.scalar('overall_loss', self.overall_loss)
            tf.summary.scalar('training_accuracy', self.training_accuracy)

        self.saver = tf.train.Saver(max_to_keep=args.max_to_keep)
        self.summary_merge = tf.summary.merge_all()
        
    
    def bi_lstm_layer(self, inputs_emb, sequence_length, dropout_flag, keep_prob, forward_only, scope):
        with tf.variable_scope(scope):                                   
            fw_cell = tf.contrib.rnn.LSTMCell(self.hidden_size)
            bw_cell = tf.contrib.rnn.LSTMCell(self.hidden_size)
            if(dropout_flag):
                if not forward_only:
                    fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=keep_prob)
                    bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=keep_prob)
                    
            outputs, (states_fw, states_bw) = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs_emb, sequence_length=sequence_length, dtype=self.dtype)
            
            states_c = fc_layer(tf.concat((states_fw.c, states_bw.c), 1), self.hidden_size, scope='lin_encoder_states_concat', activation_fn=tf.nn.relu)
            states_h = fc_layer(tf.concat((states_fw.h, states_bw.h), 1), self.hidden_size, scope='lin_encoder_output_concat', activation_fn=tf.nn.relu )
            states = states_h, states_c
            
            outputs = fc_layer(tf.concat(outputs, 2), self.hidden_size, scope='lin_encoder_output_concat', activation_fn=tf.nn.relu, reuse=True)          
            
            #outputs = tf.transpose(outputs, [1, 0, 2])            
            
            return outputs, states 
            
            
    def add_decoder_layer (self, distract_state, en_doc_state, is_training):
        with tf.variable_scope('decoder'):
            training_dec_outputs, training_distract_state, training_dec_final_state, training_attn_dists, training_p_ptrs = self.decoder_layer_training(distract_state, en_doc_state, self.decoder_embed, self.decoder_len)     
            return (training_dec_outputs, training_distract_state, training_dec_final_state, training_attn_dists, training_p_ptrs)
        
    
    def get_training_accuracy(self, logits, targets):
        predictions = tf.cast(tf.argmax(logits, axis=2), tf.int32) 
        accuracy = tf.contrib.metrics.accuracy(predictions, targets)
        return accuracy        
    

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
            o_t, s_nt = self.decoder_lstm(x_t, s_t, finished)             
            q_out, q_attn = self.att_layer_query_dec(o_t)
            q_dec_out = fc_layer(tf.concat([o_t, q_out], -1), self.hidden_size, scope='linear_doc_attn_inp', activation_fn=None)
            c_v, att_dist = self.att_layer_doc_dec(q_dec_out)
            p_ptr = tf.sigmoid(fc_layer(tf.concat([x_t, s_nt[1], c_v], -1), 1, scope='p_ptr_linear', activation_fn=None))
            o_t_v = self.decoder_output_unit(c_v, finished)
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

        outputs = tf.transpose(emit_ta.stack(), [1,0,2])
        attn_dists = tf.transpose(emit_ad.stack(), [0,1,2])
        p_ptrs = tf.transpose(emit_pg.stack(), [0,1,2])

        return outputs, distract_state, state, attn_dists, p_ptrs


    def __call__(self, x, sess, feed_previous=False):
        train_feed = self.make_train_feed(x, feed_previous)
        loss, _, accuracy, _, summary_merge, global_step = sess.run([self.overall_loss,  self.predicted_ids, self.training_accuracy, self.updates, self.summary_merge, self.global_step], train_feed)
        return loss, accuracy, summary_merge, global_step
        
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
               self.decoder_outputs: [x[2][i][2][1:] for i, a in enumerate(x[2])], 
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
               self.enc_padding_mask : [x[0][i][7] for i, a in enumerate(x[0])],
               self.encoder_query_inputs: [x[1][i][2] for i, a in enumerate(x[1])], 
               self.encoder_query_c_inputs: [x[1][i][3] for i, a in enumerate(x[1])], 
               self.encoder_query_len: [x[1][i][1] for i, a in enumerate(x[1])], 
               self.query_seq_indices : [x[1][i][0] for i, a in enumerate(x[1])], 
               }
        return test_feed
            
    def run_encoder(self, sess, batch):   
        feed_dict = self.make_test_feed(batch) 
        (en_doc_outputs, en_query_outputs, distract_state, dec_in_state) = sess.run([self.en_doc_outputs, self.en_query_outputs, self.distract_state, self.decoder_initial], feed_dict) 

        # dec_in_state is LSTMStateTuple shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
        # Given that the batch is a single example repeated, dec_in_state is identical across the batch so we just take the top row.
        #print ("DEC IN STATE 0 SHAPE: ", np.shape(dec_in_state[0]))
        #print ("DEC IN STATE SHAPE: ", np.shape(dec_in_state)) (2, batch_size, hidden_size)
        return en_doc_outputs, en_query_outputs, [a[0] for a in distract_state], [a[0] for a in dec_in_state]
        
    def decode_onestep(self, sess, batch, latest_tokens, en_doc_outputs, en_query_outputs, distract_states, dec_init_states):
    
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
            self.en_doc_outputs: en_doc_outputs,
            self.en_query_outputs: en_query_outputs,
            self.distract_state: new_distract_state,
            self.decoder_initial: new_dec_in_state,
            self.decoder_inputs: np.transpose(np.array([latest_tokens])),
            self.decoder_len: [1]*beam_size,
            self.encoder_doc_inputs_ext : [batch[0][i][4] for i, a in enumerate(batch[0])],
            self.max_docs_oovs : batch[3],
            self.enc_padding_mask : [batch[0][i][7] for i, a in enumerate(batch[0])],
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



            