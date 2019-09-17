from __future__ import print_function
import sys
import os
import tensorflow as tf
import time
from Unit import *
import numpy as np
from args import load_args
from data_loader import *
from data_loader import _buckets
from args import load_args
import pickle
from vocabulary import load_dict, sen_map2tok, load_dictionaries
import itertools


MARK_PAD = "<PAD>"
MARK_UNK = "<UNK>"
MARK_EOS = "<EOS>"
MARK_GO = "<GO>"
MARKS = [MARK_PAD, MARK_UNK, MARK_EOS, MARK_GO]
ID_PAD = 0
ID_UNK = 1
ID_EOS = 2
ID_GO = 3


log_file = 'log_train.txt'

def write_log(s):
    print(s, end='\n')
    with open(log_file, 'a') as f:
        f.write(s)
        
write_log("\n New Run Started ************************************************** \n")

    
def create_model(session, batch_size, load_checkpoint, args, is_training):
    dtype = tf.float32
    model = Model(args, batch_size=batch_size, scope_name="seq2seq", name="seq2seq", is_training=is_training)
    print ("Loading Checkpoint: ", load_checkpoint)
    if (load_checkpoint):        
        ckpt = tf.train.latest_checkpoint(args.train_dir)
        if ckpt:
            #ckpt = ckpt.model_checkpoint_path
            if ckpt and tf.train.checkpoint_exists(ckpt):
                print("Reading model parameters from %s" % ckpt)
                model.saver.restore(session, ckpt)
                print ("DONE RESTORING CHECKPOINT")
            else:
                raise Exception("Don't have any checkpoints to load: %s" % ckpt)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model

def train(args, sess, data, model):
    train_writer = tf.summary.FileWriter(args.tfboard+'/train', sess.graph)    
    
    trainset = data[0]
    evalset = data[1]
    eval_counter = 0
    for epoch in range(args.max_epochs):
        feed_previous = args.feed_previous or epoch > args.feed_previous_epoch
        print ("Feed Previous: ", feed_previous)

        k = 0
        loss, accuracy, total_epoch_time, start_time = 0.0, 0.0, 0.0, time.time()
        for x in trainset:
            loss_step, accuracy_step, summary_merge, global_step = model(x, sess, feed_previous)
            loss += loss_step
            accuracy += accuracy_step
            k += 1
            
            train_writer.add_summary(summary_merge, global_step)
            
            if (k % args.steps_per_print == 0):
                per_print_time = time.time() - start_time
                total_epoch_time += per_print_time
                cost_time = per_print_time / args.steps_per_print
                write_log("EPOCH: %d, STEP: %d, LOSS: %.3f, TIME: %.3f, ACCURACY: %.3f " % (epoch, k, loss_step, cost_time, accuracy_step))
                start_time = time.time()
                
        total_epoch_time += time.time() - start_time
        write_log("EPOCH: %d, STEP: %d, LOSS:: %.3f, TIME: %.3f, ACCURACY:: %.3f " % (epoch, k, loss/len(trainset), total_epoch_time/len(trainset), accuracy / len(trainset)))
      
        print ("SAVING CHECKPOINT ...")
        checkpoint_path = os.path.join(args.train_dir, "model.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step, write_meta_graph=False)
        print ("DONE SAVING CHECKPOINT")
        
        if (eval_counter == len(evalset)):
            eval_counter = 0
        #eval_counter = validate_training(sess, evalset, model, eval_counter)
        
        loss, accuracy, total_epoch_time, start_time = 0.0, 0.0, 0.0, time.time() 
        
        print ("DONE WITH EPOCH AND VALIDATION")

def get_validation_accuracy(targets, logits):
        targets_shape = len(targets[0])
        logits_shape = len(logits[1])
        max_seq = max(targets_shape, logits_shape)
        #print ("MAX SEQ: ", max_seq, " TARGET SHAPE: ", targets_shape, " LOGITS SHAPE: ", logits_shape)
        if max_seq - targets_shape:
            #targets = np.pad(targets, (0,max_seq - targets_shape), 'constant', constant_values=(0, 2))
            targets = np.pad(targets, [(0,0),(0,max_seq - targets_shape)], 'constant', constant_values=(0, 2))
        if max_seq - logits_shape:
            logits = np.pad(logits,[(0,0),(0,max_seq - logits_shape)], 'constant', constant_values=(0, 2))
        #print ("TARGETS: ", targets)
        #print ("LOGITS: ", logits)
        return np.mean(np.equal(targets, logits))
    
def validate_training(sess, evalset, model, counter=0):     
    print ("VALIDATION BATCH: ", counter)
    pred_ids = model.validate(evalset[counter], sess)  
    targets = [evalset[counter][2][i][3][1:] for i, a in enumerate(evalset[counter][2])]
    #print ("PREDICTED IDS: ", pred_ids)
    accuracy = get_validation_accuracy(targets, pred_ids[0])
    write_log("TEST STEP: accuracy = %.3f " % (accuracy))
    counter += 1
    return counter        

def main():

    train_graph = tf.Graph()
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)    
    # please do not use the totality of the GPU memory
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True
    config.gpu_options.allocator_type = 'BFC'
    
    with tf.Session(graph=train_graph, config=config) as sess:
    
        args = load_args()  
        
        tf.set_random_seed(args.seed)
        np.random.seed(seed=args.seed)         
        
        train_docs_all, train_queries_all, train_summaries_all, doc_count, query_count, summary_count = load_train_dataid_pickle()
        val_docs_all, val_queries_all, val_summaries_all, doc_count, query_count, summary_count = load_val_dataid_pickle()
        
        if(args.toy):
            train_docs_all, train_queries_all, train_summaries_all = train_docs_all[:args.toy], train_queries_all[:args.toy], train_summaries_all[:args.toy]
            val_docs_all, val_queries_all, val_summaries_all = val_docs_all[:args.toy], val_queries_all[:args.toy], val_summaries_all[:args.toy]
        print ("LENGTH OF TRAIN DOC ALL: ", len(train_docs_all))
            
        train_set = create_bucket(list(zip(train_docs_all, train_queries_all, train_summaries_all)))
        val_set = create_bucket(list(zip(val_docs_all, val_queries_all, val_summaries_all)))
        
        batched_train_set = batchify(train_set, _buckets, args.train_batch_size, args.max_word_size)
        batched_val_set = batchify(val_set, _buckets, args.train_batch_size, args.max_word_size)
        
        data_all = batched_train_set, batched_val_set 
        
        model = create_model(sess, args.train_batch_size, args.train_load_checkpoint, args, True)
        
        train(args, sess, data_all, model) 
        #eval_counter = validate_training(sess, batched_val_set, model, 0)        

if __name__=='__main__':
    with tf.device('/gpu:1'):
        main()