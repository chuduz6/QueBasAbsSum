 # coding: utf-8
from nltk.tokenize import sent_tokenize, word_tokenize
from args import load_args
from vocabulary import load_dict
import numpy as np
import tensorflow as tf
import pickle
import tensorflow_hub as hub
from utils import *
from args import load_args
from data_loader import load_text

def store_sentence_embeddings(docs_sents_embds, queries_sents_embds):
    store_pickle((docs_sents_embds, queries_sents_embds), 'sentence_embeddings.pickle')

def load_sentence_embeddings():
    return load_pickle('sentence_embeddings.pickle')

def get_embeddings_sentence_encoder(sents_docs, sents_queries):
    with tf.device('/cpu:0'):
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)    
        # please do not use the totality of the GPU memory
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.gpu_options.allow_growth = True
        config.gpu_options.allocator_type = 'BFC'
        
        with tf.Session(config=config) as sess_embd: 
            module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

            print ("LOADING MODULE ...")
            # Import the Universal Sentence Encoder's TF Hub module
            embed = hub.Module(module_url)
            print ("DONE LOADING MODULE")
            
            print ("LENGTH OF SENT DOCS: ", len(sents_docs))
            print ("LENGTH OF SENT QUERIES: ", len(sents_queries))                        
            
                                              
            sess_embd.run([tf.global_variables_initializer(), tf.tables_initializer()])
            
            count = 0
            chunk_size = 20000
            
            while (count + chunk_size) < len(sents_docs):
                if (count == 0):
                    print ("FIRST")
                    temp = embed(sents_docs[count:count+chunk_size])
                    temp_e = sess_embd.run(temp)
                    docs_sents_embds = temp_e
                    print ("DOC: ", (docs_sents_embds).shape)
                else:
                    print ("REMAIN")
                    temp = embed(sents_docs[count:count+chunk_size])
                    temp_e = sess_embd.run(temp)
                    docs_sents_embds = np.concatenate((docs_sents_embds, temp_e), axis=0)
                    print ("DOC: ", (docs_sents_embds).shape)
                count = count + chunk_size
                
            if count < len(sents_docs):
                temp = embed(sents_docs[count:])
                temp_e = sess_embd.run(temp)
                docs_sents_embds = np.concatenate((docs_sents_embds, temp_e), axis=0)
                print ("DOC: ", (docs_sents_embds).shape)

                
            count = 0 
            chunk_size_query = 2000
            while (count + chunk_size_query) < len(sents_queries):
                if (count == 0):
                    print ("FIRST")
                    temp = embed(sents_queries[count:count+chunk_size_query])
                    temp_e = sess_embd.run(temp)
                    queries_sents_embds = temp_e
                    print ("QUERY: ", (queries_sents_embds).shape)

                else:
                    print ("REMAIN")
                    temp = embed(sents_queries[count:count+chunk_size_query])
                    temp_e = sess_embd.run(temp)
                    queries_sents_embds = np.concatenate((queries_sents_embds, temp_e), axis=0)
                    print ("QUERY: ", (queries_sents_embds).shape)

                count = count + chunk_size_query
                
            if count < len(sents_queries):
                temp = embed(sents_queries[count:])
                temp_e = sess_embd.run(temp)
                queries_sents_embds = np.concatenate((queries_sents_embds, temp_e), axis=0)
                print ("QUERY: ", (queries_sents_embds).shape)
                
            
            return docs_sents_embds, queries_sents_embds
            
if __name__=='__main__':
    args = load_args()
    data_train = load_text('train', args.train_data_dir)
    data_val = load_text('val', args.val_data_dir)
    data_test = load_text('test', args.test_data_dir)
    data_docs = data_train[0] + data_val[0] + data_test[0]
    data_queries = data_train[1] + data_val[1] + data_test[1]
    
    data_docs_sents = []
    for doc in data_docs:
        for sent in sent_tokenize(doc):
            data_docs_sents.append(sent)
            
    data_queries_sents = []
    for query in data_queries:
        for sent in sent_tokenize(query):
            data_queries_sents.append(sent)
            
    
    docs_sents_embds, queries_sents_embds =  get_embeddings_sentence_encoder(data_docs_sents, data_queries_sents)   
    store_sentence_embeddings(docs_sents_embds, queries_sents_embds)