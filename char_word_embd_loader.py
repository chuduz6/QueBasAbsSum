 # coding: utf-8
#from gensim.models.keyedvectors import KeyedVectors
from args import load_args
from vocabulary import load_dictionaries
import numpy as np
import tensorflow as tf
import pickle
import tensorflow_hub as hub
from utils import *

        
def store_char_embedding(char_vec_list):
    store_pickle(char_vec_list, 'char_embedding.pickle')
        
def store_doc_word_embedding(word_vec_list):
    store_pickle(word_vec_list, 'doc_word_embedding.pickle')
        
def store_sum_word_embedding(word_vec_list):
    store_pickle(word_vec_list, 'sum_word_embedding.pickle')
    
def store_word_vectors_pickle(word_vectors):
    store_pickle(word_vectors, 'word_vectors.pickle')

def load_char_embedding():
    return np.array(load_pickle('char_embedding.pickle'))
    
def load_doc_word_embedding():
    return np.array(load_pickle('doc_word_embedding.pickle'))
    
def load_sum_word_embedding():
    return np.array(load_pickle('sum_word_embedding.pickle'))
                
def load_word_vectors_pickle():
    return load_pickle('word_vectors.pickle')
       
def get_word_embedding(word2vec_file, word_vocab, args):
    '''
    if(args.reload_all or args.reload_word_vectors_list):
        word_vectors = KeyedVectors.load_word2vec_format(word2vec_file)
        store_word_vectors_pickle(word_vectors)
    '''
    word_vectors = load_word_vectors_pickle()
    word_vec_list = list()
    success_count = 0
    failure_count = 0
    for word, _ in word_vocab[0].items():
        try:
            word_vec = word_vectors.word_vec(word)
            success_count += 1
        except KeyError:
            word_vec = np.random.normal(0, 1, args.embsize)
            failure_count += 1
            print ("FAILURE WORD: ", word)
        word_vec_list.append(word_vec)

    word_vec_list[2] = np.random.normal(0, 1, args.embsize)
    word_vec_list[3] = np.random.normal(0, 1, args.embsize)
    print ("SUCCESS COUNT: ", success_count, " FAILURE COUNT: ", failure_count)
    
    store_doc_word_embedding(np.array(word_vec_list))

    
def get_char_embedding(char_embd_file, char_dict, char_embd_size):
    initW = np.random.uniform(-0.25, 0.25, (len(char_dict[0]), char_embd_size))
    print("LENGTH OF CHAR DICT: ", len(char_dict[0]))
    print ("CHAR EMBD SIZE: ", char_embd_size)
    with open(char_embd_file, "r") as f:
        for line in f:
            line = line.rstrip().split(' ')
            word, vec = line[0], line[1:]
            for char, index in char_dict[0].items():
                if char == word:
                    initW[index] = np.asarray([float(x) for x in vec]) 
    
    store_char_embedding(initW)
    
if __name__=='__main__':    
    args = load_args()
    dicts = load_dictionaries()
    get_char_embedding(args.pretrained_char_embeddings_file, dicts[1], args.char_embsize)
    get_word_embedding(args.pretrained_embeddings_vec_path, dicts[0], args)
    
    