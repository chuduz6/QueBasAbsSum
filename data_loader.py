# coding: utf-8

import json
import io
#import nltk
#nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from args import load_args
from vocabulary import load_dictionaries, encode_word, decode_index
import pickle
import numpy as np
from utils import *

MARK_PAD = "<PAD>"
MARK_UNK = "<UNK>"
MARK_EOS = "<EOS>"
MARK_GO = "<GO>"
MARKS = [MARK_PAD, MARK_UNK, MARK_EOS, MARK_GO]
ID_PAD = 0
ID_UNK = 1
ID_EOS = 2
ID_GO = 3

# We use a number of buckets for sampling
_buckets_ms_macro = [(300,5,15), (300,10,20), (300,15,25), (300,20,40), (300,40,50), (350,5,15), (350,10,20), (350,15,25), (350,20,40), (350,40,50), (450,5,15), (450,10,30), (450,15,50), (450,20,100), (450,40,150), (550,5,15), (550,10,30), (550,15,60), (550,20,100), (550,40,150), (650,5,15), (650,10,30), (650,15,60), (650,20,100), (650,40,150), (750,5,15), (750,10,30), (750,15,60), (750,20,100), (750,40,240), (850,5,15), (850,10,30), (850,15,60), (850,20,100), (850,40,240), (1050,5,15), (1050,10,30), (1050,15,60), (1050,20,100), (1050,40,240), (1500,5,15), (1500,10,30), (1500,15,60), (1500,20,100), (1500,40,300), ]


_buckets = [(10, 10, 10), (50, 5, 5), (50, 8, 8), (50, 10, 10), (50, 15, 10), (50, 20, 10), (50, 25, 15), (60, 5, 5), (60, 8, 8), (60, 10, 10), (60, 15, 10), (60, 20, 10), (60, 25, 15), (70, 5, 5), (70, 8, 8), (70, 10, 10), (70, 15, 10), (70, 20, 10), (70, 25, 15), (80, 5, 5), (80, 8, 8), (80, 10, 10), (80, 15, 10), (80, 20, 10), (80, 25, 15), (90, 5, 5), (90, 8, 8), (90, 10, 10), (90, 15, 10), (90, 20, 10), (90, 25, 15), (100, 5, 5), (100, 8, 8), (100, 10, 10), (100, 15, 10), (100, 20, 10), (100, 25, 15), (110, 5, 5), (110, 8, 8), (110, 10, 10), (110, 15, 10), (110, 20, 10), (110, 25, 15), (120, 5, 5), (120, 8, 8), (120, 10, 10), (120, 15, 10), (120, 20, 10), (120, 25, 15), (130, 5, 5), (130, 8, 8), (130, 10, 10), (130, 15, 10), (130, 20, 10), (130, 25, 15), (140, 5, 5), (140, 8, 8), (140, 10, 10), (140, 15, 10), (140, 20, 10), (140, 25, 15), (150, 5, 5), (150, 8, 8), (150, 10, 10), (150, 15, 10), (150, 20, 10), (150, 25, 15),]

def create_bucket(data):
    data_set = [[] for _ in _buckets]
    counter = 0
    for s, q, t in (data):
        found = False
        for bucket_id, (s_size, q_size, t_size) in enumerate(_buckets):
            if (s[1]) <= s_size and (q[1]) <= q_size and (t[1]) <= t_size:
                data_set[bucket_id].append([s, q, t])
                found = True
                break
        '''
        if(found != True):
            counter+=1
            print ("Didn't find bucket for {}, {}, {}".format(s[1], q[1], q[1]))
        '''
    print ("BUCKET NOT FOUND: ", counter)
    return data_set
    
def get_test_data(mode, data_dir, dicts, doc_count, query_count):
    
    docs, queries, summaries = load_text(mode, data_dir)
    
    print ("TOKENIZING DATA ........")        
    docs_seq_indices = []
    docs_words_len = []
    docs_words = []
    docs_words_chars = []
    doc_ind = doc_count
    for doc in docs:
        doc_seq_indices = []
        word_only = []
        words_chars_only = []
        for sent in sent_tokenize(doc):
            temp_word = []
            for word in word_tokenize(sent):
                temp_char = []
                temp_word.append(word.lower())
                word_only.append(word.lower())
                for char in word.lower():
                    temp_char.append(encode_word(dicts[1][0], char))
                words_chars_only.append(temp_char)
            doc_seq_indices.append([doc_ind]*len(temp_word))
            doc_ind += 1
        docs_seq_indices.append(doc_seq_indices)
        docs_words_len.append(len(word_only))
        docs_words.append(word_only)
        docs_words_chars.append(words_chars_only)
        
    docs_words_ids_ext, doc_oovs = docs2ids_ext(args, dicts, docs_words)
    docs_words_ids = docs2ids(dicts, docs_words)
    
    docs_all = list(zip(docs_seq_indices, docs_words_len, docs_words_ids, docs_words_chars, docs_words_ids_ext, doc_oovs, docs_words))

    queries_only = []
    queries_seq_indices = []
    queries_words_len = []
    queries_words = []
    queries_words_chars = []
    query_ind = query_count
    for query in queries:
        queries_only.append(query)
        query_seq_indices = []
        word_only = []
        words_chars_only = []
        for sent in sent_tokenize(query):
            temp_word = []
            for word in word_tokenize(sent):
                temp_char = []
                temp_word.append(word.lower())
                word_only.append(word.lower())
                for char in word.lower():
                    temp_char.append(encode_word(dicts[1][0], char))
                words_chars_only.append(temp_char)
            query_seq_indices.append([query_ind]*len(temp_word))
            query_ind += 1
        queries_seq_indices.append(query_seq_indices)
        queries_words_len.append(len(word_only))
        queries_words.append(word_only)
        queries_words_chars.append(words_chars_only)
        
    queries_words_ids = docs2ids(dicts, queries_words)
    
    queries_all = list(zip(queries_seq_indices, queries_words_len, queries_words_ids, queries_words_chars, queries_only))
    
            
    summaries_only = []
    for summary in summaries:
        summaries_only.append(summary)
    
    summaries_all = summaries_only     
    
    print ("DONE TOKENIZING DATA ........") 
    
    store_test_dataid_pickle(docs_all, queries_all, summaries_all, doc_ind, query_ind)

    return docs_all, queries_all, summaries_all, doc_ind, query_ind

def docs2ids_ext(args, dicts, docs_words):
    all_oovs = []
    all_ids = []
    dict_size = len(dicts[0][1])
    print ("DICT SIZE: ", dict_size)
    for doc_words in docs_words:
        oovs = []
        ids = []
        for word in doc_words:
            i = encode_word(dicts[0][0], word)
            if i == ID_UNK:
                if word not in oovs:
                    oovs.append(word)
                oov_num = oovs.index(word)
                ids.append(dict_size + oov_num)
            else:
                ids.append(i)
        all_ids.append(ids)
        all_oovs.append(oovs)
    
    return all_ids, all_oovs  
    
def docs2ids(dicts, docs_words):
    all_ids = []
    for doc_words in docs_words:
        ids = []
        for word in doc_words:            
            ids.append(encode_word(dicts[0][0], word))
        all_ids.append(ids)    
    return all_ids
    
def summaries2ids (args, dicts, summaries_words, all_oovs):
    all_ids = []
    all_pointer_switches = []
    dict_size = len(dicts[0][1])
    for j, summary_words in enumerate (summaries_words):
        ids = []
        pointer_switches = []
        #print ("DOCS OOVS: ", all_oovs[j])
        for word in summary_words:
            i = encode_word(dicts[0][0], word)
            if i == ID_UNK:
                pointer_switches.append(1)
                if word in all_oovs[j]:
                    vocab_idx = dict_size + all_oovs[j].index(word)
                    ids.append(vocab_idx)
                    #print ("MATCHED WORD: ******************************", word)
                else:
                    #print ("UNMATCHED WORD: ", word)
                    ids.append(ID_UNK)
            else:
                pointer_switches.append(0)
                ids.append(i)
        all_pointer_switches.append(pointer_switches)
        all_ids.append(ids)        
    return all_ids, all_pointer_switches
    
def outputids2words(args, id_list, dicts, article_oovs):
  
    words = []
    for i in id_list:
        try:
            w = decode_index(dicts[1], i) # might be [UNK]
        except KeyError as e: # w is OOV
            assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
            article_oov_idx = i - args.doc_vocab_size
            try:
                w = article_oovs[article_oov_idx]
            except KeyError as e: # i doesn't correspond to an article oov
                raise KeyError('Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (i, article_oov_idx, len(article_oovs)))
        words.append(w)
    return words                
   
def get_train_val_data(args, mode, data_dir, dicts, doc_count, query_count, summary_count):

    docs, queries, summaries = load_text(mode, data_dir)
    
    print ("TOKENIZING DATA ........")        
    docs_seq_indices = []
    docs_words_len = []
    docs_words = []
    docs_words_chars_ids = []
    doc_ind = doc_count
    for doc in docs:
        doc_seq_indices = []
        words_only = []
        words_chars_only = []
        for sent in sent_tokenize(doc):
            temp_word = []
            for word in word_tokenize(sent):
                temp_char = []
                temp_word.append(word.lower())
                words_only.append(word.lower())
                for char in word.lower():
                    temp_char.append(encode_word(dicts[1][0], char))
                words_chars_only.append(temp_char)
            doc_seq_indices.append([doc_ind]*len(temp_word))
            doc_ind += 1
        docs_seq_indices.append(doc_seq_indices)
        docs_words_len.append(len(words_only))
        docs_words.append(words_only)
        docs_words_chars_ids.append(words_chars_only)
        
    docs_words_ids_ext, docs_oovs = docs2ids_ext(args, dicts, docs_words)
    docs_words_ids = docs2ids(dicts, docs_words)
    
    print ("DOC SEQ INDICES: ", docs_seq_indices[0])
    print ("DOC LENGTH: ", docs_words_len[0])    
    print ("DOC WORD ID REGULAR: ", docs_words_ids[0])
    print ("DOC WORD ID EXT: ", docs_words_ids_ext[0])
    print ("DOC WORD CHARS IDS: ", docs_words_chars_ids[0])
    print ("DOC OOVS: ", docs_oovs[0])
    
    
    docs_all = list(zip(docs_seq_indices, docs_words_len, docs_words_ids, docs_words_chars_ids, docs_words_ids_ext, docs_oovs))

    queries_seq_indices = []
    queries_words_len = []
    queries_words = []
    queries_words_chars = []
    query_ind = query_count
    for query in queries:
        query_seq_indices = []
        words_only = []
        words_chars_only = []
        for sent in sent_tokenize(query):
            temp_word = []
            for word in word_tokenize(sent):
                temp_char = []
                temp_word.append(word.lower())
                words_only.append(word.lower())
                for char in word.lower():
                    temp_char.append(encode_word(dicts[1][0], char))
                words_chars_only.append(temp_char)
            query_seq_indices.append([query_ind]*len(temp_word))
            query_ind += 1
        queries_seq_indices.append(query_seq_indices)
        queries_words_len.append(len(words_only))
        queries_words.append(words_only)
        queries_words_chars.append(words_chars_only)
        
    queries_words_ids = docs2ids(dicts, queries_words)
    
    print ("QUERY SEQ INDICES: ", queries_seq_indices[0])
    print ("QUERY LENGTH: ", queries_words_len[0])    
    print ("QUERY WORD REGULAR: ", queries_words_ids[0])
    print ("QUERY WORD CHARS: ", queries_words_chars[0])
    
    queries_all = list(zip(queries_seq_indices, queries_words_len, queries_words_ids, queries_words_chars))
            
    summaries_seq_indices = []
    summaries_words = []
    summaries_words_len = []
    summary_ind = summary_count
    for summary in summaries:
        summary_seq_indices = []
        words_only = [MARK_GO]
        for sent in sent_tokenize(summary):
            temp_word = []
            for word in word_tokenize(sent):
                temp_word.append(word.lower())
                words_only.append(word.lower())
            summary_seq_indices.append([summary_ind]*len(temp_word))
            summary_ind += 1
        words_only.append(MARK_EOS)
        #print ("WORDS ONLY: ", words_only)
        summaries_words.append(words_only)
        summaries_seq_indices.append(summary_seq_indices)
        summaries_words_len.append(len(words_only)-1)
    summaries_words_ids = docs2ids(dicts, summaries_words)    
    summaries_words_ids_ext, summaries_pointer_switches = summaries2ids(args, dicts, summaries_words, docs_oovs)
    print ("SUMMARY SEQ INDICES: ", summaries_seq_indices[0])
    print ("SUMMARY LENGTH: ", summaries_words_len[0])    
    print ("SUMMARY WORD REGULAR: ", summaries_words_ids[0])
    print ("SUMMARY WORD EXT: ", summaries_words_ids_ext[0])
    
    summaries_all = list(zip(summaries_seq_indices, summaries_words_len, summaries_words_ids, summaries_words_ids_ext, summaries_pointer_switches))       
    
    print ("DONE TOKENIZING DATA ........") 
    
    if(mode == 'train'):
        store_train_dataid_pickle(docs_all, queries_all, summaries_all, doc_ind, query_ind, summary_ind)
    elif (mode == 'val'):
        store_val_dataid_pickle(docs_all, queries_all, summaries_all, doc_ind, query_ind, summary_ind)
    else:    
        raise NotImplementedError
        
    return docs_all, queries_all, summaries_all, doc_ind, query_ind, summary_ind
    
'''   

def store_tokenized_train_data(docs, queries, summaries, sent_docs, sent_queries, sent_summaries, sent_embds):
    store_pickle((docs, queries, summaries, sent_docs, sent_queries, sent_summaries, sent_embds), 'tokenized_train_data.pickle')

def load_tokenized_train_data():
    return load_pickle('tokenized_train_data.pickle')
    
def store_tokenized_val_data(docs, queries, summaries, sent_docs, sent_queries, sent_summaries, sent_embds):
    store_pickle((docs, queries, summaries, sent_docs, sent_queries, sent_summaries, sent_embds), 'tokenized_val_data.pickle')

def load_tokenized_val_data():
    return load_pickle('tokenized_val_data.pickle')
    
def store_tokenized_test_data(docs, queries, sent_docs, sent_queries, sent_embds):
    store_pickle((docs, queries, sent_docs, sent_queries, sent_embds), 'tokenized_test_data.pickle')

def load_tokenized_test_data():
    return load_pickle('tokenized_test_data.pickle')
    
'''
    
def store_train_dataid_pickle(docs_all, queries_all, summaries_all, doc_ind, query_ind, summary_ind):
    store_pickle((docs_all, queries_all, summaries_all, doc_ind, query_ind, summary_ind), 'train_dataid.pickle')

def load_train_dataid_pickle():
    return load_pickle('train_dataid.pickle')
    
def store_val_dataid_pickle(docs_all, queries_all, summaries_all, doc_ind, query_ind, summary_ind):
    store_pickle((docs_all, queries_all, summaries_all, doc_ind, query_ind, summary_ind), 'val_dataid.pickle')

def load_val_dataid_pickle():
    return load_pickle('val_dataid.pickle')
    
def store_test_dataid_pickle(docs_all, queries_all, summaries_all, doc_ind, query_ind):
    store_pickle((docs_all, queries_all, summaries_all, doc_ind, query_ind), 'test_dataid.pickle')

def load_test_dataid_pickle():
    return load_pickle('test_dataid.pickle')
     
def add_pad(data, fixlen):
    data = map(lambda x: x + [ID_PAD] * (fixlen - len(x)), data)
    data = list(data)
    return np.asarray(data)

def add_pad_sents_words_chars(data, batch_size, num_sents, sent_size, max_word_size):
    omi9ted_char_count = 0
    data = np.array(data)
    ret = np.zeros([batch_size, num_sents, sent_size, max_word_size], dtype='int32')
    for i, doc in enumerate(data):
        for j, sent in enumerate(doc[0]):
            for k, word in enumerate(sent):
                for l, char in enumerate(word):
                    if (l < max_word_size):
                        ret[i,j,k,l] = char
                    else:
                        omi9ted_char_count += 1
    #print ("Omitted Character's Count: ", omi9ted_char_count)
    for i in range(batch_size):
        data[i][0] = ret[i]
    return data
    
def add_pad_sents_words(data, batch_size, num_sents, sent_size):
    data = np.array(data)
    ret = np.zeros([batch_size, num_sents, sent_size], dtype='int32')
    count = 0
    for i, doc in enumerate(data):
        for j, sent in enumerate(doc[1]):
            for k, word in enumerate(sent):
                ret[i,j,k] = word
    for i in range(batch_size):
        data[i][1] = ret[i]
    return data

def add_pad_words_chars(data,batch_size, max_words, max_word_size, index):
    omi9ted_char_count = 0
    data = np.array(data)
    ret = np.zeros([batch_size, max_words, max_word_size], dtype='int32')
    for i, doc in enumerate(data):
        for j, word in enumerate(doc[index]):
                for k, char in enumerate(word):
                    if (k < max_word_size):
                        ret[i,j,k] = char
                    else:
                        omi9ted_char_count += 1
    #print ("Omitted Character's Count: ", omi9ted_char_count)
    for i in range(batch_size):
        data[i][index] = ret[i]
    return data
    
def add_pad_indices(data, batch_size, max_num_sents, max_sent_size, index):
    data = np.array(data)
    ret = np.zeros([batch_size, max_num_sents, max_sent_size], dtype='int32')
    for i, doc in enumerate(data):
        for j, sent in enumerate(doc[index]):
            for k, indices in enumerate(sent):
                ret[i,j,k] = indices
    
    for i in range(batch_size):
        data[i][index] = ret[i]
    return data            
            
def add_pad_words(data, batch_size, max_words, index):
    data = np.array(data)
    ret = np.zeros([batch_size, max_words], dtype='int32')
    count = 0
    for i, doc in enumerate(data):
        for j, word in enumerate(doc[index]):
            ret[i,j] = word
    for i in range(batch_size):
        data[i][index] = ret[i]
    return data

def add_padding_mask(summary_all, batch_size, max_summaries_words_lengths, index):
    dec_padding_mask = np.zeros((batch_size, max_summaries_words_lengths), dtype=np.float32)
    for i, summary in enumerate(summary_all):
        for j in range(summary[1]):
            dec_padding_mask[i][j] = 1.0
        summary_all[i,index] = dec_padding_mask[i]
    return summary_all
    
def batchify ( data_set, _buckets, batch_size, max_word_size):
    print ("BATCH SIZE IS: ", batch_size)
    batched_data_set = []
    docs_all, queries_all, summaries_all = [], [], []
    docs_words_lengths, queries_words_lengths, summaries_words_lengths = [], [], []
    docs_sents_num, queries_sents_num = [], []
    docs_sents_words_lengths, queries_sents_words_lengths = [], []
    docs_num_oovs = []
    
    num_data = 0
    counter = 0
    for bucket_id in range (len(_buckets)):
        if(len(data_set[bucket_id])==0):
            continue
        for j in range(len(data_set[bucket_id])):
            counter += 1
            doc_all, query_all, summary_all = data_set[bucket_id][j]
            summary_all = list(summary_all)
            summary_all.append(np.zeros(summary_all[1]))
            summary_all = tuple(summary_all)
            
            doc_all = list(doc_all)
            doc_all.append(np.zeros(doc_all[1]))
            doc_all = tuple(doc_all)            
            
            docs_all.append(doc_all)
            docs_words_lengths.append(doc_all[1])
            docs_sents_num.append(len(doc_all[0]))
            docs_sents_words_lengths.append(max(len(sent) for sent in doc_all[0]))
            docs_num_oovs.append(len(doc_all[5]))
            
            queries_all.append(query_all)
            queries_words_lengths.append(query_all[1])
            queries_sents_num.append(len(query_all[0]))
            queries_sents_words_lengths.append(max(len(sent) for sent in query_all[0]))
            
            summaries_all.append(summary_all)
            summaries_words_lengths.append(summary_all[1])
            
            num_data += 1
            
            if(num_data == batch_size):
                num_data = 0
                max_docs_words_lengths = max(docs_words_lengths)
                max_docs_sents_num = max(docs_sents_num)
                max_docs_sents_words_lengths = max(docs_sents_words_lengths)
                max_docs_num_oovs = max(docs_num_oovs)
                
                max_queries_words_lengths = max(queries_words_lengths)
                max_queries_sents_num = max(queries_sents_num)
                max_queries_sents_words_lengths = max(queries_sents_words_lengths)
                
                max_summaries_words_lengths = max(summaries_words_lengths)
                
                docs_all = add_pad_words_chars(docs_all, batch_size, max_docs_words_lengths, max_word_size, 3)
                queries_all = add_pad_words_chars(queries_all, batch_size, max_queries_words_lengths, max_word_size, 3)
                
                docs_all = add_pad_words(docs_all, batch_size, max_docs_words_lengths, 2)
                queries_all = add_pad_words(queries_all, batch_size, max_queries_words_lengths, 2)
                summaries_all = add_pad_words(summaries_all, batch_size, max_summaries_words_lengths + 1, 2) # summaries_words_ids
                summaries_all = add_pad_words(summaries_all, batch_size, max_summaries_words_lengths + 1, 3) # summaries_words_ids_ext
                summaries_all = add_pad_words(summaries_all, batch_size, max_summaries_words_lengths + 1, 4) # summaries_pointer_switches  
                docs_all = add_pad_words(docs_all, batch_size, max_docs_words_lengths, 4)
                summaries_all = add_padding_mask(summaries_all, batch_size, max_summaries_words_lengths, 5) # summary padding
                docs_all = add_padding_mask(docs_all, batch_size, max_docs_words_lengths, 6)

                
                docs_all = add_pad_indices(docs_all, batch_size, max_docs_sents_num, max_docs_sents_words_lengths, 0)
                queries_all = add_pad_indices(queries_all, batch_size, max_queries_sents_num, max_queries_sents_words_lengths, 0)
                
                batched_data_set.append([docs_all, queries_all, summaries_all, max_docs_num_oovs])

                docs_all, queries_all, summaries_all = [], [], []
                docs_words_lengths, queries_words_lengths, summaries_words_lengths = [], [], []
                docs_sents_num, queries_sents_num = [], []
                docs_sents_words_lengths, queries_sents_words_lengths = [], []
                docs_num_oovs = []
                
    print ("TOTAL DOCUMENTS BATCHED: ", counter)
    print ("TOTAL NUMBER OF BATCHES: ", len(batched_data_set))
    return batched_data_set
    
def batchify_test ( data_set, beam_size, max_word_size):
    print ("TEST BATCH SIZE IS: ", beam_size)
    batched_data_set = []
    docs_all, queries_all, summaries_all = [], [], []
    docs_words_lengths, queries_words_lengths = [], []
    docs_sents_num, queries_sents_num = [], []
    docs_sents_words_lengths, queries_sents_words_lengths = [], []
    docs_num_oovs = []

    num_data = 0
    counter = 0
    for data in (data_set):
        counter += 1
        
        doc_all, query_all, summary_all = data
        doc_all = list(doc_all)
        doc_all.append(np.zeros(doc_all[1]))
        doc_all = tuple(doc_all) 
        
        for i in range(beam_size):           
            docs_all.append(doc_all)
            docs_words_lengths.append(doc_all[1])
            docs_sents_num.append(len(doc_all[0]))
            docs_sents_words_lengths.append(max(len(sent) for sent in doc_all[0]))
            docs_num_oovs.append(len(doc_all[5]))
            
            queries_all.append(query_all)
            queries_words_lengths.append(query_all[1])
            queries_sents_num.append(len(query_all[0]))
            queries_sents_words_lengths.append(max(len(sent) for sent in query_all[0]))
                
            summaries_all.append(summary_all)
        
        
        
            
        max_docs_words_lengths = max(docs_words_lengths)
        max_docs_sents_num = max(docs_sents_num)
        max_docs_sents_words_lengths = max(docs_sents_words_lengths)
        max_docs_num_oovs = max(docs_num_oovs)
        
        max_queries_words_lengths = max(queries_words_lengths)
        max_queries_sents_num = max(queries_sents_num)
        max_queries_sents_words_lengths = max(queries_sents_words_lengths)
                 
        docs_all = add_pad_words_chars(docs_all, beam_size, max_docs_words_lengths, max_word_size, 3)
        queries_all = add_pad_words_chars(queries_all, beam_size, max_queries_words_lengths, max_word_size, 3)
        
        docs_all = add_pad_words(docs_all, beam_size, max_docs_words_lengths, 2)
        queries_all = add_pad_words(queries_all, beam_size, max_queries_words_lengths, 2)
        docs_all = add_pad_words(docs_all, beam_size, max_docs_words_lengths, 4)
        
        docs_all = add_pad_indices(docs_all, beam_size, max_docs_sents_num, max_docs_sents_words_lengths, 0)
        queries_all = add_pad_indices(queries_all, beam_size, max_queries_sents_num, max_queries_sents_words_lengths, 0)
        
        docs_all = add_padding_mask(docs_all, beam_size, max_docs_words_lengths, 7)

        batched_data_set.append([docs_all, queries_all, summaries_all, max_docs_num_oovs])

        docs_all, queries_all, summaries_all = [], [], []
        docs_words_lengths, queries_words_lengths, summaries_words_lengths = [], [], []
        docs_sents_num, queries_sents_num = [], []
        docs_sents_words_lengths, queries_sents_words_lengths = [], []
        docs_num_oovs = []
                    
    print ("TOTAL DOCUMENTS BATCHED: ", counter)
    print ("TOTAL NUMBER OF BATCHES: ", len(batched_data_set))
    return batched_data_set
    

    
def batch_iter(docid, queryid, sumid, batch_size, num_epochs):

    docid = np.array(docid)
    queryid = np.array(queryid)
    queryid = np.array(queryid)

    num_batches_per_epoch = (len(docid) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(docid))
            yield docid[start_index:end_index], queryid[start_index:end_index], sumid[start_index:end_index]
            
if __name__=='__main__':            
    args = load_args()
    dicts = load_dictionaries()
    train_docs_all, train_queries_all, train_summaries_all, doc_count, query_count, summary_count = get_train_val_data(args, 'train', args.train_data_dir, dicts, 0, 0, 0)
    print ("train: ", train_docs_all[0][2][0])
    print ("COUNT: ", doc_count)
    print ("LENGTH OF TRAIN: ", len(train_docs_all))
    val_docs_all, val_queries_all, val_summaries_all, doc_count, query_count, summary_count = get_train_val_data(args, 'val', args.val_data_dir, dicts, doc_count, query_count, summary_count)
    print ("train: ", val_docs_all[0][2][0])
    print ("COUNT: ", doc_count)
    print ("LENGTH OF TRAIN: ", len(val_docs_all))
    test_docs_all, test_queries_all, test_summaries_all, doc_count, query_count = get_test_data('test', args.test_data_dir, dicts, doc_count, query_count)
    print ("train: ", test_docs_all[0][2][0])
    print ("COUNT: ", doc_count)
    print ("LENGTH OF TRAIN: ", len(test_docs_all))
    