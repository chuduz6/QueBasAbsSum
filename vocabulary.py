# coding: utf-8
from args import load_args
import io
import json
from nltk.tokenize import sent_tokenize, word_tokenize
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

def create_char_dict(dict_path, words, max_vocab=None):
    print("Creating Character dict {}....".format(dict_path))
    counter = {}
    for word in words:
        for char in word:
            try:
                counter[char] += 1
            except:
                counter[char] = 1
    counter = list(counter.items())
    counter.sort(key=lambda x: -x[1])
    characters = list(map(lambda x: x[0], counter))
    characters = [MARK_PAD, MARK_UNK, MARK_EOS, MARK_GO] + characters

    if max_vocab:
        characters = characters[:max_vocab]

    tok2id = dict()
    id2tok = dict()
    with open(dict_path, 'w') as dict_file:
        for idx, tok in enumerate(characters):
            print(idx, tok, file=dict_file)
            tok2id[tok] = idx
            id2tok[idx] = tok

    print("Create dict {} with {} characters.".format(dict_path, len(characters)))
    return (tok2id, id2tok)
    

def load_dict(dict_path, max_vocab=None):
    print("Try load dict from {}.".format(dict_path))
    try:
        dict_file = open(dict_path)
        dict_data = dict_file.readlines()
        dict_file.close()
    except:
        print("Load dict {dict} failed, create later.".format(dict=dict_path))
        return None

    dict_data = list(map(lambda x: x.split(), dict_data))
    if max_vocab:
        dict_data = list(filter(lambda x: int(x[0]) < max_vocab, dict_data))
    tok2id = dict(map(lambda x: (x[1], int(x[0])), dict_data))
    id2tok = dict(map(lambda x: (int(x[0]), x[1]), dict_data))
    print("Load dict {} with {} words.".format(dict_path, len(tok2id)))
    return (tok2id, id2tok)


def create_dict(dict_path, corpus, max_vocab=None):
    print("Create dict {}.".format(dict_path))
    counter = {}
    counter2 = 0
    for line in corpus:
        for word in word_tokenize(line):
            counter2 += 1
            try:
                counter[word.lower()] += 1
            except:
                counter[word.lower()] = 1
    print ("TOTAL SET OF WORDS: ", counter2)
    for mark_t in MARKS:
        if mark_t in counter:
            del counter[mark_t]
            logging.warning("{} appears in corpus.".format(mark_t))

    counter = list(counter.items())
    counter.sort(key=lambda x: -x[1])
    
    for k, cntr in enumerate(counter):
        #print ("K: :", k, " CNTR: ", cntr)
        if cntr[1] < 4:
            del counter[k]
            
    words = list(map(lambda x: x[0], counter))
    words = [MARK_PAD, MARK_UNK, MARK_EOS, MARK_GO] + words
    if max_vocab:
        words = words[:max_vocab]

    tok2id = dict()
    id2tok = dict()
    with open(dict_path, 'w') as dict_file:
        for idx, tok in enumerate(words):
            print(idx, tok, file=dict_file)
            tok2id[tok] = idx
            id2tok[idx] = tok

    print("Create dict {} with {} words.".format(dict_path, len(words)))
    return (tok2id, id2tok)


def corpus_map2id(data, tok2id):
    ret = []
    unk = 0
    tot = 0
    for doc in data:
        tmp = []
        for word in doc:
            tot += 1
            try:
                tmp.append(tok2id[word])
            except:
                tmp.append(ID_UNK)
                unk +=1
        ret.append(tmp)
    print ("TOTAL :", tot, " UNK :", unk)
    return ret, (tot - unk)/tot
    
def corpus_map2id_c(data, tok2id):
    ret = []
    unk = 0
    tot = 0
    for doc in data:
        tmp = []
        for word in doc:
            tmp2 = []
            for char in word:
                try:
                    tmp2.append(tok2id[char])
                except:
                    tmp2.append(ID_UNK)
                    unk +=1
            tmp.append(tmp2)        
        ret.append(tmp)
    return ret
    
def sen_map2tok(sen, id2tok):
    return list(map(lambda x: id2tok[x], sen))

def encode_word(tok2id, word):
    if word not in tok2id:
        word = MARK_UNK   
    return tok2id[word]
    
def decode_index(id2tok, index):
    return id2tok[index]
    
def store_dictionaries(doc_dict, char_dict):
    store_pickle((doc_dict, char_dict), 'load_dictionaries.pickle')
    
def load_dictionaries():
    return load_pickle('load_dictionaries.pickle')

if __name__=='__main__':  
    
    args = load_args()
    data_train = load_text('train', args.train_data_dir)
    data_val = load_text('val', args.val_data_dir)
    data_test = load_text('test', args.test_data_dir)
    data_doc = data_train[0] + data_train[1] + data_val[0] + data_val[1] + data_test[0] + data_test[1]
    data_sum = data_train[2] + data_val[2] + data_test[2]
    doc_dict = create_dict(args.doc_dict_path, data_doc + data_sum, args.doc_vocab_size)
    char_dict = create_char_dict(args.char_dict_path, list(doc_dict[0].keys()), args.char_vocab_size)
    
    store_dictionaries(doc_dict, char_dict)
    
    #temp = [self.vocab.encode_word_encoder(word) for word in lines.split()]
    '''
    doc_dict = load_dict('doc_dict.txt', 100)
    print (encode_word(doc_dict[0], 'in'))
    print (decode_index(doc_dict[1], 11))
    '''
