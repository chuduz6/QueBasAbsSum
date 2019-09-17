 # coding: utf-8
 
import tensorflow as tf
import pickle
import argparse

def add_arguments(parser):
    # Dictionary parameters
    parser.add_argument("--doc_dict_path", type=str, default='doc_dict.txt', help="Document Dictionary Path")
    parser.add_argument("--char_dict_path", type=str, default='char_dict.txt', help="Summary Dictionary Path")
    
    parser.add_argument("--create_dict_flag", type=bool, default=True, help="Create Dictionary Flag")
    parser.add_argument("--doc_vocab_size", type=int, default=75000, help="Document Vocabulary Size.")
    parser.add_argument("--char_vocab_size", type=int, default=150, help="Summary Vocabulary size.")
    parser.add_argument("--max_word_size", type=int, default=20, help="Max Word Size.")
    
    parser.add_argument("--train_test_split", type=float, default=0.33, help="Test Split ratio")
    parser.add_argument("--load_pretrained_embeddings", type=bool, default=True, help="Load Pretrained Embedding Flag")
    parser.add_argument("--pretrained_embeddings_vec_path", type=str, default='wiki.en.vec', help="Pretrained Embeddings File")
    parser.add_argument("--pretrained_char_embeddings_file", type=str, default='glove.840B.300d-char.txt', help="Pretrained Charcter Embeddings File")

    parser.add_argument("--reload_all", type=bool, default=False, help="Reload All")
    parser.add_argument("--reload_train_val_data", type=bool, default=False, help="Reload Data")
    parser.add_argument("--reload_test_data", type=bool, default=False, help="Reload Data")

    parser.add_argument("--reload_word_vectors_list", type=bool, default=False, help="Reload Word Vecors List")
    parser.add_argument("--stack_resolve", type=str, default='dum', help="How to resolve stack, option sum or other ways")
    


    # Optimization Parameters
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning Rate") # ***********************************
    parser.add_argument("--momentum", type=float, default=0.9, help="Learning Rate") # ***********************************
    parser.add_argument("--size", type=int, default=400, help="Number of Hidden Size of Cell")
    parser.add_argument("--embsize", type=int, default=300, help="Embedding Size")
    parser.add_argument("--char_embsize", type=int, default=300, help="Embedding Size")

    parser.add_argument("--num_layers", type=int, default=3, help="Number of Layers")
    parser.add_argument("--highway_num_layers", type=int, default=2, help="Number of Layers")

    parser.add_argument("--max_gradient", type=float, default=2.0, help="Max Gradient for Clipping")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Train Batch Size") # ***********************************
    parser.add_argument("--test_batch_size", type=int, default=1, help="Test Batch Size.")
    parser.add_argument("--beam_width", type=int, default=12, help="Beam Width for Inference Decoding. Beam Search Decoder")
    parser.add_argument("--max_epochs", type=int, default=250, help="Max Epochs")
    parser.add_argument("--doc_encoder_keep_prob", type=float, default=0.8, help="Max Gradient for Clipping")
    parser.add_argument("--query_encoder_keep_prob", type=float, default=0.8, help="Max Gradient for Clipping")
    parser.add_argument("--decoder_keep_prob", type=float, default=0.8, help="Max Gradient for Clipping")
    parser.add_argument("--doc_encoder_dropout_flag", type=bool, default=True, help="Dropout Flag for Document Encoder")
    parser.add_argument("--query_encoder_dropout_flag", type=bool, default=True, help="Dropout Flag for Query Encoder")
    parser.add_argument("--decoder_dropout_flag", type=bool, default=True, help="Dropout Flag for Decoder")
    parser.add_argument("--beam_max_iterations", type=int, default=50, help="Max Iterations for Beam Search Decoder")
    parser.add_argument("--max_to_keep", type=int, default=100, help="Max Savers to Save")


    parser.add_argument("--train_forward_only", type=bool, default=True, help="")
    parser.add_argument("--test_forward_only", type=bool, default=True, help="")


    # Data Directory Paramters
    parser.add_argument("--train_data_dir", type=str, default='data2/5', help="Training Data Path")  # ***********************************
    parser.add_argument("--val_data_dir", type=str, default='data2/5', help="Validation Data Path") # *************************************
    parser.add_argument("--test_data_dir", type=str, default='data2/5', help="Test Data Path")

    parser.add_argument("--sample_data_dir", type=str, default='data_sample_train.json', help="Sample Data Path")

    # Output Data Directory Parameters
    parser.add_argument("--test_output", type=str, default='test_output.txt', help="Test Output")
    parser.add_argument("--train_dir", type=str, default='model', help="Training Directory")
    parser.add_argument("--tfboard", type=str, default='tfboard', help="Tensorboard Log Directory")
    parser.add_argument("--steps_per_print", type=int, default=100, help="") # ***********************************
    parser.add_argument("--steps_per_validation", type=int, default=10, help="")
    parser.add_argument("--steps_per_checkpoint", type=int, default=10, help="")
    parser.add_argument("--toy", type=int, default=0, help="")
    
    parser.add_argument("--mode", type=str, default='train', help="Tensorboard Log Directory")
    parser.add_argument("--share_cnn_weights", type=bool, default=True, help="")
    parser.add_argument("--out_channel_dims", type=str, default='300', help="Test Output")
    parser.add_argument("--filter_heights", type=str, default='5', help="Test Output")
    

    parser.add_argument("--feed_previous", type=bool, default=False, help="") # ***********************************
    parser.add_argument("--feed_previous_epoch", type=int, default=25000, help="") # ***********************************

    parser.add_argument("--train_load_checkpoint", type=bool, default=True, help="") # ***********************************
    parser.add_argument("--test_load_checkpoint", type=bool, default=True, help="")

    parser.add_argument("--decode", type=bool, default=False, help="")
    parser.add_argument("--geneos", type=bool, default=True, help="")

    parser.add_argument("--seed", type=int, default=143, help="")


def dump_args(args):
    with open("args.pickle", "wb") as f:
        pickle.dump(args, f)
        print ("Done Dumping Args Pickle")


def load_args():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    dump_args(args)
    with open('args.pickle', 'rb') as handle:
        args = pickle.load(handle)        
    return args

#args = load_args()   
#print (args.full_data_dir) 