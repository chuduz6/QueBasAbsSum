import torch
import torch.nn as nn
import math
import tensorflow as tf
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, max_len=5000):
        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2) * -(math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.emb_size = emb_size

    def forward(self, emb):
        # We must wrap the self.pe in Variable to compute, not the other
        # way - unwrap emb(i.e. emb.data). Otherwise the computation
        # wouldn't be watched to build the compute graph.
        emb = emb * math.sqrt(self.emb_size)
        # print(self.pe.size())
        emb = emb + Variable(self.pe[:emb.size(0)], requires_grad=False)
        emb = tf.layers.batch_normalization(conv1d_output, True)    

        return emb