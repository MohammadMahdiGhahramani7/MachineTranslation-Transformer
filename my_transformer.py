import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math

class IO_Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # x: [batch_size, seq_len_SRC/TRG]
        return self.embed(x) # [batch_size, seq_len_SRC/TRG, d_model]
        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        '''
        Why considering max_seq_len?
          Since seq_len_SRC is not necessarily equal to seq_len_TRG and since
          we want to use this class both for SRC and TRG sentences, we set:
                max_seq_len = MAX(seq_len_SRC, seq_len_TRG - 1).
        '''
        super().__init__()

        self.d_model = d_model
        positional_emb = torch.zeros(max_seq_len, d_model)

        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                positional_emb[pos, i] = math.sin(pos / (10000 ** (i/d_model)))
                positional_emb[pos, i + 1] = math.cos(pos / (10000 ** (i/d_model)))
                
        self.register_buffer('positional_emb', positional_emb)
        self.positional_emb.requires_grad = False
    
    def forward(self, x):
        '''
        x is the embedded vector, coming from the previous class as the output.
        The reason we increase the embedding values before addition is to make the
        positional encoding relatively smaller. This means the original meaning in
        the embedding vector wont be lost when we add them together.
        '''
        # x: [batch_size, seq_len_SRC/TRG, d_model]      
        x = x * math.sqrt(self.d_model)

        _, seq_len, _ = x.size()
        x = x + self.positional_emb[:seq_len, :]
        # self.positional_emb[:seq_len, :]: [seq_len_SRC/TRG, d_model]
        # x:                                [batch_size, seq_len_SRC/TRG, d_model] 

        return x


def attention(q, k, v, d_k, mask=None, dropout=None):
    #(1) q, k, v : [batch_size, N, seq_len_SRC/TRG, d_k]
    #(2) k, v : [batch_size, N, seq_len_SRC, d_k] ---- q: [batch_size, N, seq_len_TRG, d_k]
    #(1) ---> First Attention Layers in Encoder and Decoder
    #(2) ---> Middle Attention Layer in Decoder

    scores = torch.matmul(q, k.permute(0, 1, 3, 2)) /  math.sqrt(d_k)
    #(1) scores: [batch_size, N, seq_len_SRC/TRG, seq_len_SRC/TRG]
    #(2) scores: [batch_size, N, seq_len_TRG, seq_len_SRC]
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    #(1) output: [batch_size, N, seq_len_SRC/TRG, d_k]
    #(2) output: [batch_size, N, seq_len_TRG, d_k]
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.N = heads
        self.d_model = d_model
        self.d_k = d_model // heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        #(1) q, k, v : [batch_size, seq_len_SRC/TRG, d_model]
        #(2) k, v : [batch_size, seq_len_SRC, d_model] ---- q: [batch_size, seq_len_TRG, d_model]
        #(1) ---> First Attention Layers in Encoder and Decoder
        #(2) ---> Middle Attention Layer in Decoder

        batch_size = q.size(0)
                
        k = self.k_linear(k).view(batch_size, -1, self.N, self.d_k).permute(0, 2, 1, 3)
        q = self.q_linear(q).view(batch_size, -1, self.N, self.d_k).permute(0, 2, 1, 3)
        v = self.v_linear(v).view(batch_size, -1, self.N, self.d_k).permute(0, 2, 1, 3)
        #(1) q, k, v : [batch_size, N, seq_len_SRC/TRG, d_k]
        #(2) k, v : [batch_size, N, seq_len_SRC, d_k] ---- q: [batch_size, N, seq_len_TRG, d_k]

        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        #(1) scores: [batch_size, N, seq_len_SRC/TRG, d_k]
        #(2) scores: [batch_size, N, seq_len_TRG, d_k]
        
        concat = scores.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        #(1) concat: [batch_size, seq_len_SRC/TRG, d_model]
        #(2) concat: [batch_size, seq_len_TRG, d_model]
        output = self.out(concat)
        #(1) output: [batch_size, seq_len_SRC/TRG, d_model]
        #(2) output: [batch_size, seq_len_TRG, d_model]
    
        return output


class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.d_model = d_model
        self.eps = eps

        self.Gamma = nn.Parameter(torch.ones(self.d_model)) #learnable
        self.Beta = nn.Parameter(torch.zeros(self.d_model)) #learnable
        

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        mio = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)

        x_hat = (x - mio) / (torch.sqrt(var + self.eps))
        y = self.Gamma * x_hat + self.Beta
        # y: [batch_size, seq_len, d_model]
        return y
        
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 

        self.lin1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.lin2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = self.dropout(F.relu(self.lin1(x)))
        # x: [batch_size, seq_len, d_ff]
        x = self.lin2(x)
        # x: [batch_size, seq_len, d_model]
        return x


class SingleEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout = 0.1):

        super().__init__()

        self.norm1 = Norm(d_model)
        self.norm2 = Norm(d_model)
        self.attention = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.drp1 = nn.Dropout(dropout)
        self.drp2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # x: [batch_size, seq_len_SRC, d_model]
        x_copied = x
        x = self.attention(x, x, x, mask) # Attention
        # x: [batch_size, seq_len_SRC, d_model]
        x = self.norm1(x_copied + self.drp1(x)) # Add & Norm
        # x: [batch_size, seq_len_SRC, d_model]
        
        x_copied = x
        x = self.ff(x) # Feed forward
        # x: [batch_size, seq_len_SRC, d_model]
        x = self.norm2(x_copied + self.drp2(x)) # Add & Norm
        # x: [batch_size, seq_len_SRC, d_model]
        return x


class SingleDecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):

        super().__init__()

        self.norm1 = Norm(d_model)
        self.norm2 = Norm(d_model)
        self.norm3 = Norm(d_model)
        
        self.drp1 = nn.Dropout(dropout)
        self.drp2 = nn.Dropout(dropout)
        self.drp3 = nn.Dropout(dropout)
        
        self.attention1 = MultiHeadAttention(heads, d_model)
        self.attention2 = MultiHeadAttention(heads, d_model)

        self.ff = FeedForward(d_model)

    def forward(self, y, enc, src_mask, trg_mask):
        # y: [batch_size, seq_len_TRG, d_model]
        y_copied = y
        y = self.attention1(y, y, y, trg_mask) # Attention: Bottom
        y = self.norm1(y_copied + self.drp1(y)) # Add & Norm
        # y: [batch_size, seq_len_TRG, d_model]

        # enc: [batch_size, seq_len_SRC, d_model]
        enc = self.attention2(y, enc, enc, src_mask) # Attention: Middle
        # enc: [batch_size, seq_len_TRG, d_model] ---> (2)
        enc = self.norm2(y + self.drp2(enc)) # Add & Norm : Very important
        # enc: [batch_size, seq_len_TRG, d_model] ---> (2)

        enc_copied = enc
        enc = self.ff(enc) # Feed forward: Up
        # enc: [batch_size, seq_len_TRG, d_model]
        out = self.norm3(enc_copied + self.drp3(enc)) # Add & Norm
        # out: [batch_size, seq_len_TRG, d_model]

        return out
        

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, max_seq_len):

        super().__init__()

        self.N = N # how many encoding layer
        self.emb = IO_Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList([SingleEncoderLayer(d_model, heads) for _ in range(N)])

    def forward(self, src, mask):
        # src: [batch_size, seq_len_SRC]
        x = self.emb(src)
        # x: [batch_size, seq_len_SRC, d_model]
        x = self.pe(x)
        # x: [batch_size, seq_len_SRC, d_model]

        for i in range(self.N):
            x = self.layers[i](x, mask)
        # x: [batch_size, seq_len_SRC, d_model]
        return x
        
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, max_seq_len):

        super().__init__()

        self.N = N
        self.emb = IO_Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList([SingleDecoderLayer(d_model, heads) for _ in range(N)])

    def forward(self, trg, enc, src_mask, trg_mask):
        # trg: [batch_size, seq_len_TRG]
        x = self.emb(trg)
        # x: [batch_size, seq_len_TRG, d_model]
        x = self.pe(x)
        # x: [batch_size, seq_len_TRG, d_model]

        for i in range(self.N):
            x = self.layers[i](x, enc, src_mask, trg_mask)
        # x: [batch_size, seq_len_TRG, d_model]
        return x
        

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, max_seq_len):

        super().__init__()

        self.encoder = Encoder(src_vocab, d_model, N, heads, max_seq_len)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, max_seq_len)

        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask=None, trg_mask=None):
 
        # src: [batch_size, seq_len_SRC]
        # trg: [batch_size, seq_len_TRG]
        enc = self.encoder(src, src_mask)
        # enc: [batch_size, seq_len_SRC, d_model]
        dec = self.decoder(trg, enc, src_mask, trg_mask)
        # dec: [batch_size, seq_len_TRG, d_model]
        output = self.out(dec)
        # output: [batch_size, seq_len_TRG, trg_vocab]
        return output

