import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import math

def attention(query, key, value, mask=None): #最后两维度：T,D
    d_k = key.shape[-1]
    scores = torch.matmul(query, key.transpose(-2, -1)) #matmul广播，对最后两个维度处理 N,T,T
    scores = scores / np.sqrt(d_k) #N,T,T
    #attn = torch.matmul(torch.softmax(scores), value)
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)

    weight = F.softmax(scores, dim=-1)
    return torch.matmul(weight, value)

class MultiHeadAttention(nn.Module):
    '''Q, K, V 的形状从(batch_size, max_sequence_length, dim_embed)变为(batch_size, num_heads, max_sequence_length, dim_head)，'''
    #todo:实际上这里没有max_sequence_length， 就是64,784
    def __init__(self, d_model, num_heads, dropout=0.1) -> None:
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.w_key = nn.Linear(d_model, d_model)
        self.w_query = nn.Linear(d_model, d_model)
        self.w_value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        #self.atten = None

    def forward(self, x, y, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1) # head导致query等多了一维
            '''mask.unsqueeze(1)的目的是给mask增加一个维度， 这样不同的头使用相同的mask， 在attention这个函数中， masked_fill可以实现这个目的。'''
        #print("IN attention x,y: ", x.shape, y.shape)
        query = self.w_query(x)
        key = self.w_key(y)
        value = self.w_value(y)

        #print("Q,K,V", query.shape, key.shape, value.shape)

        batchsize = query.shape[0]

        # query = query.view(batchsize, -1, self.num_heads, self.d_k).transpose(1, 2)
        # key = key.view(batchsize, -1, self.num_heads, self.d_k).transpose(1, 2)
        # value = value.view(batchsize, -1, self.num_heads, self.d_k).transpose(1, 2)
        #todo 貌似应该少一个维度
        query = query.view(batchsize, self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(batchsize, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(batchsize, self.num_heads, self.d_k).transpose(1, 2)

        atten = attention(query, key, value, mask)
        #print("attention", atten.shape)
        

        #atten = atten.transpose(1, 2).contiguous().view(batchsize, -1, self.d_model) #.contiguous()方法用于确保张量在内存中是连续存储的。当对张量进行某些操作（如transpose、permute、view等）后，张量的数据可能不再在内存中连续存储，这时如果想要进一步对这个张量进行某些操作（特别是需要直接访问底层数据的操作，如view），就需要调用.contiguous()来确保数据是连续的。
        atten = atten.view(batchsize, self.d_model) #.contiguous()方法用于确保张量在内存中是连续存储的。当对张量进行某些操作（如transpose、permute、view等）后，张量的数据可能不再在内存中连续存储，这时如果想要进一步对这个张量进行某些操作（特别是需要直接访问底层数据的操作，如view），就需要调用.contiguous()来确保数据是连续的。
        #(batch_size, max_sequence_length, dim_embed)
        out = self.dropout(self.fc_out(atten))
        #print("attention out: ", atten.shape)
        return out
    
class FFN(nn.Module):
    def __init__(self, d_model, d_ffn, drop_prob) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(drop_prob)
        )

    def forward(self, x):
        return self.ffn(x)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ffn, drop_prob):
        super().__init__()
        self.self_atten = MultiHeadAttention(d_model, num_heads, drop_prob)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ffn, drop_prob)
        self.layer_norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, x_mask):
        #print("IN EncoderBlock.forward(): ", x.shape)
        x = x + self.sub_layer1(x, x_mask)
        x = x + self.sub_layer2(x)
        return x

    def sub_layer1(self, x, x_mask):
        #print("IN EncoderBlock.sub1(): ", x.shape)
        x = self.self_atten(x, x, x_mask) #todo 问题是在这一行x变成了64,1,784
        #print("1111", x.shape)
        x = self.layer_norm1(x)
        #print("6666", x.shape)
        return x

    def sub_layer2(self, x):
        #print("IN EncoderBlock.sub2(): ", x.shape)
        x = self.ffn(x)
        x = self.layer_norm2(x)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ffn, drop_prob, num_blocks):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderBlock(d_model ,num_heads, d_ffn, drop_prob) for _ in range(num_blocks)]
        )

    def forward(self, x, x_mask):
        #print("In encoder shape: ", x.shape)
        for layer in self.layers:
            x = layer(x, x_mask)
        #print("Out encoder shape: ", x.shape)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ffn, drop_prob):
        super().__init__()
        #self attention
        self.self_atten = MultiHeadAttention(d_model, num_heads, drop_prob)
        self.layer_norm1 = nn.LayerNorm(d_model)
        #target source attention
        self.tgt_src_atten = MultiHeadAttention(d_model, num_heads, drop_prob)
        self.layer_norm2 = nn.LayerNorm(d_model)
        #feed forward
        self.ffn = FFN(d_model, d_ffn, drop_prob)
        self.layer_norm3 = nn.LayerNorm(d_model)

    def forward(self, y, y_mask, x, x_mask):
        #print("IN DecoderBlock.forward(): ", y.shape)
        y = y + self.sub_layer1(y, y_mask)
        y = y + self.sub_layer2(y, x, x_mask)
        y = y + self.sub_layer3(y)
        return y

    def sub_layer1(self, y, y_mask):
        y = self.self_atten(y, y, y_mask)
        y = self.layer_norm1(y)
        return y

    def sub_layer2(self, y, x, x_mask):
        y = self.tgt_src_atten(y, x, x_mask)
        y = self.layer_norm2(y)
        return y

        
    def sub_layer3(self, y):
        y = self.ffn(y)
        y = self.layer_norm3(y)
        return y

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ffn, drop_prob, num_blocks):
        super().__init__()

        self.layers = nn.ModuleList(
            [DecoderBlock(d_model, num_heads, d_ffn, drop_prob) for _ in range(num_blocks)]
        )

    def forward(self, x, x_mask, y, y_mask):
        #print("In decoder shape: ", y.shape)
        for layer in self.layers:
            y = layer(y, y_mask, x, x_mask)
        return y

# 定义位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                self.encoding[pos, i] = math.sin(pos / (10000 ** (2 * i / d_model)))
                if i + 1 < d_model:
                    self.encoding[pos, i + 1] = math.cos(pos / (10000 ** (2 * i / d_model)))
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)

class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ffn, drop_prob, num_blocks, d_out):
        super().__init__()

        self.encoder = Encoder(d_model, num_heads, d_ffn, drop_prob, num_blocks)
        self.decoder = Decoder(d_model, num_heads, d_ffn, drop_prob, num_blocks)
        self.fc_out = nn.Linear(d_model, d_out)

    def forward(self, x, y, x_mask, y_mask):
        #print(x.shape, y.shape)
        x = self.encoder(x, x_mask)
        y = self.decoder(x, x_mask, y, y_mask)
        #print(y.shape)
        y = self.fc_out(y)
        
        return y


    

