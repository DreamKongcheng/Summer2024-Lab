import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

class PatchEmbedded(nn.Module):
    def __init__(self, img_size=28, channels=1, patch_size=7, embed_dim=128):
        """
        img_size: 28
        channels: 1
        patch_size: 7

        """
        super().__init__()
        #self.nunum_batches = num_batches
        self.img_size = img_size
        self.channels = channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        patch_dim = patch_size ** 2 * channels
        num_patches = (img_size // patch_size) ** 2

        self.embedding = nn.Linear(patch_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.position_embedding = nn.Parameter(torch.randn(1, 1 + num_patches, embed_dim))

        #b, n=h*w/p^2, p^2*c


    def forward(self, x):
        #print(x.shape) #64 1 28 28
        x = rearrange(x, "b c (h h1) (w w1) -> b (h w) (c h1 w1)", h1 = self.patch_size, w1 = self.patch_size) #分割成n个patch
        #print("x after rearrange: ", x.shape) #b n c*p*p
        x = self.embedding(x) #b, n, d_model
        #添加cls token
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_token, x), dim=1) #b, n+1, d_model
        x = x + self.position_embedding
        #print(x.shape)  64, 17, 128
        return x


def attention(query, key, value):
    d_k = key.shape[-1]
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / (d_k ** 0.5)

    return torch.matmul(F.softmax(scores, dim=-1), value)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads

        self.w_query = nn.Linear(d_model, d_model)
        self.w_key = nn.Linear(d_model, d_model)
        self.w_value = nn.Linear(d_model, d_model)

        self.fc_out = nn.Linear(d_model, d_model)
        


    def forward(self, x):
        # x = x.view(x.size(0), -1, self.num_heads, self.d_k).transpose(1, 2) 
        #b, len, d_model ->b, len, h, d//h->b,h,len,d//h
        #print(x.shape) 64 8 17 16
        query = self.w_query(x)
        key = self.w_key(x)
        value = self.w_value(x)

        query = query.view(query.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(key.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(key.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)


        atten = attention(query, key, value) #b,h,len,d//h
        atten = atten.transpose(1, 2).contiguous().view(x.size(0), -1, self.num_heads * self.d_k)
        
        return self.fc_out(atten) #b,h,d

class MLPBlock(nn.Module):
    def __init__(self, d_model, d_liner, drop_prob):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_liner),
            nn.GELU(),
            nn.Dropout(drop_prob),
            nn.Linear(d_liner, d_model),
            nn.Dropout(drop_prob)
        )
    def forward(self, x):
        return self.net(x)




class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_liner, num_heads, drop_prob):
        super().__init__()
        self.LayerNorm1 = nn.LayerNorm(d_model)
        self.atten = MultiHeadAttention(d_model, num_heads)
        self.drop_out2 = nn.Dropout(drop_prob)
        self.LayerNorm2 = nn.LayerNorm(d_model)
        self.mlp = MLPBlock(d_model, d_liner, drop_prob)
        self.drop_out1 = nn.Dropout(drop_prob)

    def forward(self, x):
        x = x + self.sub_layer1(x) #MSA部分与残差链接
        x = x + self.sub_layer2(x) #MLP部分与残差链接
        return x


    def sub_layer1(self, x):
        x = self.LayerNorm1(x)
        x = self.atten(x)
        x = self.drop_out1(x)
        return x
    def sub_layer2(self, x):
        x = self.LayerNorm2(x)
        x = self.mlp(x)
        x = self.drop_out2(x)
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, num_blocks, d_model, d_liner, num_heads, drop_prob):
        super().__init__()
        self.num_blocks = num_blocks
        self.layers = nn.ModuleList(
            [EncoderBlock(d_model, d_liner, num_heads, drop_prob) for _ in range(num_blocks)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        #print(x.shape)
        return x
    
class VIT(nn.Module):
    def __init__(self, num_blocks, d_model, d_liner, num_heads, drop_prob, img_size, channels, patch_size, num_classes):
        super().__init__()
        self.patch_embedding = PatchEmbedded(img_size, channels, patch_size, d_model)
        self.dropout = nn.Dropout(drop_prob)
        self.encoder = TransformerEncoder(num_blocks, d_model, d_liner, num_heads, drop_prob)
        self.ln = nn.LayerNorm(d_model)
        self.mlp_head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.dropout(x)
        x = self.encoder(x)
        x = self.ln(x)
        x = x[:, 0, :]
        x = self.mlp_head(x)
        return x
    
