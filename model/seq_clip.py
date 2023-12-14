import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
import argparse
import math
T = torch.Tensor
TN = Optional[T]
D = torch.device
CPU = torch.device('cpu')


# Turn token into embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Add a positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen=5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# Add sublayer Connection
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

# Positionwised feed forward layer
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.fc1(x).relu()))

# A method to clone N
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# A manual implementation of encoder layer
class ManualEncoderLayer(nn.Module):
    def __init__(self, emb_dim, dropout, nhead, ff_dim):
        super(ManualEncoderLayer, self).__init__()
        self.attention = torch.nn.MultiheadAttention(emb_dim, nhead)
        self.ffn = PositionwiseFeedForward(emb_dim, ff_dim)
        self.emb_dim = emb_dim
        self.sub_layer1 = SublayerConnection(emb_dim, dropout)
        self.sub_layer2 = SublayerConnection(emb_dim, dropout)
        self.dim_ff = ff_dim

    def forward(self, x, src_mask, padding_mask):
        x = self.sub_layer1(x, lambda x: self.attention(x, x, x, attn_mask=src_mask, key_padding_mask=padding_mask)[0])
        x = self.sub_layer2(x, self.ffn)
        return x

# A manual implementation of encoder
class ManualEncoder(nn.Module):
    def __init__(self, layer, N):
        super(ManualEncoder, self).__init__()
        self.layers = clones(layer,N)
        self.norm = nn.LayerNorm(layer.emb_dim)

    def forward(self, x, src_mask, padding_mask):
        for mod in self.layers:
          x = mod(x, src_mask, padding_mask)
        x = self.norm(x)
        return x

# A manual implementation of transformer
class ManualTransformer(nn.Module):
    def __init__(self, num_encoder_layers, emb_size, nhead=8, dim_feedforward=512, dropout=0.1):
        super(ManualTransformer, self).__init__()
        self.encoder = ManualEncoder(ManualEncoderLayer(emb_size, dropout, nhead, dim_feedforward), num_encoder_layers)
        self.positional_encoding = PositionalEncoding(emb_size, dropout)
        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_src_mask(self, src):
        return torch.zeros((src.shape[0], src.shape[0]), device='cuda:0').type(torch.bool)

    def encode(self, src, src_padding_mask):
        src = self.positional_encoding(src)
        src_mask = self.get_src_mask(src)
        return self.encoder(src, src_mask, src_padding_mask)

    def forward(self, src, src_padding_mask):
        memory = self.encode(src, src_padding_mask)
        return memory


# A transformer mapper
class TransformerMapper(nn.Module):
    def __init__(self, clip_dim, clip_length, embed_dim,num_encoder_layers):
        super(TransformerMapper, self).__init__()
        # clip_dim -> clip_length * embed_dim
        self.fc = nn.Linear(clip_dim , embed_dim * clip_length)

        self.transformer = ManualTransformer(num_encoder_layers, clip_dim)
        self.indices = torch.tensor([0]).to('cuda:0')
        self.cls_encoding = nn.Parameter(torch.randn(1,1,clip_dim), requires_grad=True)

    def forward(self,x,padding_mask):
        x = torch.cat((self.cls_encoding.repeat(1,x.shape[1],1),x),dim=0)
        encoding = self.transformer(x,padding_mask)
        encoding = torch.transpose(encoding,dim0=0,dim1=1)
        cls = torch.index_select(encoding, 1, self.indices)
        output = self.fc(cls)
        return output


class ClipCaptionModel(nn.Module):
    def __init__(self, prefix_length: int, num_layers: int,
                 prefix_size: int = 512, ):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        # get embedding for the captions
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]

        self.clip_project = TransformerMapper(clip_dim=prefix_size, embed_dim=self.gpt_embedding_size,
                                              clip_length=prefix_length, num_encoder_layers=num_layers)

    def get_dummy_token(self, batch_size: int, device: D) -> T:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: T, prefix: T, pad_mask: T, mask: Optional[T] = None, labels: Optional[T] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix = torch.transpose(prefix,0,1)
        prefix_projections = self.clip_project(prefix,pad_mask).view(-1, self.prefix_length, self.gpt_embedding_size)
        #print(embedding_text.size()) #torch.Size([5, 67, 768])
        #print(prefix_projections.size()) #torch.Size([5, 1, 768])
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

class ClipCaptionPrefix(ClipCaptionModel):
    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self