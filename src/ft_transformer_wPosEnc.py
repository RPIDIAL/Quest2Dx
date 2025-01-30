import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor

from einops import rearrange, repeat

import math
import numpy as np

# from KANLayer import *

# Added Positional Encoding 2023.11.12 following https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# feedforward and attention

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        attn = sim.softmax(dim = -1)
        dropped_attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)

        return out, attn

# transformer

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout),
                FeedForward(dim, dropout = ff_dropout),
                # KANLayer(dim,)
            ]))

    def forward(self, x, return_attn = False):
        post_softmax_attns = []

        for attn, ff in self.layers:
            attn_out, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)

            x = attn_out + x
            x = ff(x) + x

        if not return_attn:
            return x

        return x, torch.stack(post_softmax_attns)

# numerical embedder

class NumericalEmbedder(nn.Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        x = rearrange(x, 'b n -> b n 1')
        return x * self.weights + self.biases

# main class

class FTTransformer(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head = 16,
        dim_out = 1,
        num_special_tokens = 3,
        # num_special_tokens = 2,
        attn_dropout = 0.,
        ff_dropout = 0.,
        path_col_name_emb = None, # Added 2023.01.10,
        missing_method = 'impute',
        columns_to_drop = None,
        subset_col_idx = None,
        # device = None # Added 2023.01.10
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'
        
        # increase the embedding dimension if column name embeddings are used
        if path_col_name_emb is not None:
            dim_post = dim * 2
        self.alt_test_flag = False
        self.small_data_flag = True if subset_col_idx is not None else False
        self.subset_col_idx = subset_col_idx

        # categories related calculations
        self.num_categories = len(categories)
        # self.num_unique_categories = sum(categories)
        self.num_unique_categories = sum(categories)+1

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        if self.num_unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
            categories_offset = categories_offset.cumsum(dim = -1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

            # categorical embedding

            self.categorical_embeds = nn.Embedding(total_tokens, dim)

        # continuous

        self.num_continuous = num_continuous

        if self.num_continuous > 0:
            self.numerical_embedder = NumericalEmbedder(dim, self.num_continuous)

        # cls token

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_post)) #(dim*2)

        # missing token - added on 2023.11.14
        # self.missing_token = nn.Parameter(torch.randn(1, 1))
        self.missing_token = nn.Parameter(torch.randn(1, 1, dim))
        self.missing_method_flag = missing_method
        self.columns_to_drop = columns_to_drop

        # transformer

        self.transformer = Transformer(
            dim = dim_post, # (dim*2)
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout
        )

        # to logits

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim_post), #(dim*2)
            nn.ReLU(),
            nn.Linear(dim_post, dim_out) # (dim*2)
        )

        # Added positional encoding 2023.11.12
        self.pos_encoder = PositionalEncoding(dim, ff_dropout)

        # Load column name embeddings - added 2023.01.10
        if path_col_name_emb is not None:
            # self.col_name_emb = torch.Tensor(np.load(path_col_name_emb)).to(device) # shape = [num_cols, bert_dim]
            col_name_emb = torch.Tensor(np.load(path_col_name_emb)) # shape = [num_cols, bert_dim]
            if self.small_data_flag:
                col_name_emb = col_name_emb[self.subset_col_idx,:] # subset of columns
            if self.alt_test_flag:
                added_cols = torch.zeros(len(categories) - col_name_emb.shape[0]+5,col_name_emb.shape[-1])
                col_name_emb = torch.cat((col_name_emb, added_cols), dim = 0)
            self.register_buffer('col_name_emb', col_name_emb)
            self.emb_proj = nn.Linear(self.col_name_emb.shape[1], dim)
            # drop the columns of self.col_name_emb that are at the positions in columns_to_drop
            if self.columns_to_drop is not None:
                self.col_name_emb = self.col_name_emb[list(set(range(self.col_name_emb.shape[0]) ) ^ set(self.columns_to_drop)),:]

    def forward(self, x_categ, x_numer, return_attn = False):
        # assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'

        xs = []
        # Added to deal with missing tokens 2023.11.14 - moved to top so that the categorical embedder works
        # x_categ = torch.where(torch.isnan(x_categ), repeat(self.missing_token, '1 d -> b d', b =x_categ.shape[0]), x_categ)
        # replace all nan value in x_categ with -1
        # x_categ = torch.where(torch.isnan(x_categ), torch.tensor(9999), x_categ) # 2024.03.19 - nan encoding into 9999 handled in dataset class now
        
        # split x_categ into two parts: one for unique category ids and the other for special missing token (identified by 9999)
        # x_categ_unique = x_categ.clone()
        # x_categ_missing = x_categ.clone()
        # x_categ_unique[x_categ_unique == 9999] = 0

        # If answer embedings have been precomputed, then skip the categorical embedder
        if x_categ.shape[-1] == 1: # TODO: revise if correct for non precomputed embeddings
            if self.num_unique_categories > 0:
                missing_idx = torch.where(x_categ == -1)
                # replace all -1 with 0
                x_categ[x_categ == -1] = 0
                
                x_categ = x_categ + self.categories_offset

                x_categ = self.categorical_embeds(x_categ)

                # replace missing tokens with missing token
                x_categ[missing_idx] = self.missing_token

                xs.append(x_categ)
        else:
            xs.append(x_categ)

        # add numerically embedded tokens
        if self.num_continuous > 0:
            x_numer = self.numerical_embedder(x_numer)

            xs.append(x_numer)

        # concat categorical and numerical

        x = torch.cat(xs, dim = 1)

        b = x.shape[0] # x.shape = [b, num_cols, d]
        # Add BERT embedding if exists
        # self.col_name_emb = None
        if self.col_name_emb is not None:   
            # self.col_name_emb = torch.cat((self.col_name_emb, self.added_cols), dim = 0)        
            x = torch.cat((x,repeat(self.emb_proj(self.col_name_emb), 'f d -> b f d', b = b)),axis=-1)

        # append cls tokens
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)

        # Add positional encoding if BERT embedding does not exist
        if self.col_name_emb is None:
            x = self.pos_encoder(x)

        # # Added to deal with missing tokens 2023.11.14
        # x = torch.where(torch.isnan(x), repeat(self.missing_token, '1 1 d -> b f d', b = b, f = x.shape[1]), x)

        # Added Positional encoding 2023.11.12 - modified 2024.01.10 to add column name embeddings
        # x = self.pos_encoder(x) if self.col_name_emb is None else x + self.emb_proj(self.col_name_emb)
        # x = self.pos_encoder(x) if self.col_name_emb is None else torch.cat((x,repeat(self.emb_proj(self.col_name_emb), 'f d -> b f d', b = b)),axis=-1) #TODO: fix dimensions on transformer

        # attend

        x, attns = self.transformer(x, return_attn = True)

        # get cls token

        x = x[:, 0]

        # out in the paper is linear(relu(ln(cls)))

        logits = self.to_logits(x)

        if not return_attn:
            return logits, logits

        return logits, attns