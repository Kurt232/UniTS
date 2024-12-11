# !/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = x.size(-1) // -np.prod(shape)
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze(model):
    for p in model.parameters():
        p.requires_grad = True


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."

    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.hidden), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(cfg.hidden), requires_grad=True)
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta
        # return x


class Embeddings(nn.Module):
    "The embedding module from word, position and token_type embeddings."

    def __init__(self, cfg, pos_embed=None):
        super().__init__()
        self.lin = nn.Linear(cfg.feature_num, cfg.hidden)

        if pos_embed is None:
            self.pos_embed = nn.Embedding(cfg.seq_len, cfg.hidden)  # position embedding
        else:
            self.pos_embed = pos_embed

        self.norm = LayerNorm(cfg)
        # self.drop = nn.Dropout(cfg.p_drop_hidden)
        self.emb_norm = cfg.emb_norm

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), seq_len)  # (S,) -> (B, S)

        # factorized embedding
        e = self.lin(x)
        if self.emb_norm:
            e = self.norm(e)
        e = e + self.pos_embed(pos)
        # return self.drop(self.norm(e))
        return self.norm(e)


class Projecter(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.lin = nn.Linear(cfg.feature_num, cfg.hidden)
        self.norm = LayerNorm(cfg)

    def forward(self, x):
        # factorized embedding
        e = self.lin(x)
        return self.norm(e)


class EmbeddingsA(nn.Module):
    "The embedding module from word, position and token_type embeddings."

    def __init__(self, cfg, pos_embed=None):
        super().__init__()

        if pos_embed is None:
            self.pos_embed = nn.Embedding(cfg.seq_len, cfg.hidden)  # position embedding
        else:
            self.pos_embed = pos_embed
        self.emb_norm = cfg.emb_norm
        self.norm = LayerNorm(cfg)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), seq_len)  # (S,) -> (B, S)

        e = x + self.pos_embed(pos)
        # return self.drop(self.norm(e))
        return self.norm(e)

class MultiProjection(nn.Module):
    """ Multi-Headed Dot Product Attention """

    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_k = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_v = nn.Linear(cfg.hidden, cfg.hidden)

    def forward(self, x):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        return q, k, v


class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """

    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_k = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_v = nn.Linear(cfg.hidden, cfg.hidden)
        # self.drop = nn.Dropout(cfg.p_drop_attn)
        self.scores = None  # for visualization
        self.n_heads = cfg.n_heads

    def forward(self, x):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        # scores = self.drop(F.softmax(scores, dim=-1))
        scores = F.softmax(scores, dim=-1)
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """

    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden, cfg.hidden_ff)
        self.fc2 = nn.Linear(cfg.hidden_ff, cfg.hidden)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(gelu(self.fc1(x)))


class Transformer(nn.Module):
    """ Transformer with Self-Attentive Blocks"""

    def __init__(self, cfg, embed=None):
        super().__init__()
        if embed is None:
            self.embed = Embeddings(cfg)
        else:
            self.embed = embed

        # To used parameter-sharing strategies
        self.n_layers = cfg.n_layers
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg.hidden, cfg.hidden)
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)

    def forward(self, x):
        h = self.embed(x)

        for _ in range(self.n_layers):
            h = self.attn(h)
            h = self.norm1(h + self.proj(h))
            h = self.norm2(h + self.pwff(h))
        return h


class BaseModule(nn.Module):

    def load_self(self, model_file, map_location=None):
        state_dict = self.state_dict()
        model_dicts = torch.load(model_file, map_location=map_location).items()
        for k, v in model_dicts:
            if k in state_dict:
                state_dict.update({k: v})
        self.load_state_dict(state_dict)


class Decoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.linear = nn.Linear(cfg.hidden, cfg.hidden)
        self.norm = LayerNorm(cfg)
        self.pred = nn.Linear(cfg.hidden, cfg.feature_num)

    def forward(self, input_seqs):
        h_masked = gelu(self.linear(input_seqs))
        h_masked = self.norm(h_masked)
        return self.pred(h_masked)


# LIMU-BERT Model
class BertModel4Pretrain(nn.Module):

    def __init__(self, cfg, output_embed=False, freeze_encoder=False, freeze_decoder=False):
        super().__init__()
        self.encoder = Transformer(cfg)
        if freeze_encoder:
            freeze(self.encoder)
        self.decoder = Decoder(cfg)
        if freeze_decoder:
            freeze(self.decoder)
        self.output_embed = output_embed

    def forward(self, input_seqs, masked_pos=None):
        h_masked = self.encoder(input_seqs)
        if self.output_embed:
            return h_masked
        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, h_masked.size(-1))
            h_masked = torch.gather(h_masked, 1, masked_pos)
        return self.decoder(h_masked)


class CompositeClassifier(BaseModule):

    def __init__(self, encoder_cfg, classifier=None, freeze_encoder=False, freeze_decoder=False, freeze_classifier=False):
        super().__init__()
        self.encoder = Transformer(encoder_cfg)
        if freeze_encoder:
            freeze(self.encoder)
        self.decoder = Decoder(encoder_cfg)
        if freeze_decoder:
            freeze(self.decoder)
        self.classifier = classifier
        if freeze_classifier:
            freeze(self.classifier)

    def forward(self, input_seqs, training=False):
        h = self.encoder(input_seqs)
        h = self.classifier(h, training)
        return h


class CompositeClassifierDA(BaseModule):

    def __init__(self, ae_cfg, classifier=None, output_embed=False, freeze_encoder=False, freeze_decoder=False,
                 freeze_classifier=False):
        super().__init__()
        self.encoder = Transformer(ae_cfg)
        if freeze_encoder:
            freeze(self.encoder)
        self.decoder = Decoder(ae_cfg)
        if freeze_decoder:
            freeze(self.decoder)
        self.classifier = classifier
        if freeze_classifier:
            freeze(self.classifier)
        self.output_embed = output_embed
        self.domain_classifier = self.init_domain_classifier()

    def forward(self, input_seqs, training=False, output_clf=True, masked_pos=None, embed=False, lam=1.0):
        h = self.encoder(input_seqs)
        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))
            h = torch.gather(h, 1, masked_pos)
        h_features = self.classifier(h, embed=True)
        if embed:
            return h_features
        if output_clf:
            class_output = self.classifier(h, training)
            if training:
                h_reverse = ReverseLayerF.apply(h_features, lam)
                domain_output = self.domain_classifier(h_reverse)
                return class_output, domain_output
            else:
                return class_output
        else:
            r = self.decoder(h)
            return r

    def init_domain_classifier(self):
        domain_classifier = nn.Sequential()
        domain_classifier.add_module('gru_d_fc1', nn.Linear(20, 72))
        domain_classifier.add_module('gru_d_relu1', nn.ReLU())
        domain_classifier.add_module('gru_d_fc2', nn.Linear(72, 2))
        return domain_classifier


class CompositeMobileClassifier(BaseModule):

    def __init__(self, encoder_cfg, classifier=None, freeze_encoder=False, freeze_classifier=False):
        super().__init__()
        self.encoder = Transformer(encoder_cfg)
        if freeze_encoder:
            freeze(self.encoder)
        self.classifier = classifier
        if freeze_classifier:
            freeze(self.classifier)

    def forward(self, input_seqs, training=False):
        h = self.encoder(input_seqs)
        h = self.classifier(h, training)
        return h


class ClassifierGRU(BaseModule):
    def __init__(self, cfg, input=None, output=None):
        super().__init__()
        for i in range(cfg.num_rnn):
            if input is not None and i == 0:
                self.__setattr__('gru' + str(i),
                                 nn.GRU(input, cfg.rnn_io[i][1], num_layers=cfg.num_layers[i]
                                        , bidirectional=cfg.rnn_bidirection[i], batch_first=True))
            else:
                self.__setattr__('gru' + str(i),
                                 nn.GRU(cfg.rnn_io[i][0], cfg.rnn_io[i][1], num_layers=cfg.num_layers[i]
                                        , bidirectional=cfg.rnn_bidirection[i], batch_first=True))
        for i in range(cfg.num_linear):
            if output is not None and i == cfg.num_linear - 1:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], output))
            else:
                self.__setattr__('lin' + str(i), nn.Linear(cfg.linear_io[i][0], cfg.linear_io[i][1]))
        self.activ = cfg.activ
        self.dropout = cfg.dropout
        self.num_rnn = cfg.num_rnn
        self.num_linear = cfg.num_linear
        self.bidirectional = any(cfg.rnn_bidirection)

    def forward(self, input_seqs, training=False, embed=False):
        h = input_seqs
        for i in range(self.num_rnn):
            rnn = self.__getattr__('gru' + str(i))
            h, _ = rnn(h)
            if self.activ:
                h = F.relu(h)
        # if self.bidirectional:
        #     h = h.view(h.size(0), h.size(1), 2, -1)
        #     h = torch.cat([h[:, -1, 0], h[:, 0, 1]], dim=1)
        # else:
        # Our experiments shows h = h[:, -1, :] can achieve better performance than the above codes
        # even though the GRU layers are bidirectional.
        h = h[:, -1, :]
        if embed:
            return h
        if self.dropout:
            h = F.dropout(h, training=training)
        for i in range(self.num_linear):
            linear = self.__getattr__('lin' + str(i))
            h = linear(h)
            if self.activ:
                h = F.relu(h)
        return h


def fetch_classifier(method, model_cfg, input=None, output=None):
    if 'limu' in method:
        model = ClassifierGRU(model_cfg, input=input, output=output)
    elif 'unihar' in method:
        model = ClassifierGRU(model_cfg, input=input, output=output)
    else:
        model = None
    return model

from dataclasses import dataclass, field

'''
    "feature_num" : 6,
    "hidden": 72,
    "hidden_ff": 144,
    "n_layers": 4,
    "n_heads": 4,
    "seq_len": 20,
    "emb_norm": true

    "num_rnn": 2,
    "num_layers": [2, 1],
    "rnn_io": [[6,20], [20, 10]],
    "rnn_bidirection": [false, false],
    "num_linear": 1,
    "linear_io": [[10, 3]],
    "activ": false,
    "dropout": true,
    "encoder": "limu_bert"
'''

@dataclass
class ModelArgs:
    # unihar settings
    feature_num: int = 6
    hidden: int = 36
    hidden_ff: int = 72
    n_layers: int = 1
    n_heads: int = 4
    seq_len: int = 100
    emb_norm: bool = True
    num_class: int = 7

    num_rnn: int = 1
    num_layers: list = field(default_factory=lambda: [1])
    rnn_io: list = field(default_factory=lambda: [[6, 10]])
    rnn_bidirection: list = field(default_factory=lambda: [True])
    num_linear: int = 1
    linear_io: list = field(default_factory=lambda: [[20, 7]])
    activ: bool = False
    dropout: bool = True



class Model(nn.Module):
    def __init__(self, cfg, freeze_encoder=False, freeze_classifier=False):
        super().__init__()
        self.encoder = Transformer(cfg)
        if freeze_encoder:
            freeze(self.encoder)
        self.classifier = ClassifierGRU(cfg, input=cfg.hidden, output=cfg.num_class)
        if freeze_classifier:
            freeze(self.classifier)

    def forward(self, input_seqs, training=False):
        h = self.encoder(input_seqs)
        h = self.classifier(h, training)
        return h

if __name__ == '__main__':
    model = Model(ModelArgs())
    x = torch.randn(1, 100, 6)
    y = model(x)
    print(y.shape)
