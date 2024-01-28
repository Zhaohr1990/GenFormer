import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.masking import TriangularCausalMask, ProbMask
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, ProbAttention, AttentionLayer
from layers.Embed import StatePosEmbedding
import numpy as np

class Model(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = 1
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = StatePosEmbedding(configs.num_grps, configs.d_model, configs.dropout)
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers_markov)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

    def forward(self, x_state_enc, x_enc, x_mark_enc, enc_self_mask=None): 
        """
            Perform forward pass of the deep learning model
        """

        enc_out = self.enc_embedding(x_state_enc, x_enc, x_mark_enc) 
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        if self.output_attention:
            return torch.matmul(enc_out[:, -self.pred_len:, :], self.enc_embedding.state_embedding.weight.transpose(1, 0))[:, :, :-1], attns
        else:
            return torch.matmul(enc_out[:, -self.pred_len:, :], self.enc_embedding.state_embedding.weight.transpose(1, 0))[:, :, :-1] # [B, L, D]
