import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.masking import TriangularCausalMask, ProbMask
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, ProbAttention, AttentionLayer
from layers.Embed import StateTimeEmbedding_wo_pos, StateTimeEmbedding_wo_time
import numpy as np

class Model(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        if configs.embedding_mode == 'StateTime_wo_pos':
            self.enc_embedding = StateTimeEmbedding_wo_pos(configs.enc_in, configs.num_grps, configs.d_model, configs.freq, configs.dropout)
            self.dec_embedding = StateTimeEmbedding_wo_pos(configs.dec_in, configs.num_grps, configs.d_model, configs.freq, configs.dropout)
        elif configs.embedding_mode == 'StateTime_wo_time':
            self.enc_embedding = StateTimeEmbedding_wo_time(configs.enc_in, configs.num_grps, configs.d_model, configs.dropout)
            self.dec_embedding = StateTimeEmbedding_wo_time(configs.dec_in, configs.num_grps, configs.d_model, configs.dropout)
        else:
            raise ValueError('Example not found.')
        
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
                ) for l in range(configs.e_layers)
            ],
            [
                ConvLayer(
                    configs.d_model
                ) for l in range(configs.e_layers - 1)
            ] if configs.distil else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_state_enc, x_enc, x_mark_enc, x_state_dec, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None): 

        enc_out = self.enc_embedding(x_state_enc, x_enc, x_mark_enc) 
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_state_dec, x_dec, x_mark_dec) 
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
