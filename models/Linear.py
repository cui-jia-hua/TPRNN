from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_truncated_Layers import Bottleneck_Construct, Conv_Construct, MaxPooling_Construct, AvgPooling_Construct
from data_embed.Pyraformer_Embed import DataEmbedding, CustomEmbedding

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.input_len = configs.input_len
        self.output_len = configs.output_len

        configs.window_size = eval(configs.window_size)
        # self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.dropout, configs.freq)
        # self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.dropout, configs.freq)

        # self.conv_layers = eval(configs.CSCM)(configs.d_model, configs.window_size, configs.d_bottleneck)

        self.channel_project = nn.Linear(configs.enc_in, configs.enc_in, bias=False)
        self.predict = nn.Linear(self.input_len,self.output_len)
        self.channel_project_2 = nn.Linear(configs.enc_in, configs.enc_in, bias=False)

        # decoder
        self.dec = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec):
        # data embedding
        # enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # data conv
        # enc_out = self.conv_layers(enc_out)

        # x: [Batch, Input length, Channal]
        # x = self.channel_project(x)
        x = self.predict(x.permute(0,2,1)).permute(0,2,1)
        # x = self.channel_project(x)
        # x = self.dec(x.permute(0,2,1))
        return x # to [Batch, Output length, Channal]
