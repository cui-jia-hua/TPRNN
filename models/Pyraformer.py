import torch
import torch.nn as nn
from layers.Pyraformer_Layers import EncoderLayer, Decoder, Predictor
from layers.Pyraformer_Layers import Bottleneck_Construct, Conv_Construct, MaxPooling_Construct, AvgPooling_Construct
from layers.Pyraformer_Layers import get_mask, get_subsequent_mask, refer_points
from data_embed.Pyraformer_Embed import DataEmbedding, CustomEmbedding


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self, opt):
        super().__init__()

        self.d_model = opt.d_model
        self.model_type = opt.model
        self.window_size = opt.window_size
        self.truncate = opt.truncate
        if opt.decoder_type == 'attention':
            self.mask, self.all_size = get_mask(opt.input_len, opt.window_size, opt.inner_size)
        else:
            self.mask, self.all_size = get_mask(opt.input_len+1, opt.window_size, opt.inner_size)
        self.decoder_type = opt.decoder_type
        if opt.decoder_type == 'FC':
            self.indexes = refer_points(self.all_size, opt.window_size)

        self.layers = nn.ModuleList([
            EncoderLayer(opt.d_model, opt.d_inner_hid, opt.n_heads, opt.d_k, opt.d_v, dropout=opt.dropout, \
                normalize_before=False) for i in range(opt.e_layers)
            ])

        self.enc_embedding = DataEmbedding(opt.enc_in, opt.d_model, opt.dropout, opt.freq)
        if opt.data_loader_type == "custom_s":
            self.enc_embedding = CustomEmbedding(opt.enc_in, opt.d_model, opt.covariate_size, opt.seq_num, opt.dropout)
        self.conv_layers = eval(opt.CSCM)(opt.d_model, opt.window_size, opt.d_bottleneck)

    def forward(self, x_enc, x_mark_enc):

        seq_enc = self.enc_embedding(x_enc, x_mark_enc)

        # 通过一个size*size的mask来获取不同节点attention计算的范围（不使用TVM时）
        mask = self.mask.repeat(len(seq_enc), 1, 1).to(x_enc.device)
        seq_enc = self.conv_layers(seq_enc)

        for i in range(len(self.layers)):
            seq_enc, _ = self.layers[i](seq_enc, mask)

        if self.decoder_type == 'FC':
            indexes = self.indexes.repeat(seq_enc.size(0), 1, 1, seq_enc.size(2)).to(seq_enc.device)
            indexes = indexes.view(seq_enc.size(0), -1, seq_enc.size(2))
            all_enc = torch.gather(seq_enc, 1, indexes)
            seq_enc = all_enc.view(seq_enc.size(0), self.all_size[0], -1)
        elif self.decoder_type == 'attention' and self.truncate:
            seq_enc = seq_enc[:, :self.all_size[0]]

        return seq_enc


class Model(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self, opt):
        super().__init__()

        self.predict_step = opt.output_len
        self.d_model = opt.d_model
        self.input_size = opt.input_len
        self.decoder_type = opt.decoder_type
        self.channels = opt.enc_in

        opt.window_size = eval(opt.window_size)

        self.encoder = Encoder(opt)
        if opt.decoder_type == 'attention':
            mask = get_subsequent_mask(opt.input_len, opt.window_size, opt.output_len, opt.truncate)
            self.decoder = Decoder(opt, mask)
            self.predictor = Predictor(opt.d_model, opt.enc_in)
        elif opt.decoder_type == 'FC':
            self.predictor = Predictor(4 * opt.d_model, opt.output_len * opt.enc_in)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, pretrain=False):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """
        if self.decoder_type == 'attention':
            enc_output = self.encoder(x_enc, x_mark_enc)
            dec_enc = self.decoder(x_dec, x_mark_dec, enc_output)

            if pretrain:
                dec_enc = torch.cat([enc_output[:, :self.input_size], dec_enc], dim=1)
                pred = self.predictor(dec_enc)
            else:
                pred = self.predictor(dec_enc)
        elif self.decoder_type == 'FC':
            enc_output = self.encoder(x_enc, x_mark_enc)[:, -1, :]
            pred = self.predictor(enc_output).view(enc_output.size(0), self.predict_step, -1)

        return pred

