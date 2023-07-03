import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer



# -------------CSICB---------------------
class _Scale_Construct(nn.Module):
    def __init__(self, c_in, window_size, dropout_rate):
        """
        :param c_in: channel
        :param window_size
        :param dropout_rate
        """
        super(_Scale_Construct, self).__init__()
        # conv
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=window_size,
                                  stride=window_size)
        # pooling
        self.avgpooling = nn.AvgPool1d(kernel_size=window_size)
        self.maxpooling = nn.MaxPool1d(kernel_size=window_size)

        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.BatchNorm1d(c_in)
        self.fusion = nn.Linear(4,1,bias=False)

    def forward(self, x, CSICB):
        """
        :param x: [Batch, Channel, Length]
        :param CSICB: ["ALL", "Conv", "Pooling"]
        :return: [Batch, Channel, Length_2]
        """
        if CSICB == "All":
            x = self.fusion(torch.stack([self.downConv(x),self.avgpooling(x),self.maxpooling(x),self.maxpooling(-x)],-1)).squeeze()
        elif CSICB == "Conv":
            x = self.downConv(x)
        elif CSICB == "Pooling":
            x = self.avgpooling(x)+self.maxpooling(x)+self.maxpooling(-x)
        # x = self.activation(self.norm(x))
        # x = self.dropout(x)
        # x = self.norm(x)
        return x


class Scale_Construct(nn.Module):
    def __init__(self, c_in, window_size, dropout_rate=0.1):
        """
        :param c_in: channel
        :param window_size
        :param dropout_rate
        """
        super(Scale_Construct, self).__init__()

        self.construct_layers = []
        for i in range(len(window_size)):
            self.construct_layers.append(_Scale_Construct(c_in, window_size[i], dropout_rate))
        self.construct_layers = nn.ModuleList(self.construct_layers)

    def forward(self, x, CSICB):
        """
        :param x: [Batch, Length, Channel]
        :param CSICB
        :return: list:[scale, Batch, Length, Channel]
        """
        temp_input = x.permute(0, 2, 1) # [B, C, L]
        all_inputs = [x]
        for i in range(len(self.construct_layers)):
            temp_input = self.construct_layers[i](temp_input, CSICB)
            all_inputs.append(temp_input.transpose(1, 2))

        return all_inputs


# -------------intra-scale interaction-------------
class InnerScale(nn.Module):
    def __init__(self, length, c_in, batch_size, hidden_innerscale_size, dropout_rate):
        """
        :param length
        :param c_in: channel
        :param batch_size
        :param hidden_innerscale_size:rnn hidden state
        :param dropout_rate
        """
        super(InnerScale, self).__init__()

        self.batch_size = batch_size
        self.inner_interact = nn.LSTM(c_in, c_in, batch_first=True)
        self.hidden_layer_size = hidden_innerscale_size if hidden_innerscale_size!=0 else c_in
        self.hidden_cell = (
            torch.nn.init.xavier_uniform_(torch.zeros(1, self.batch_size, c_in)).cuda(),
            torch.nn.init.xavier_uniform_(torch.zeros(1, self.batch_size, c_in)).cuda()
        )

        self.out_net1 = nn.Sequential(*[
            nn.Linear(c_in, self.hidden_layer_size),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_layer_size, c_in),
            # nn.Sigmoid(),
        ])
        self.out_net2 = nn.Sequential(*[
            # nn.Linear(c_in, c_in),
            # nn.Dropout(dropout_rate),
            # nn.Linear(self.hidden_layer_size, c_in),
            nn.Sigmoid(),
        ])


    def forward(self, x):
        """
        :param x: [B, L, C]
        :return: output: [B, L, C]
        """
        output, (h0, c0) = self.inner_interact(x, self.hidden_cell)

        output = self.out_net1(output) * self.out_net2(x)

        return output

# -------------inter-scale interaction-------------
class InterScale(nn.Module):
    def __init__(self, length_up, length_down, hidden_size, dropout_rate, cin):
        """
        :param length_up
        :param length_down
        :param hidden_size: global information length
        :param dropout_rate
        :param cin: channel
        """
        super(InterScale, self).__init__()

        self.bottleneck_interact1 = nn.Linear(length_up, hidden_size)
        self.bottleneck_interact2 = nn.Linear(cin, cin)
        self.bottleneck_interact3 = nn.Linear(hidden_size, length_down)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x1, x2):
        """
        :param x1: [B, L_up, C]
        :param x2: [B, L_down, C]
        :return: new_x2: [B, L, C]
        """
        mid = self.bottleneck_interact1(x1.transpose(1, 2)).transpose(1, 2)
        mid = self.bottleneck_interact2(mid)
        upper = self.bottleneck_interact3(mid.transpose(1, 2)).transpose(1, 2)
        upper = self.dropout(upper)

        new_x2 = x2 + upper

        return new_x2

class ScaleLayer(nn.Module):
    def __init__(self, c_in, batch_size, length_up, length_down, hidden_innerscale_size, hidden_interscale_size, dropout_rate):
        """
        :param c_in: channel
        :param batch_size
        :param length_up
        :param length_down
        :param hidden_innerscale_size: rnn hidden state
        :param hidden_interscale_size: global information length
        :param dropout_rate
        """
        super(ScaleLayer, self).__init__()

        self.innerscale = InnerScale(length_up, c_in, batch_size, hidden_innerscale_size, dropout_rate)
        self.interscale = InterScale(length_up, length_down, hidden_interscale_size, dropout_rate, c_in)

    def forward(self, x1, x2, scale_interact):
        """
        interscale and intrascale
        :param x1: [B, L_up, C]
        :param x2: [B, L_down, C]
        :return: new_x1, new_x2
        """
        if scale_interact == "Inner":
            new_x1 = self.innerscale(x1)
            new_x2 = x2
        elif scale_interact == 'Inter':
            new_x1 = x1
            new_x2 = self.interscale(new_x1, x2)
        elif scale_interact == "None":
            new_x1 = x1
            new_x2 = x2
        elif scale_interact == "All":
            new_x1 = self.innerscale(x1)
            new_x2 = self.interscale(new_x1, x2)


        return new_x1, new_x2

# -------TPRNN--------
class Model(nn.Module):

    def __init__(self, opt):
        super().__init__()

        self.output_len = opt.output_len
        self.input_len = opt.input_len
        self.channels = opt.enc_in
        self.hidden_interscale_size = opt.hidden_interscale_size
        self.hidden_innerscale_size = opt.hidden_innerscale_size
        self.dropout_rate = opt.dropout_rate
        self.batch_size = opt.batch_size
        self.window_size = eval(opt.window_size)    # type(window_size) == list
        self.scale_nums = len(self.window_size)     # C
        self.multiscale_or_not = opt.multi_scale
        self.scale_interact_or_not = opt.scale_interact
        self.CSICB = opt.CSICB

        # structure
        self.scale_construct = Scale_Construct(self.channels, self.window_size, self.dropout_rate)
        self.scale_interact = []
        self.scale_length = [self.input_len] # [l,l/2,l/4...]
        for i in self.window_size:
            self.scale_length.append(self.scale_length[-1] // i)

        for i in range(1,len(self.window_size)+1):
            self.scale_interact.append(ScaleLayer(self.channels, self.batch_size, self.scale_length[-i], self.scale_length[-i-1],
                                       self.hidden_innerscale_size, self.hidden_interscale_size, self.dropout_rate))
        self.scale_interact = nn.ModuleList(self.scale_interact)


        self.last_scale = InnerScale(self.input_len, self.channels, self.batch_size, self.hidden_innerscale_size, self.dropout_rate)
        self.predict = nn.ModuleList([
            nn.Linear(i, self.output_len) for i in self.scale_length
        ])
        self.pred_fusion = nn.Linear(self.scale_nums+1,1,bias=False)


    def forward(self, x, *_):
        """
        Input: x [B, L, C]
        Output:
        """
        means = x.mean(1, keepdim=True).detach()
        x = x - means

        scaled_x = self.scale_construct(x, self.CSICB)[::-1]

        output = []
        x2 = scaled_x[0]
        for i in range(self.scale_nums):
            x1, x2 = self.scale_interact[i](x2, scaled_x[i+1], self.scale_interact_or_not)
            output.insert(0, x1)
        if self.scale_interact_or_not == 'None':
            output.insert(0, x2) # [96,48,...,6]
        else:
            output.insert(0, self.last_scale(x2)) # [96,48,...,6]

        if self.multiscale_or_not:
            y = []
            for i in range(self.scale_nums+1):
                y.append(self.predict[i](output[i].transpose(1,2)).transpose(1,2))
            y = torch.stack(y, -1)
            y = self.pred_fusion(y).squeeze()

        else:
            y = self.predict[0](output[0].transpose(1, 2)).transpose(1, 2)

        y = y + means

        return y

