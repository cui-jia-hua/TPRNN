import torch
import torch.nn as nn
from utils.tools import Norm
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer



# -------------构建多尺度---------------------
class _Scale_Construct(nn.Module):
    def __init__(self, c_in, window_size, dropout_rate):
        super(_Scale_Construct, self).__init__()

        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=window_size,
                                  stride=window_size)
        self.avgpooling = nn.AvgPool1d(kernel_size=window_size)
        self.maxpooling = nn.MaxPool1d(kernel_size=window_size)

        self.fusion = nn.Linear(4,1,bias=False)

        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.BatchNorm1d(c_in)

    def forward(self, x):
        """
        :param x: [Batch, Channel, Length]
        :return:
        """
        x = self.fusion(
            torch.stack([self.downConv(x), self.avgpooling(x), self.maxpooling(x), self.maxpooling(-x)], -1)).squeeze()

        return x


class Scale_Construct(nn.Module):
    def __init__(self, c_in, window_size, dropout_rate=0.1):
        super(Scale_Construct, self).__init__()

        self.construct_layers = []
        for i in range(len(window_size)):
            self.construct_layers.append(_Scale_Construct(c_in, window_size[i], dropout_rate))
        self.construct_layers = nn.ModuleList(self.construct_layers)

    def forward(self, x):

        temp_input = x.permute(0, 2, 1) # [B, C, L]
        all_inputs = [x]
        for i in range(len(self.construct_layers)):
            temp_input = self.construct_layers[i](temp_input)
            all_inputs.append(temp_input.transpose(1, 2))

        return all_inputs


# -------------尺度内信息融合-------------
class InnerScale(nn.Module):
    def __init__(self, length, c_in, batch_size, hidden_innerscale_size, dropout_rate):
        """
        :param length:序列长度
        :param c_in: channel数量
        :param batch_size:
        :param hidden_innerscale_size:rnn隐藏层维度
        :param dropout_rate:
        """
        super(InnerScale, self).__init__()

        self.batch_size = batch_size
        self.inner_interact = nn.LSTM(c_in, c_in, batch_first=True)
        self.hidden_layer_size = hidden_innerscale_size if hidden_innerscale_size != 0 else c_in
        self.hidden_cell = (
            torch.nn.init.xavier_uniform_(torch.zeros(1, self.batch_size, c_in)).cuda(),
            torch.nn.init.xavier_uniform_(torch.zeros(1, self.batch_size, c_in)).cuda()
        )

        # 门控
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
            # nn.Dropout(dropout_rate)
        ])


    def forward(self, x):
        """
        尺度内聚合到最后一个节点
        :param x: [B, L, C]
        :return: output: [B, L, C]
        """
        output, (h0, c0) = self.inner_interact(x, self.hidden_cell)

        # 门控
        # output = self.out_net1(output)
        # output = self.out_net2(x)
        # output = x
        # output = self.out_net1(x)
        output = self.out_net1(output) * self.out_net2(x)
        # output = self.out_net1(output) * x

        return output


# class InnerScale(nn.Module):
#     def __init__(self, length_up, c_in, batch_size, hidden_innerscale_size, dropout_rate):
#         super(InnerScale, self).__init__()
#
#         hidden_innerscale_size = c_in if hidden_innerscale_size==0 else hidden_innerscale_size
#         # self.embed = nn.Linear(c_in, hidden_innerscale_size)
#         self.encoder = Encoder(
#             [
#                 EncoderLayer(
#                     AttentionLayer(
#                         FullAttention(False, 5, attention_dropout=dropout_rate), hidden_innerscale_size, 1),
#                     hidden_innerscale_size
#                 )
#             ]
#         )
#         # self.rev_embed = nn.Linear(hidden_innerscale_size, c_in)
#
#         # self.attn = nn.Sequential(*[
#         #             nn.Linear(c_in, hidden_innerscale_size),
#         #             Encoder([
#         #                 EncoderLayer(
#         #                     AttentionLayer(
#         #                         FullAttention(False, 5, attention_dropout=dropout_rate),
#         #                         d_model=hidden_innerscale_size, n_heads=4
#         #                     ),
#         #                     d_model=hidden_innerscale_size
#         #                 )
#         #             ])
#         #         ])
#
#
#     def forward(self, x):
#         """
#         尺度内聚合到最后一个节点
#         :param x: [B, L, C]
#         :return: output: [B, L, C]
#         """
#         x,_ = self.encoder(x)
#
#         return x


# -------------尺度间信息传递-------------


class InterScale(nn.Module):
    def __init__(self, length_up, length_down, hidden_size, dropout_rate, cin):
        """
        :param length_up:大尺度序列长度
        :param length_down: 小尺度序列长度
        :param hidden_size: 交互时的全局信息长度
        :param dropout_rate:
        :param cin: channel个数
        """
        super(InterScale, self).__init__()

        # 缩小-交互-膨胀
        self.bottleneck_interact1 = nn.Linear(length_up, hidden_size)
        self.bottleneck_interact2 = nn.Linear(cin, cin)
        self.bottleneck_interact3 = nn.Linear(hidden_size, length_down)

        # last node
        # self.last_point = nn.Linear(1,length_down)

        # 全连接
        # self.full_connect = nn.Linear(length_up, length_down)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x1, x2):
        """
        将上层尺度最后节点x1拼接到下层尺度序列x2上
        :param x1: 上层尺度整条序列：[B, L_up, C]
        :param x2: 下层尺度整条序列：[B, L_down, C]
        :return: new_x2 [B, L, C]
        """

        # 全连接
        # upper = self.full_connect(x1.transpose(1, 2)).transpose(1, 2)

        # last node
        # upper = self.dropout(self.last_point(x1[:,-1:,:].transpose(1,2)).transpose(1,2))

        # TPRNN
        mid = self.bottleneck_interact1(x1.transpose(1, 2)).transpose(1, 2)
        mid = self.bottleneck_interact2(mid)
        upper = self.bottleneck_interact3(mid.transpose(1, 2)).transpose(1, 2)
        upper = self.dropout(upper)

        new_x2 = x2 + upper

        return new_x2


class ScaleLayer(nn.Module):
    def __init__(self, c_in, batch_size, length_up, length_down, hidden_innerscale_size, hidden_interscale_size, dropout_rate):
        """
        :param c_in: channel个数
        :param batch_size:
        :param length_up: 大尺度序列长度
        :param length_down: 小尺度序列长度
        :param hidden_innerscale_size: rnn隐藏层维度
        :param hidden_interscale_size: 尺度间传递的全局信息长度
        :param dropout_rate:
        """
        super(ScaleLayer, self).__init__()

        self.innerscale = InnerScale(length_up, c_in, batch_size, hidden_innerscale_size, dropout_rate)
        self.interscale = InterScale(length_up, length_down, hidden_interscale_size, dropout_rate, c_in)

    def forward(self, x1, x2, scale_interact):
        """
        尺度内+尺度间
        :param x1上层尺度: [B, L_up, C]
        :param x2下层尺度: [B, L_down, C]
        :return: new_x1, new_x2
        """
        # 是否使用尺度间交互
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


# -------------分解-------------
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class series_decomp_simple(nn.Module):
    def __init__(self):
        super(series_decomp_simple, self).__init__()

    def forward(self, x):
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        return x, means


# -------模型本体--------
class Model(nn.Module):

    def __init__(self, opt):
        super().__init__()
        # 参数
        self.output_len = opt.output_len
        self.input_len = opt.input_len
        self.channels = opt.enc_in
        self.hidden_interscale_size = opt.hidden_interscale_size
        self.hidden_innerscale_size = opt.hidden_innerscale_size
        self.dropout_rate = opt.dropout_rate
        self.batch_size = opt.batch_size
        self.window_size = eval(opt.window_size)    # type(window_size) == list
        self.scale_nums = len(self.window_size)     # 除去初始尺度以外的尺度数量
        # 模型选择
        self.multiscale_or_not = opt.multi_scale   # 是否用所有尺度进行预测
        self.scale_interact_or_not = opt.scale_interact
        self.CSICB = opt.CSICB
        # decomp
        # self.decomp = series_decomp(opt.moving_avg)
        self.decomp = series_decomp_simple()
        # 网络构成
        self.scale_construct = Scale_Construct(self.channels, self.window_size, self.dropout_rate)
        self.scale_interact = []
        self.scale_length = [self.input_len] # 序列长度由大到小 [l,l/2,l/4...]
        for i in self.window_size:
            self.scale_length.append(self.scale_length[-1] // i)

        for i in range(1,len(self.window_size)+1):
            self.scale_interact.append(ScaleLayer(self.channels, self.batch_size, self.scale_length[-i], self.scale_length[-i-1],
                                       self.hidden_innerscale_size, self.hidden_interscale_size, self.dropout_rate))
        self.scale_interact = nn.ModuleList(self.scale_interact)
        # 最小尺度序列内单独处理
        self.last_scale = InnerScale(self.input_len, self.channels, self.batch_size, self.hidden_innerscale_size, self.dropout_rate)
        self.predict = nn.ModuleList([
            nn.Linear(i, self.output_len) for i in self.scale_length
        ])
        # 各尺度单独预测再聚合
        self.pred_fusion = nn.Linear(self.scale_nums+1,1,bias=False)

        # attention聚合
        # self.global_token = nn.Parameter(torch.rand(self.channels, 1), requires_grad=True)
        # self.predict_fusion = nn.Sequential(
        #     nn.Linear(self.channels, self.channels),
        #     nn.Dropout(self.dropout_rate)
        # )

        # 用一个大的线性层来融合预测
        # self.predict_all = nn.Linear(sum(self.scale_length), self.output_len)

        # self.trend_pred = nn.ModuleList([nn.Linear(self.input_len, self.output_len, bias=False)]*self.channels)

    def forward(self, x, *_):
        """
        Input: x [B, L, C]
        Output:
        """
        # x = self.norm.norm(x)
        x, trend = self.decomp(x)

        scaled_x = self.scale_construct(x)[::-1] # scaled_x: n*[B, L, C] 从小到大 6,12,24,48,96

        output = []
        x2 = scaled_x[0]
        for i in range(self.scale_nums):
            x1, x2 = self.scale_interact[i](x2, scaled_x[i+1], self.scale_interact_or_not)
            output.insert(0, x1)
            # output.insert(0, x2)
        if self.scale_interact_or_not == 'None':
            output.insert(0, x2) # [96,48,...,6]
        else:
            output.insert(0, self.last_scale(x2)) # [96,48,...,6]

        if self.multiscale_or_not:     # 不同尺度同时预测
            y = []
            # 用一个4变1的层来融合预测
            # fusion_weight = [i/sum(self.scale_length) for i in self.scale_length]
            fusion_weight = [1/len(self.scale_length)] * len(self.scale_length)
            for i in range(self.scale_nums+1):
                y.append(self.predict[i](output[i].transpose(1,2)).transpose(1,2))
            # y = sum([i*j for i,j in zip(y,fusion_weight)])

            # 各尺度单独预测再聚合
            y = torch.stack(y, -1)
            y = self.pred_fusion(y).squeeze()

            # 用一个大的线性层来融合预测
            # y = torch.cat(output,1)
            # y = self.predict_all(y.transpose(1,2)).transpose(1,2)

            # attention聚合
            # new_y = [self.predict_fusion(i[:,-1,:]) for i in output]
            # score = torch.stack([torch.matmul(i,self.global_token) for i in new_y])
            # score = torch.softmax(score.transpose(0,1),1) #[B, scale_nums, 1]
            # y = [y[i]*score[:,i:i+1] for i in range(self.scale_nums+1)]
            # y = sum(y) / len(y)
            # y = sum(y)



        else:                   # 只用最后一个尺度预测
            y = self.predict[0](output[0].transpose(1, 2)).transpose(1, 2)

        # y=self.norm.renorm(y)

        # trend预测
        # trends = []
        # for i in range(self.channels):
        #     trends.append(self.trend_pred[i](trend[:,:,i]))
        # trends = torch.stack(trends,-1)
        # y = y + trends
        y=y + trend
        return y


