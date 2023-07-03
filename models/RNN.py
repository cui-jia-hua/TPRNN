import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):
    """
    Vanilla LSTM
    configs:
    nidden_layer_size: RNN中隐藏层的大小
    input_len: RNN输入长度
    input_size: 输入数据的embedding维度
    output_len: RNN输出（预测）长度
    layer_num: RNN层数
    batch_size: batch size
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.input_len = configs.input_len
        self.input_size = configs.enc_in
        self.output_len = configs.output_len
        self.hidden_layer_size = configs.enc_in
        self.layer_num = configs.layer_num
        self.batch_size = configs.batch_size

        self.lstm = nn.LSTM(self.input_size, self.hidden_layer_size, self.layer_num, batch_first=True)

        self.linear = nn.Linear(self.hidden_layer_size, self.output_len)

        self.hidden_cell = (torch.nn.init.xavier_uniform_(torch.zeros(self.layer_num, self.batch_size, self.hidden_layer_size)).cuda(),
                            torch.nn.init.xavier_uniform_(torch.zeros(self.layer_num, self.batch_size, self.hidden_layer_size)).cuda())


    def forward(self, x, *_):
        # x:[B, S, H] (batch, seq, hidden)
        # 有点问题，不能多变量预测
        lstm_out, hidden_cell = self.lstm(x, self.hidden_cell)
        predictions = self.linear(lstm_out[:,-1,:])
        predictions = predictions.view(*predictions.shape,1)
        return predictions
