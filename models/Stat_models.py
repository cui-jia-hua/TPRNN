import torch.nn as nn

class Naive_repeat(nn.Module):
    def __init__(self, configs):
        super(Naive_repeat, self).__init__()
        self.output_len = configs.output_len
        
    def forward(self, x):
        B,L,D = x.shape
        x = x[:,-1,:].reshape(B,1,D).repeat(self.output_len,axis=1)
        return x # [B, L, D]


class Naive_mean(nn.Module):
    def __init__(self, configs):
        super(Naive_mean, self).__init__()
        self.output_len = configs.output_len

    def forward(self, x):
        B, L, D = x.shape
        x = x.mean(1).reshape(B, 1, D).repeat(self.output_len, axis=1)
        return x  # [B, L, D]
