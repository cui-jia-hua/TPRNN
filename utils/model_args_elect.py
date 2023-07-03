"""
'RNN': RNN,
'Autoformer': Autoformer,
'Transformer': Transformer,
'Informer': Informer,
'DLinear': DLinear,
'Pyraformer': Pyraformer,
'Transformer_t':Transformer_truncated
"""

# RNN
RNN_args = {
    "--hidden_layer_size":256, # RNN hidden size
    "--layer_num":2, # RNN layer size
    "--output_attention": False, # whether to output attention in encoder
    "--learning_rate": 0.0001,
}

# LSTNet
LSTNet_args = {
    "--hidRNN":100,
    "--hidCNN":100,
    "--CNN_kernel":6,
    "--skip":24,
    "--hidSkip":5,
    "--highway_window": 24,
    "--learning_rate": 0.0001,
    "--dropout": 0.3,
    "--output_fun": 'sigmoid',
    "--output_attention": False,  # whether to output attention in encoder
}

# DLinear
DLinear_args = {
    "--individual": False, # DLinear: a linear layer for each variate(channel) individually
    "--output_attention": False, # whether to output attention in encoder
    "--learning_rate": 0.0001,
}

# TCN
TCN_args = {
    "--hidden_channels":"[32,32,32]",
    "--kernel_size":3,
    "--dropout_rate":0.5,
    "--learning_rate": 0.0001,
    "--output_attention": False,  # whether to output attention in encoder
}

# Linear
Linear_args = {
    "--output_attention": False, # whether to output attention in encoder
    "--learning_rate": 0.0001,
    "--CSCM": 'Bottleneck_Construct',
    "--d_model": 512,       # dimension of model hidden layers
    # CSCM structure. selection: [Bottleneck_Construct, Conv_Construct, MaxPooling_Construct, AvgPooling_Construct]
    "--window_size": "[2,2,2]",  # The number of children of a parent node.
    "--dropout": 0.05,      # dropout rate
    "--d_bottleneck": 128,  # bottelneck
}

# Autoformer
Autoformer_args = {
    "--embed_type": 0, # 0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding
    "--d_model": 512,       # dimension of model hidden layers
    "--n_heads": 6,         # num of heads
    "--e_layers": 2,        # num of encoder layers
    "--d_layers": 1,        # num of decoder layers
    "--d_ff": 2048,         # dimension of fcn
    "--moving_avg": 25,     # window size of moving average for decompose
    "--factor": 1,
    "--dropout": 0.05,      # dropout rate
    "--activation": "gelu", # activation
    "--output_attention": True, # whether to output attention in encoder
    "--learning_rate": 0.0001,
}

# Informer
Informer_args = {
    "--mix": True,          # use mix attention in generative decoder
    "--embed_type": 0,      # 0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding
    "--d_model": 512,       # dimension of model hidden layers
    "--n_heads": 6,         # num of heads
    "--e_layers": 2,        # num of encoder layers
    "--d_layers": 1,        # num of decoder layers
    "--d_ff": 2048,         # dimension of fcn
    "--factor": 3,
    "--dropout": 0.05,      # dropout rate
    "--activation": "gelu", # activation
    "--output_attention": True, # whether to output attention in encoder
    "--distil": True,       # whether to distil
    "--learning_rate": 0.0001,
}

# Transformer
Transformer_args = {
    "--embed_type": 0, # 0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding
    "--d_model": 512,       # dimension of model hidden layers
    "--n_heads": 6,         # num of heads
    "--e_layers": 2,        # num of encoder layers
    "--d_layers": 1,        # num of decoder layers
    "--d_ff": 2048,         # dimension of fcn
    "--factor": 5,
    "--dropout": 0.05,      # dropout rate
    "--activation": "gelu", # activation
    "--output_attention": True, # whether to output attention in encoder
    "--learning_rate": 0.0001,
}

# Pyraformer
Pyraformer_args = {
    "--d_model": 512,       # dimension of model hidden layers
    "--d_inner_hid": 256,       # dimension of model hidden layers
    "--n_heads": 6,  # num of heads
    "--e_layers": 2,  # num of encoder layers
    "--dropout": 0.05,  # dropout rate
    "--output_attention": False,  # whether to output attention in encoder
    "--decoder_type": "FC",  # pyraformer: choose which decoder to predict, [attention,FC]
    "--d_k": 128,  #
    "--d_v": 128,  #
    "--covariate_size": 4,
    "--seq_num": 321,
    "--d_bottleneck": 128,  #
    "--window_size": "[4, 4, 4]",  # The number of children of a parent node.
    "--inner_size": 3,  # The number of ajacent nodes.
    "--CSCM": 'Bottleneck_Construct',  # CSCM structure. selection: [Bottleneck_Construct, Conv_Construct, MaxPooling_Construct, AvgPooling_Construct]
    "--truncate": False,  # Whether to remove coarse-scale nodes from the attention structure
    "--learning_rate": 0.0001,
}

#FEDformer
FEDformer_args = {
    "--version": "Fourier",
    "--mode_select": "random",
    "--modes": 64,
    "--L": 3,
    "--base": "legendre",
    "--cross_activation": "tanh",
    "--embed_type": 0,    # 0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding
    "--d_model": 512,  # dimension of model hidden layers
    "--n_heads": 8,  # num of heads
    "--e_layers": 2,  # num of encoder layers
    "--d_layers": 1,  # num of decoder layers
    "--d_ff": 2048,  # dimension of fcn
    "--moving_avg": 25,  # window size of moving average for decompose
    "--factor": 1,
    "--dropout": 0.05,  # dropout rate
    "--activation": "gelu",  # activation
    "--output_attention": True,  # whether to output attention in encoder
    "--learning_rate": 0.0001,
}

#TPRNN
TPRNN_args = {
    "--output_attention": False,    # whether to output attention in encoder
    "--window_size": "[2,2,2]",   # ETTh2,ETTm2
    "--learning_rate": 0.001,
    "--dropout_rate": 0.7,
    "--hidden_innerscale_size": 0,  # 0=c_in
    "--hidden_interscale_size": 6
}