import argparse
import os
import torch
from exp.exp_main import Exp_Main
from utils.model_args_elect import *
import random
import nni
import time
import numpy as np



def fix_seed(fix_seed=2021):
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    os.environ['PYTHONHASHSEED'] = str(fix_seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.


def set_parser(parser):
    # basic config
    parser.add_argument('--use_nni', action="store_true", default=False, help='use nni?')
    parser.add_argument('--model', type=str, required=True, default='TPRNN', help='model type(for finding args)')

    # data loader
    parser.add_argument('--data_loader_type', type=str, required=True, default='custom',
                        help='dataset type [ETTh1,ETTh2,ETTm1,ETTm2,custom]')
    parser.add_argument('--data', type=str, required=True, default='elect',
                        help='dataset type [elect,ETTh1,ETTh2,ETTm1,ETTm2,weather,ili,exchange_rate,traffic]')
    parser.add_argument('--data_scale', type=str, default="True", help='whether to scale data before input model')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--checkpoints', type=str, default='./model_checkpoints/', help='location of model checkpoints')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, timeF=time embedding, other=kind like onehot embedding')

    # forecasting task(must)
    parser.add_argument('--input_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0,
                        help='start token length, (decoder input data which already known)')
    parser.add_argument('--output_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--vis', type=bool, default=False, help='whether to visual result')

    # models
    parser.add_argument('--multi_scale', type=bool, default=True, help='whether to use all scale to forecast')
    parser.add_argument('--scale_interact', type=str, default="All", help='All, Inner,None')
    parser.add_argument('--CSICB', type=str, default="All", help='All,Conv,Pooling')


    # optimization
    parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    tmp_args = parser.parse_args()

    # model parameters
    model_args = eval(tmp_args.model + "_args")
    # NNI
    if tmp_args.use_nni:
        nni_args = nni.get_next_parameter()
        print("use nni args:")
        print(nni_args)
        for i in nni_args.keys():
            model_args[i] = nni_args[i]


    for i, j in model_args.items():
        parser.add_argument(i, type=type(j), default=j)

    # args
    args = parser.parse_args()

    # GPU
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # datasets
    from utils.dataset_params import data_parser
    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info['data_path']
        args.target = data_info['target']
        args.enc_in, args.dec_in, args.c_out = data_info[args.features]
        args.freq = data_info['freq']
        if args.data_loader_type == "custom_s":
            args.enc_in, args.dec_in, args.c_out = 1, 1, 1
        # Pyraformer, Transformer_t, Transformer_t_s
        if args.freq == "d":
            args.covariate_size = 3

    print('Args in experiment:')
    print(args)
    return args


if __name__ == "__main__":
    fix_seed(42)

    parser = argparse.ArgumentParser(description='Time Series Forecasting Baseline')
    args = set_parser(parser)

    Exp = Exp_Main

    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_{}_{}_trainepochs_{}_{}'.format(
            args.model,
            args.data,
            args.data_loader_type,
            args.input_len,
            args.output_len,
            args.train_epochs,
            time.strftime("%Y%m%d %T")
        )

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, args.vis)

        torch.cuda.empty_cache()

