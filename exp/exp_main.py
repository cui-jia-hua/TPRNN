from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, RNN
from models import Pyraformer, Linear, TPRNN, FEDformer, TCN, LSTNet
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
from torchstat import stat

from tqdm import tqdm
import nni
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'RNN': RNN,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'Pyraformer': Pyraformer,
            'Linear': Linear,
            'TPRNN':TPRNN,
            'FEDformer': FEDformer,
            'TCN':TCN,
            'LSTNet':LSTNet,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=0.0005)  # elect
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        # criterion = nn.L1Loss()
        return criterion

    def _forward_propagation(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        """
        :param batch_x: [Batch, Length, Dims]
        :param batch_x_mark: [Batch, Length, Dims]
        :param dec_inp
        :param batch_y_mark
        :return:
        outputs
        """
        if self.args.output_attention:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            if self.args.model == "Pyraformer" and self.args.decoder_type == "FC":
                # Add a predict token into the history sequence
                predict_token = torch.zeros(batch_x.size(0), 1, batch_x.size(-1), device=batch_x.device)
                batch_x = torch.cat([batch_x, predict_token], dim=1)
                batch_x_mark = torch.cat([batch_x_mark, batch_y_mark[:, 0:1, :]], dim=1)
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        return outputs

    def _preprocess_data(self, df):
        """
        :param df
        :return: (batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp)
        """
        batch_x = df[0].float().to(self.device)
        batch_y = df[1].float()
        batch_x_mark = df[2].float().to(self.device)
        batch_y_mark = df[3].float().to(self.device)

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.output_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        return (batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp)

    def _postprocess_data(self, outputs, batch_y, is_detach = False):
        """
        :param outputs: [Batch, Length, Dims]
        :param batch_y
        :param criterion
        :param is_detach
        :return: (outputs, batch_y)
        """
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.output_len:, f_dim:]
        batch_y = batch_y[:, -self.args.output_len:, f_dim:].to(self.device)

        if is_detach:
            outputs = outputs.detach().cpu()
            batch_y = batch_y.detach().cpu()

        return (outputs, batch_y)


    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, df in enumerate(vali_loader):
                #prepare data
                batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = self._preprocess_data(df)

                # model
                outputs = self._forward_propagation(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # calculate loss
                outputs, batch_y = self._postprocess_data(outputs, batch_y, True)
                loss = criterion(outputs, batch_y)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss


    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()


        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            i=0
            for df in tqdm(train_loader, mininterval=2,
                              desc='  - (Training)   ', leave=False):
                iter_count += 1
                model_optim.zero_grad()
                # prepare data
                batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = self._preprocess_data(df)

                # model
                outputs = self._forward_propagation(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # calculate loss
                outputs, batch_y = self._postprocess_data(outputs, batch_y, False)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 2000 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # backward
                loss.backward()
                model_optim.step()
                i+=1

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            if self.args.use_nni:
                nni.report_intermediate_result(vali_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            # early_stopping(vali_loss, self.model, path, setting)
            early_stopping(vali_loss, self.model, path, setting)
            if early_stopping.early_stop:
                print("Early stopping")
                break


        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model


    def test(self, setting, vis=False, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './results_vis/' + setting + '/'
        if vis and not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, df in enumerate(test_loader):
                # prepare data
                batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp = self._preprocess_data(df)

                # model
                outputs = self._forward_propagation(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # calculate loss
                pred, true = self._postprocess_data(outputs, batch_y, True)
                preds.append(pred.numpy())
                trues.append(true.numpy())
                inputx.append(batch_x.detach().cpu().numpy())
                if vis and i % 200 == 0:
                    input = batch_x.detach().cpu().numpy()
                    for j in range(true.shape[-1]):
                        gt = np.concatenate((input[0, :, j], true[0, :, j]), axis=0)
                        pd = np.concatenate((input[0, :, j], pred[0, :, j]), axis=0)
                        visual(gt, pd, os.path.join(folder_path, str(i) + '_' + str(j) + '.png'))

        if self.args.test_flop:
            test_params_flop(self.model, (batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}, mape:{}'.format(mse, mae, rse, mape))
        if self.args.use_nni:
            nni.report_final_result(mse)
        f = open("result.txt", 'a')
        f.write(setting + str(self.args) + "  \n")
        f.write('mse:{}, mae:{}, rse:{}, mape:{}'.format(mse, mae, rse, mape))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return
