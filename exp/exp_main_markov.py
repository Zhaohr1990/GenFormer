from data_provider.data_factory import data_provider_sim
from exp.exp_basic import Exp_Basic
from models import Markov_sim
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.loss_function import focal_loss

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class Exp_Main_Markov(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main_Markov, self).__init__(args)

    def _build_model(self):
        model = Markov_sim.Model(self.args).float()
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider_sim(self.args, flag, if_markov=True)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = focal_loss(alpha=self.args.class_weights.cuda(), gamma=2, num_classes=self.args.num_grps)
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_state_x, batch_state_y, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader): 
                batch_state_y = batch_state_y.int()

                # decoder input
                dec_inp = torch.zeros_like(batch_state_y[:, -self.args.pred_len:, :]).int() + self.args.num_grps
                dec_inp = torch.cat([batch_state_y[:, :self.args.label_len, :], dec_inp], dim=1).int().to(self.device)
                # encoder - decoder
                if self.args.output_attention:
                    outputs = self.model(dec_inp, None, None)[0]
                else:
                    outputs = self.model(dec_inp, None, None)

                outputs = outputs[:, -self.args.pred_len:, :].squeeze(1)
                targets = batch_state_y[:, -self.args.pred_len:, :].squeeze(1).squeeze(1).long().to(self.device)
                
                loss = criterion(outputs, targets)
                loss = loss.item()
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss


    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

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
            for i, (batch_state_x, batch_state_y, batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader): 
                iter_count += 1
                model_optim.zero_grad()
                batch_state_y = batch_state_y.int()

                # decoder input
                dec_inp = torch.zeros_like(batch_state_y[:, -self.args.pred_len:, :]).int() + self.args.num_grps
                dec_inp = torch.cat([batch_state_y[:, :self.args.label_len, :], dec_inp], dim=1).int().to(self.device)
                # encoder - decoder
                if self.args.output_attention:
                    outputs = self.model(dec_inp, None, None)[0]
                else:
                    outputs = self.model(dec_inp, None, None)

                outputs = outputs[:, -self.args.pred_len:, :].squeeze(1)
                targets = batch_state_y[:, -self.args.pred_len:, :].squeeze(1).squeeze(1).long().to(self.device)

                loss = criterion(outputs, targets)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        #best_model_path = path + '/' + 'checkpoint.pth'
        #self.model.load_state_dict(torch.load(best_model_path))

        return

    
    