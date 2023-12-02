import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from utils.preprocess import cluster_MarkovChain_states
#from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

class Dataset_Temperature_Sim(Dataset):
    def __init__(self, root_path, num_grps, flag='train', size=None, scale=True, freq='h'):
        # init
        if size == None:
            self.seq_len = 24 * 2
            self.label_len = 24 * 2
            self.pred_len = 24
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
 
        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.scale = scale
        self.freq = freq
        self.root_path = root_path
        self.__read_data__()
    
    # Function to load data
    def __read_data__(self):
        # load raw data
        df_amount_path = os.path.join(self.root_path, 'guangdong_temp_data_h_normalized_25.csv')
        df_state_path = os.path.join(self.root_path, 'guangdong_temp_state_h_normalized_25.csv')
        df_amount_raw = pd.read_csv(df_amount_path)
        df_state_raw = pd.read_csv(df_state_path)
        df_amount_data = df_amount_raw[df_amount_raw.columns[1:]] # Remove the datetime column
        df_state_data = df_state_raw[df_state_raw.columns[1:]] # Remove the datetime column
        
        # Train test split (for now, test data = validation data, we hardcode the index)
        border1s = [0, 365 * 10 * 24 - self.seq_len, 365 * 10 * 24 - self.seq_len]
        border2s = [365 * 10 * 24, 365 * 11 * 24, 365 * 11 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # State data as integer
        self.state_data = df_state_data.values[border1:border2]

        # Scaling (Normalization) and Amount data as float
        self.scaler = StandardScaler()
        if self.scale:
            train_data = df_amount_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_amount_data.values)
        else:
            data = df_amount_data.values
        self.amount_data = data[border1:border2]

        # Time feature data as float
        df_stamp = df_amount_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
        data_stamp = data_stamp.transpose(1, 0)  
        self.data_stamp = data_stamp
    
    # Function to get item
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x_state = self.state_data[s_begin:s_end]
        seq_y_state = self.state_data[r_begin:r_end]
        seq_x = self.amount_data[s_begin:s_end]
        seq_y = self.amount_data[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x_state, seq_y_state, seq_x, seq_y, seq_x_mark, seq_y_mark
    
    # Function to calculate the length
    def __len__(self):
        return len(self.amount_data) - self.seq_len - self.pred_len + 1
    
    # Function to inverse transform to data based on scaler
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Wind_Sim(Dataset):
    def __init__(self, root_path, num_grps, flag='train', size=None, tail_pct=1/3, freq='h'):   
        # init
        if size == None:
            self.seq_len = 8 * 7 * 2
            self.label_len = 8 * 7
            self.pred_len = 8
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.num_grps = num_grps
        self.set_type = type_map[flag]
        self.freq = freq
        self.root_path = root_path
        self.tail_pct = tail_pct
        self.__read_data__()
    
    ## Function to load data
    def __read_data__(self):
        # Load raw data
        df_amount_path = os.path.join(self.root_path, 'wind_data_fl_6_gauss.csv')
        df_amount_raw = pd.read_csv(df_amount_path)
        df_amount_data = df_amount_raw[df_amount_raw.columns[1:]] # Remove the datetime column
        
        # Generate states by clustering
        # check if file for state clustering exists
        df_state_path = os.path.join(self.root_path, f'state_wind_gauss_{self.num_grps}.csv')
        if os.path.isfile(df_state_path):
            df_state_raw = pd.read_csv(df_state_path)
        else:
            # perform clustering here
            df_state_raw = cluster_MarkovChain_states(df_amount_path, self.num_grps, gaussian_marginal=True, absolute_tail=False, segregate_samples=True, tail_samples_pct=self.tail_pct)
            # save
            df_state_raw.to_csv(df_state_path, sep=',', index=False, encoding='utf-8')
        df_state_data = df_state_raw[df_state_raw.columns[1:]] # Remove the datetime column

        # Train, validate, test 
        # Train test split (for now, we do not have test data)
        self.train_test_split = [0.9, 0.1, 0]
        train_test_cum = np.cumsum(self.train_test_split)
        max_sample = len(df_amount_raw)
        border1s = [0, int(max_sample * train_test_cum[0]), int(max_sample * train_test_cum[1])]
        border2s = [int(max_sample * train_test_cum[0]), int(max_sample * train_test_cum[1]), int(max_sample * train_test_cum[2])]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # State data as integer
        self.state_data = df_state_data.values[border1:border2].astype('int')
        self.amount_data = df_amount_data.values[border1:border2]

        # Time feature data
        df_stamp = df_amount_raw[['time']][border1:border2]
        data_stamp = time_features(pd.to_datetime(df_stamp['time'].values), freq=self.freq)
        data_stamp = data_stamp.transpose(1, 0)
        self.data_stamp = data_stamp

    ## Function to get item
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x_state = self.state_data[s_begin:s_end]
        seq_y_state = self.state_data[r_begin:r_end]
        seq_x = self.amount_data[s_begin:s_end]
        seq_y = self.amount_data[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x_state, seq_y_state, seq_x, seq_y, seq_x_mark, seq_y_mark
    
    ## Function to calculate the length
    def __len__(self):
        return len(self.amount_data) - self.seq_len - self.pred_len + 1

class Dataset_Toy_Example(Dataset):
    def __init__(self, root_path, num_grps, flag='train', num_step=201, num_sample=1000, size=None, tail_pct=1/3):   
        # init
        if size == None:
            self.seq_len = 200
            self.label_len = 200
            self.pred_len = 10
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.root_path = root_path
        self.tail_pct = tail_pct
        self.num_step = num_step
        self.num_sample = num_sample
        self.len_tf_single = self.num_step + 1 - self.seq_len - self.pred_len
        self.sample_id = np.repeat(np.linspace(0, self.num_sample - 1, self.num_sample), self.num_step)
        self.time_id = np.tile(np.linspace(0, self.num_step - 1, self.num_step), self.num_sample)
        self.__read_data__(num_grps)

    ## Function to calculate index
    def __getindex__(self, index):
        index_sample = index // self.len_tf_single
        index_inseries = index - self.len_tf_single * index_sample

        return np.where((self.sample_id == index_sample) & (self.time_id == index_inseries))[0].item()

    ## Function to load data
    def __read_data__(self, num_grps):
        # load raw data
        df_amount_path = os.path.join(self.root_path, 'data_toy_example_gauss.csv')
        df_amount_raw = pd.read_csv(df_amount_path)
        df_amount_data = df_amount_raw[df_amount_raw.columns[1:]] # Remove the datetime column
        df_time_data = df_amount_raw[df_amount_raw.columns[:1]]

        # check if file for state clustering exists
        df_state_path = os.path.join(self.root_path, f'state_toy_example_gauss_{num_grps}.csv')
        if os.path.isfile(df_state_path):
            df_state_raw = pd.read_csv(df_state_path)
        else:
            # perform clustering here
            df_state_raw = cluster_MarkovChain_states(df_amount_path, num_grps, gaussian_marginal=True, absolute_tail=False, segregate_samples=True, tail_samples_pct=self.tail_pct)
            # save
            df_state_raw.to_csv(df_state_path, sep=',', index=False, encoding='utf-8')
        df_state_data = df_state_raw[df_state_raw.columns[1:]] # Remove the datetime column
        
        # Train, validate, test 
        # Train test split (for now, we do not have test data)
        self.train_test_split = [0.9, 0.1, 0]
        train_test_cum = np.cumsum(self.train_test_split)
        border1s = [0, int(self.num_step * self.num_sample * train_test_cum[0]), int(self.num_step * self.num_sample * train_test_cum[1])]
        border2s = [int(self.num_step * self.num_sample * train_test_cum[0]), int(self.num_step * self.num_sample * train_test_cum[1]), int(self.num_step * self.num_sample * train_test_cum[2])]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # State data as integer
        self.state_data = df_state_data.values[border1:border2].astype('int')
        self.amount_data = df_amount_data.values[border1:border2]
        self.data_stamp = (df_time_data.values[border1:border2] * 1000).astype('int')
        
    ## Function to get item
    def __getitem__(self, index):
        s_begin = self.__getindex__(index)
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x_state = self.state_data[s_begin:s_end]
        seq_y_state = self.state_data[r_begin:r_end]
        seq_x = self.amount_data[s_begin:s_end]
        seq_y = self.amount_data[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x_state, seq_y_state, seq_x, seq_y, seq_x_mark, seq_y_mark

    ## Function to calculate the length
    def __len__(self):
        return int(self.len_tf_single * self.num_sample * self.train_test_split[self.set_type])