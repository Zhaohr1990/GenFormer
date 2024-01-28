import os
import numpy as np
import pandas as pd
import torch
import math
from utils.timefeatures import time_features
from utils.tools import set_random_seed
import warnings

warnings.filterwarnings('ignore')

def reshape_2d_to_3d(data_in, num_sample, out_type='tensor'):
  # Convert to np
  if type(data_in) == torch.Tensor:
    pt = data_in.numpy()
  elif type(data_in) == np.ndarray:
    pt = data_in
  else:
    raise ValueError('Data type not supported')
  
  # Calculate number of time steps
  d = pt.shape[0]
  num_step = int(np.floor(d / num_sample))

  # Truncate numpy array
  pt = pt[:(num_step * num_sample), :]

  # Reshape numpy array
  pt = np.transpose(pt).reshape(-1)
  pt = np.transpose(pt.reshape([-1, num_sample, num_step]), [1, 2, 0])

  # Convert back to tensor if needed
  if out_type == 'tensor':
    pt = torch.from_numpy(pt)

  return pt

def reshape_3d_to_2d(data_in, out_type='tensor'):
  # Convert to np
  if type(data_in) == torch.Tensor:
    pt = data_in.numpy()
  elif type(data_in) == np.ndarray:
    pt = data_in
  else:
    raise ValueError('Data type not supported')
  
  # Reshape
  d = pt.shape[2]
  a = [pt[:, :, k].reshape(-1) for k in range(d)]
  amount_2d = np.stack(a, axis=1)

  # Convert back to tensor if needed
  if out_type == 'tensor':
    amount_2d = torch.from_numpy(amount_2d)

  return amount_2d

def simulate_markov(exp_markov, state):
  """
    Simulate a Markov state sequence.
    exp_markov:
    state:
  """
  # Define softmax
  s_m = torch.nn.Softmax(dim=2)
  
  # Calculate the number of iterations
  d = state.shape
  start_pt = exp_markov.args.seq_len - exp_markov.args.seq_len_markov
  num_autoreg = d[1] - start_pt

  # Simulate
  exp_markov.model.eval()
  with torch.no_grad():
    for i in range(num_autoreg):
      s_begin = i + start_pt
      s_end = s_begin + exp_markov.args.seq_len_markov
      r_begin = s_end - exp_markov.args.seq_len_markov
      r_end = r_begin + exp_markov.args.seq_len_markov + 1

      batch_state_y = state[:, r_begin:r_end, :].int()
      dec_inp = torch.zeros_like(batch_state_y[:, -1:, :]).int() + exp_markov.args.num_grps
      dec_inp = torch.cat([batch_state_y[:, :exp_markov.args.seq_len_markov, :], dec_inp], dim=1).int().to(exp_markov.device)

      outputs = exp_markov.model(dec_inp, None, None)
      outputs_softmax = s_m(outputs)
      state[:, s_end:r_end, :] = torch.multinomial(outputs_softmax.squeeze(1), 1).unsqueeze(2)

  return state

def simulate(exp, amount, state, time):
  """
    Utilize the deep learning model to simulate amount sequences.
    exp:
    amount:
    state:
    time:
  """
  # Calculate the number of iterations
  d = state.shape
  num_step = d[1]
  num_autoreg = int(np.floor((num_step - exp.args.seq_len) / exp.args.pred_len))

  # Simulate
  exp.model.eval()
  with torch.no_grad():
    for i in range(num_autoreg):
      s_begin = i * exp.args.pred_len
      s_end = s_begin + exp.args.seq_len
      r_begin = s_end - exp.args.label_len
      r_end = r_begin + exp.args.label_len + exp.args.pred_len

      batch_state_x = state[:, s_begin:s_end, :].int().to(exp.device)
      batch_state_y = state[:, r_begin:r_end, :].int().to(exp.device)
      batch_x = amount[:, s_begin:s_end, :].float().to(exp.device)
      batch_y = amount[:, r_begin:r_end, :].float().to(exp.device)
      batch_x_mark = time[:, s_begin:s_end, :].float().to(exp.device)
      batch_y_mark = time[:, r_begin:r_end, :].float().to(exp.device)

      outputs = exp.model(batch_state_x, batch_x, batch_x_mark, batch_state_y, batch_y, batch_y_mark)
      amount[:, s_end:r_end, :] = outputs[:, -exp.args.pred_len:, :]

  return amount

def fix_correlation(exp, amount_sim):
  """
    Correct spatial correlation by applying a transformation based on Cholesky decomposition.
    exp:
    amount_sim:
  """
  # Reshape amount_sim
  amount_sim_2d = reshape_3d_to_2d(amount_sim, out_type='numpy')
  
  # Gather amount data
  train_set, _ = exp._get_data(flag='train', if_markov=False)
  vali_set, _ = exp._get_data(flag='val', if_markov=False)
  test_set, _ = exp._get_data(flag='test', if_markov=False)
  amount_2d = np.concatenate((train_set.amount_data, vali_set.amount_data, test_set.amount_data), axis=0)
  
  # Cholesky decomposition
  L_transformer = np.linalg.cholesky(np.cov(amount_sim_2d.T))
  L_target = np.linalg.cholesky(np.cov(amount_2d.T))
  
  # Fix correlation
  # from a numerical analysis POV, avoid
  # amount_sim_fixed = np.matmul(L_target, np.matmul(np.linalg.inv(L_transformer), amount_sim_2d.T)).T
  amount_sim_fixed = np.matmul(L_target, np.linalg.solve(L_transformer, amount_sim_2d.T)).T

  return amount_sim_fixed

def reshuffle(sample_seq, resample_seq):
  """
    Perform reshuffling to correct the marginal distribution.
    sample_seq:
    resample_seq:
  """
  idx = resample_seq.argsort()
  # idx_inv = idx.argsort()

  idx_inv = np.empty_like(idx)
  idx_inv[idx] = np.arange(len(resample_seq))
  idx2 = sample_seq.argsort()
  
  return sample_seq[idx2][idx_inv]

def fix_marginal_distribution_gauss(amount_sim, rng_seed=512):
  """
    Correct marginal distribution by performing reshuffling.
    amount_sim:
    rng_seed: 
  """
  # Sampling
  d = amount_sim.shape
  np.random.seed(rng_seed)

  # Reshuffle
  amount_fixed = np.copy(amount_sim)
  for i in range(d[1]):
    y = np.random.normal(0, 1, d[0])
    amount_fixed[:, i] = reshuffle(y, amount_sim[:, i])

  return amount_fixed

def simulate_main(exp, exp_markov, state, amount, time):
  """
    Perform inference to generate new synthetic realizations.
    exp: trained deep learning object
    exp_markov: trained Markov sequence generator
    state:
    amount:
    time: 
  """
  print("Simulate state sequence from Markov model")
  set_random_seed(0)
  state = simulate_markov(exp_markov, state)
  
  print("Infer amount sequence from the deep learning model")
  set_random_seed(512)
  amount_Dl = simulate(exp, amount, state, time)

  print("Correct correlation")
  amount_Chol = fix_correlation(exp, amount_Dl)

  print("Correct Gaussian marginal distribution") 
  set_random_seed(54321)
  amount_Rshfl = fix_marginal_distribution_gauss(amount_Chol)

  # Reshape amount
  num_sample = state.shape[0]
  amount_Rshfl = reshape_2d_to_3d(amount_Rshfl, num_sample, out_type='tensor')

  return state, amount_Dl, amount_Chol, amount_Rshfl

def ecdf(data):
  # Compute ECDF
  x = np.sort(data)
  n = x.size
  y = np.arange(1, n+1) / n
    
  return(x, y)

def estimate_cor(amount, num_sample):
  # Reshape_2d_to_3d if needed
  if len(amount.shape) == 2:
    amount = reshape_2d_to_3d(amount, num_sample, out_type='numpy')
  elif len(amount.shape) == 3 and type(amount) != np.ndarray:
    amount = amount.numpy()
  
  # Estimate correlation
  d = amount.shape 
  c = [[np.corrcoef(amount[:, l:, k].flatten(), amount[:, :(d[1]-l), k].flatten())[0, 1] for k in range(d[2])] for l in range(d[1])]

  return np.stack(c)

def prepare_amount(df_amount_raw, num_sample_hist, seq_len):
  # Load data
  df_amount_data = df_amount_raw[df_amount_raw.columns[1:]].to_numpy()

  # Reshape data
  amount = reshape_2d_to_3d(df_amount_data, num_sample_hist)

  # Calculate replicate number
  num_step = amount.shape[1]
  rep_num = int(np.floor(num_step / seq_len))

  # Replicate
  amount_sim = amount[:, :seq_len, :]
  amount_sim = torch.repeat_interleave(amount_sim, rep_num, dim=0)
  dim_a = amount_sim.shape
  zeros_tail = torch.zeros(dim_a[0], num_step - seq_len, dim_a[2]).float()
  amount_sim = torch.cat([amount_sim, zeros_tail], dim=1).float()

  return amount_sim

def read_data_for_simulation(df_amount_raw, df_state_raw, num_sample, seq_len, freq='h'):
  # Load data
  df_amount_data = df_amount_raw[df_amount_raw.columns[1:]].to_numpy()
  df_state_data = df_state_raw[df_state_raw.columns[1:]].to_numpy()
  df_time_data = df_state_raw[df_state_raw.columns[:1]]
  
  if df_time_data.time.dtype == "object":
    data_stamp = time_features(pd.to_datetime(df_time_data['time'].values), freq=freq)
    data_stamp = data_stamp.transpose(1, 0)
  else:
    data_stamp = df_time_data.to_numpy()

  # Reshape data
  amount = reshape_2d_to_3d(df_amount_data, num_sample)
  state = reshape_2d_to_3d(df_state_data, num_sample)
  time = reshape_2d_to_3d(data_stamp, num_sample)

  # Prepare amount data (append zeros in case length do not match)
  dim_a = amount.shape
  dim_s = state.shape
  seq_len = min(seq_len, dim_a[1]) # update seq_len
  amount_sim = amount[:, :seq_len, :]
  zeros_tail = torch.zeros(dim_a[0], dim_a[1] - seq_len, dim_a[2]).float()
  amount_sim = torch.cat([amount_sim, zeros_tail], dim=1).float()

  return amount, state, time, amount_sim  









