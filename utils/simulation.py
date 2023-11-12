import os
import numpy as np
import pandas as pd
import torch
import math
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

def read_data_for_simulation(df_amount_raw, df_state_raw, num_sample, seq_len):
  # Load data
  df_amount_data = df_amount_raw[df_amount_raw.columns[1:]].to_numpy()
  df_state_data = df_state_raw[df_state_raw.columns[1:]].to_numpy()
  df_time_data = df_state_raw[df_state_raw.columns[:1]].to_numpy()

  # Reshape data
  amount = reshape_2d_to_3d(df_amount_data, num_sample)
  state = reshape_2d_to_3d(df_state_data, num_sample)
  time = reshape_2d_to_3d(df_time_data, num_sample)

  # Prepare amount data (append zeros in case length do not match)
  dim_a = amount.shape
  dim_s = state.shape
  seq_len = min(seq_len, dim_a[1]) # update seq_len
  amount_sim = amount[:, :seq_len, :]
  zeros_tail = torch.zeros(dim_a[0], dim_a[1] - seq_len, dim_a[2]).float()
  amount_sim = torch.cat([amount_sim, zeros_tail], dim=1).float()

  return amount, state, time, amount_sim

def simulate(exp, amount, state, time):
  # Calculate the number of iterations
  d = state.shape
  num_step = d[1]
  num_sample = d[0]
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

def fix_correlation(amount, amount_sim):
  # Reshape amount and amount_sim
  amount_2d = reshape_3d_to_2d(amount, out_type='numpy')
  amount_sim_2d = reshape_3d_to_2d(amount_sim, out_type='numpy')
  
  # Cholesky decomposition
  L_transformer = np.linalg.cholesky(np.cov(amount_sim_2d.T))
  L_target = np.linalg.cholesky(np.cov(amount_2d.T))
  
  # Fix correlation
  # from a numerical analysis POV, avoid
  # amount_sim_fixed = np.matmul(L_target, np.matmul(np.linalg.inv(L_transformer), amount_sim_2d.T)).T
  amount_sim_fixed = np.matmul(L_target, np.linalg.solve(L_transformer, amount_sim_2d.T)).T

  return amount_sim_fixed

def reshuffle(sample_seq, resample_seq):
  idx = resample_seq.argsort()
  # idx_inv = idx.argsort()

  idx_inv = np.empty_like(idx)
  idx_inv[idx] = np.arange(len(resample_seq))
  idx2 = sample_seq.argsort()
  
  return sample_seq[idx2][idx_inv]

def fix_marginal_distribution_gauss(amount_sim, rng_seed=512):
  # Sampling
  d = amount_sim.shape
  np.random.seed(rng_seed)
  y = np.random.normal(0, 1, d)

  # Reshuffle
  amount_fixed = amount_sim
  for i in range(d[1]):
    amount_fixed[:, i] = reshuffle(y[:, i], amount_sim[:, i])

  return amount_fixed

def ecdf(data):
  # Compute ECDF
  x = np.sort(data)
  n = x.size
  y = np.arange(1, n+1) / n
    
  return(x, y)

def estimate_cor(amount, num_sample):
  # Reshape_2d_to_3d if needed
  if len(amount.shape) == 2:
    amount = reshape_2d_to_3d(amount, num_sample, out_type='tensor')
  elif len(amount.shape) == 3 and type(amount) == np.ndarray:
    amount = torch.from_numpy(amount)
  
  # Estimate correlation
  d = amount.shape[2]
  c = [torch.corrcoef(amount[:, :, k].T)[:, 0] for k in range(d)]
  c = torch.stack(c, dim=1)

  return c

  
  

  









