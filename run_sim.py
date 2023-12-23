import argparse
import os
import torch
from exp.exp_main_sim import Exp_Main_Sim
from exp.exp_main_markov import Exp_Main_Markov
from exp.exp_main_toy_example import Exp_Main_Toy
from utils.simulation import simulate_main
import random
import numpy as np
from utils.tools import set_random_seed

def main():
    #fix_seed = 714
    #random.seed(fix_seed)
    #torch.manual_seed(fix_seed)
    #np.random.seed(fix_seed)
    set_random_seed(714)

    parser = argparse.ArgumentParser(description='Informer for Time Series Simulation')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=0, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--example_name', type=str, required=True, default='Toy_Wind',
                        help='example name')

    # data loader
    parser.add_argument('--root_path', type=str, default='../data/', help='data directory')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='../checkpoints/', help='location of model checkpoints')
    parser.add_argument('--state_init_root_path', type=str, default='../data/wind_data_state_sim_300.pt', help='directory for state init')
    parser.add_argument('--amount_init_root_path', type=str, default='../data/wind_data_amount_sim.pt', help='directory for amount init')
    parser.add_argument('--time_init_root_path', type=str, default='../data/wind_data_time_sim.pt', help='directory for time init')

    # inference task
    parser.add_argument('--seq_len_markov', type=int, default=12, help='Markov p')
    parser.add_argument('--seq_len', type=int, default=48, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=48, help='prediction sequence length')

    # model define
    parser.add_argument('--num_grps', type=int, default=300, help='number of discrete states')
    parser.add_argument('--tail_pct', type=float, default=1/3, help='percentage of states in tail')
    parser.add_argument('--tail_factor_state', type=float, default=1.2, help='amplification factor for states in tail')
    parser.add_argument('--enc_in', type=int, default=6, help='encoder input size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers_markov', type=int, default=3, help='num of encoder layers in Markov model')
    parser.add_argument('--e_layers', type=int, default=4, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=4, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=7, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='exp', help='exp description')
    parser.add_argument('--lradj', type=str, default='type2', help='adjust learning rate')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    args.dec_in = args.enc_in
    args.c_out = args.enc_in

    print('Args in experiment:')
    print(args)
    
    exp = Exp_Main_Sim(args) if args.example_name == 'Toy_Wind' else Exp_Main_Toy(args)
    exp_markov = Exp_Main_Markov(args)

    if args.is_training:
        # setting record of experiments
        setting_markov = '{}_{}_{}_ng{}_sl{}_dm{}_nh{}_el{}_df{}_fc{}_dt{}_{}'.format(
            args.model_id,
            'Markov',
            args.example_name,
            args.num_grps,
            args.seq_len_markov,
            args.d_model,
            args.n_heads,
            args.e_layers_markov,
            args.d_ff,
            args.factor,
            args.des, 0)

        print('>>>>>>>start training for Markov model: {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting_markov))
        exp_markov.train(setting_markov)

        torch.cuda.empty_cache()
        
        # setting record of experiments
        setting = '{}_{}_{}_ng{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_dt{}_{}'.format(
            args.model_id,
            'Informer',
            args.example_name,
            args.num_grps,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.des, 0)

        print('>>>>>>>start training for deep learning model: {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        torch.cuda.empty_cache()

    else:
        setting_markov = '{}_{}_{}_ng{}_sl{}_dm{}_nh{}_el{}_df{}_fc{}_dt{}_{}'.format(
            args.model_id,
            'Markov',
            args.example_name,
            args.num_grps,
            args.seq_len_markov,
            args.d_model,
            args.n_heads,
            args.e_layers_markov,
            args.d_ff,
            args.factor,
            args.des, 0)
        
        setting = '{}_{}_{}_ng{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_dt{}_{}'.format(
            args.model_id,
            'Informer',
            args.example_name,
            args.num_grps,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.des, 0)
        
        state_dict_path = os.path.join(args.checkpoints, setting_markov, 'checkpoint.pth')
        amount_dict_path = os.path.join(args.checkpoints, setting, 'checkpoint.pth')
        
        if args.use_gpu == True:
            exp.model.load_state_dict(torch.load(amount_dict_path))
            exp_markov.model.load_state_dict(torch.load(state_dict_path))
        else:
            exp.model.load_state_dict(torch.load(amount_dict_path, map_location=torch.device('cpu')))
            exp_markov.model.load_state_dict(torch.load(state_dict_path, map_location=torch.device('cpu')))

        print('>>>>>>>simulation : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        state = torch.load(args.state_init_root_path)
        amount = torch.load(args.amount_init_root_path)
        time = torch.load(args.time_init_root_path)
        
        state_sim, amount_Dl, amount_Chol, amount_Rshfl = simulate_main(exp, exp_markov, state, amount, time)

        torch.save(state_sim, os.path.join(args.root_path, 'state_sim_'+setting+'.pt'))
        torch.save(amount_Dl, os.path.join(args.root_path, 'amount_Dl_'+setting+'.pt'))
        torch.save(amount_Chol, os.path.join(args.root_path, 'amount_Chol_'+setting+'.pt'))
        torch.save(amount_Rshfl, os.path.join(args.root_path, 'amount_Rshfl_'+setting+'.pt'))

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
