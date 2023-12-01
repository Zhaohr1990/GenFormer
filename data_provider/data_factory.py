from data_provider.data_loader import Dataset_Temperature_Sim, Dataset_Wind_Sim, Dataset_Toy_Example
from torch.utils.data import DataLoader

"""
def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
"""

def data_provider_sim(args, flag, if_markov=False):
    '''flag: train/val/test, if_markov is true when loading data for markov model, else for informer model'''

    # set shuffle and drop_last
    if flag == 'test' or flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    
    # construct dataset
    
    if args.example_name == 'Toy_Temperature':
        data_set = Dataset_Temperature_Sim(
            root_path=args.root_path,
            num_grps=args.num_grps,
            flag=flag,
            size=[args.seq_len_markov, args.seq_len_markov, 1] if if_markov else [args.seq_len, args.label_len, args.pred_len],
            freq=freq
        )
    elif args.example_name == 'Toy_Wind':
        data_set = Dataset_Wind_Sim(
            root_path=args.root_path,
            num_grps=args.num_grps,
            flag=flag,
            size=[args.seq_len_markov, args.seq_len_markov, 1] if if_markov else [args.seq_len, args.label_len, args.pred_len],
            freq=freq
        )
    elif args.example_name == 'Toy_Example':
        data_set = Dataset_Toy_Example(
            root_path=args.root_path,
            num_grps=args.num_grps,
            flag=flag,
            size=[args.seq_len_markov, args.seq_len_markov, 1] if if_markov else [args.seq_len, args.label_len, args.pred_len],
        )
    else:
        raise ValueError('Example not found.')
    
    # construct dataloader
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)

    return data_set, data_loader


    
