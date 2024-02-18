import pickle
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler
from collections import Counter
from torch.utils.data import Dataset

def get_dataloader(args, data, weighted = False, total = False):

    if total:
        
        if weighted:
            train_label_ids = data['train'].label_ids
            class_counts = Counter(train_label_ids)
            class_weights = {class_label: 1.0 / count for class_label, count in class_counts.items()}
            sample_weights = [class_weights[class_label] for class_label in train_label_ids]
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_label_ids), replacement=True)
            train_dataloader = DataLoader(data['train'], sampler = sampler, batch_size = args.train_batch_size, num_workers = args.num_workers, pin_memory = True)

        else:
            train_dataloader = DataLoader(data['train'], shuffle=True, batch_size = args.train_batch_size, num_workers = args.num_workers, pin_memory = True)

        dev_dataloader = DataLoader(data['dev'], batch_size = args.eval_batch_size, num_workers = args.num_workers, pin_memory = True)
        test_dataloader = DataLoader(data['test'], batch_size = args.eval_batch_size, num_workers = args.num_workers, pin_memory = True)

        dataloader = {
            'train': train_dataloader,
            'dev': dev_dataloader,
            'test': test_dataloader
        }  
        
    else:

        if weighted:
            train_label_ids = data['train'].label_ids
            class_counts = Counter(train_label_ids)
            class_weights = {class_label: 1.0 / count for class_label, count in class_counts.items()}
            sample_weights = [class_weights[class_label] for class_label in train_label_ids]
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_label_ids), replacement=True)
            train_dataloader = DataLoader(data['train'], sampler = sampler, batch_size = args.train_batch_size, num_workers = args.num_workers, pin_memory = True)

        else:
            train_dataloader = DataLoader(data['train'], shuffle=True, batch_size = args.train_batch_size, num_workers = args.num_workers, pin_memory = True)


        dev_dataloader = DataLoader(data['dev'], batch_size = args.eval_batch_size, num_workers = args.num_workers, pin_memory = True)
        
        if args.ood:
            ood_train_dataloader = DataLoader(data['ood_train'], shuffle = True, batch_size = args.train_batch_size, pin_memory = True)
            ood_dev_dataloader = DataLoader(data['ood_dev'], batch_size = args.eval_batch_size, pin_memory = True)

        test_dataloader = DataLoader(data['test'], batch_size = args.eval_batch_size, num_workers = args.num_workers, pin_memory = True)

        
        dataloader = {
            'train': train_dataloader,
            'dev': dev_dataloader,
            'test': test_dataloader
        }  

        if args.ood:
            dataloader.update({
                'ood_train': ood_train_dataloader,
                'ood_dev': ood_dev_dataloader,
            })
    
    return dataloader

def get_v_a_data(data_args, feats_path, max_seq_len):
    
    if not os.path.exists(feats_path):
        raise Exception('Error: The directory of features is empty.')    

    feats = load_feats(data_args, feats_path)
    data = padding_feats(feats, max_seq_len)
    
    return data 
    
def load_feats(data_args, feats_path):

    with open(feats_path, 'rb') as f:
        feats_series = f.read()

    feats = pickle.loads(feats_series)

    train_feats = [feats[x] for x in data_args['train_data_index']]
    dev_feats = [feats[x] for x in data_args['dev_data_index']]
    test_feats = [feats[x] for x in data_args['test_data_index']]

    outputs = {
        'train': train_feats,
        'dev': dev_feats,
        'test': test_feats
    }

    return outputs

def padding(feat, max_length, padding_mode = 'zero', padding_loc = 'end'):
    """
    padding_mode: 'zero' or 'normal'
    padding_loc: 'start' or 'end'
    """
    assert padding_mode in ['zero', 'normal']
    assert padding_loc in ['start', 'end']
    
    if feat.ndim == 1:
        return feat

    length = feat.shape[0]

    if length > max_length:
        return feat[:max_length, :]

    if padding_mode == 'zero':
        pad = np.zeros([max_length - length, feat.shape[-1]])
    elif padding_mode == 'normal':
        mean, std = feat.mean(), feat.std()
        pad = np.random.normal(mean, std, (max_length - length, feat.shape[1]))
    
    if padding_loc == 'start':
        feat = np.concatenate((pad, feat), axis = 0)
    else:
        feat = np.concatenate((feat, pad), axis = 0)

    return feat

def padding_feats(feats, max_seq_len):

    p_feats = {}

    for dataset_type in feats.keys():
        f = feats[dataset_type]

        tmp_list = []
        length_list = []
        
        for x in f:
            x_f = np.array(x) 
            x_f = x_f.squeeze(1) if x_f.ndim == 3 else x_f

            length_list.append(min(len(x_f), max_seq_len))
            p_feat = padding(x_f, max_seq_len)
            tmp_list.append(p_feat)

        p_feats[dataset_type] = {
            'feats': tmp_list,
            'lengths': length_list
        }

    return p_feats   