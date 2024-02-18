
import os
import torch
import numpy as np
import pandas as pd
import random
import logging
import copy
from .metrics import Metrics, OID_Metrics
import torch
import torch.nn.functional as F

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, args, delta=1e-6, modality = 'text'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.         
        """
        self.patience = args.wait_patience
        self.logger = logging.getLogger(args.logger_name)
        self.monitor = args.eval_monitor
        self.counter = 0
        self.best_score = 1e8 if self.monitor == 'loss' else 1e-6
        self.early_stop = False
        self.delta = delta
        self.best_model = None
        self.modality = modality

    def __call__(self, score, model):
        
        better_flag = score <= (self.best_score - self.delta) if self.monitor == 'loss' else score >= (self.best_score + self.delta) 

        if better_flag:
            self.counter = 0
            self.best_model = copy.deepcopy(model)
            self.best_score = score 

        else:
            self.counter += 1
            self.logger.info(f'{self.modality}: EarlyStopping counter: {self.counter} out of {self.patience}')  

            if self.counter >= self.patience:
                self.early_stop = True
         
def set_torch_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def set_output_path(args, save_model_name):

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    pred_output_path = os.path.join(args.output_path, save_model_name)
    if not os.path.exists(pred_output_path):
        os.makedirs(pred_output_path)

    model_path = os.path.join(pred_output_path, args.model_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    return pred_output_path, model_path

def save_npy(npy_file, path, file_name):
    npy_path = os.path.join(path, file_name)
    np.save(npy_path, npy_file)

def load_npy(path, file_name):
    npy_path = os.path.join(path, file_name)
    npy_file = np.load(npy_path)
    return npy_file

def save_model(model, model_dir):

    save_model = model.module if hasattr(model, 'module') else model 
    model_file = os.path.join(model_dir, 'pytorch_model.bin')

    torch.save(save_model.state_dict(), model_file)

def restore_model(model, model_dir, device):
    output_model_file = os.path.join(model_dir, 'pytorch_model.bin')
    m = torch.load(output_model_file, map_location=device)
    model.load_state_dict(m)
    return model

def save_results(args, test_results, debug_args = None):
    
    save_keys = ['y_pred', 'y_true', 'features', 'scores']
    for s_k in save_keys:
        if s_k in test_results.keys():
            save_path = os.path.join(args.output_path, s_k + '.npy')
            np.save(save_path, test_results[s_k])

    results = {}
    metrics = Metrics(args)
    oid_metrics = OID_Metrics(args)
    
    for key in metrics.eval_metrics:
         if key in test_results.keys():
            results[key] = round(test_results[key] * 100, 2)

    for key in oid_metrics.eval_metrics:
        if key in test_results.keys():
            results[key] = round(test_results[key] * 100, 2)

    if 'best_eval_score' in test_results:
        eval_key = 'eval_' + args.eval_monitor
        results.update({eval_key: test_results['best_eval_score']})

    _vars = [args.dataset, args.ood_dataset, args.method, args.text_backbone, args.video_feats, args.audio_feats, args.seed, args.log_id]
    _names = ['dataset', 'ood_dataset', 'method', 'text_backbone', 'video_feats', 'audio_feats',  'seed', 'log_id']

    if debug_args is not None:
        _vars.extend([args[key] for key in debug_args.keys()])
        _names.extend(debug_args.keys())

    
    vars_dict = {k:v for k,v in zip(_names, _vars)}
    results = dict(results,**vars_dict)

    keys = list(results.keys())
    values = list(results.values())
    
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)
        
    results_path = os.path.join(args.results_path, args.results_file_name)
    
    if not os.path.exists(results_path) or os.path.getsize(results_path) == 0:
        ori = []
        ori.append(values)
        df1 = pd.DataFrame(ori,columns = keys)
        df1.to_csv(results_path,index=False)
    else:
        df1 = pd.read_csv(results_path)
        new = pd.DataFrame(results,index=[1])
        df1 = df1.append(new,ignore_index=True)
        df1.to_csv(results_path,index=False)
    data_diagram = pd.read_csv(results_path)
    
    print('test_results', data_diagram)


def softmax_cross_entropy_with_softtarget(input, num_labels, device):
    """
    :param input: (batch, *)
    :param target: (batch, *) same shape as input, each item must be a valid distribution: target[i, :].sum() == 1.
    """
    ood_length = input.shape[0]
    ood_targets = (1. / num_labels) * torch.ones(ood_length).to(device)

    logprobs = torch.nn.functional.log_softmax(input.view(input.shape[0], -1), dim=1)
    batchloss = - torch.sum(ood_targets.view(ood_targets.shape[0], -1) * logprobs, dim=1)
    
    return torch.mean(batchloss)