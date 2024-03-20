import os
import logging
import csv
import random
import numpy as np

from .mm_pre import MMDataset
from .text_pre import get_t_data
from .utils import get_v_a_data
from .text_pre import TextDataset
from .__init__ import benchmarks

__all__ = ['DataManager']

class DataManager:
    
    def __init__(self, args):
        
        self.logger = logging.getLogger(args.logger_name)

        bm = benchmarks[args.dataset]
        max_seq_lengths = bm['max_seq_lengths']
        feat_dims = bm['feat_dims']

        args.text_seq_len = max_seq_lengths['text']
        args.video_seq_len = max_seq_lengths['video'][args.video_feats]
        args.audio_seq_len = max_seq_lengths['audio'][args.audio_feats]

        args.text_feat_dim = feat_dims['text'][args.text_backbone]
        args.video_feat_dim = feat_dims['video'][args.video_feats]
        args.audio_feat_dim = feat_dims['audio'][args.audio_feats]

        self.label_list = bm["intent_labels"]
        self.logger.info('Lists of intent labels are: %s', str(self.label_list))  
        
        args.ood_label_id = len(self.label_list)
        args.num_labels = len(self.label_list) 
         
        self.data = prepare_data(args, self.logger, self.label_list, bm)

def prepare_data_total(args, logger, label_list, bm):          
    
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i
        
    data_path = os.path.join(args.data_path, 'total')
    ood_bm = bm['ood_data'][args.ood_dataset]
    label_map[ood_bm['ood_label']] = args.ood_label_id

    outputs = get_data(args, logger, data_path, bm, label_map) 
    train_label_ids, dev_label_ids, test_label_ids = outputs['train_label_ids'], outputs['dev_label_ids'], outputs['test_label_ids']

    if args.method in ['text', 'text_ood']:

        text_data = outputs['text_data']

        text_train_data = TextDataset(train_label_ids, text_data['train'])
        text_dev_data = TextDataset(dev_label_ids, text_data['dev'])
        text_test_data = TextDataset(test_label_ids, text_data['test'])

        data = {'train': text_train_data, 'dev': text_dev_data, 'test': text_test_data}

    else:
        
        text_data, video_data, audio_data = outputs['text_data'], outputs['video_data'], outputs['audio_data']
        mm_train_data = MMDataset(train_label_ids, text_data['train'], video_data['train'], audio_data['train'])
        mm_dev_data = MMDataset(dev_label_ids, text_data['dev'], video_data['dev'], audio_data['dev'])
        mm_test_data = MMDataset(test_label_ids, text_data['test'], video_data['test'], audio_data['test'])

        data = {'train': mm_train_data, 'dev': mm_dev_data, 'test': mm_test_data}    
        
    return data

def prepare_data(args, logger, label_list, bm):          
    
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i
        
    data_path = os.path.join(args.data_path, 'in-scope')

    ind_outputs = get_data(args, logger, data_path, bm, label_map) 
    train_label_ids, dev_label_ids, test_label_ids = ind_outputs['train_label_ids'], ind_outputs['dev_label_ids'], ind_outputs['test_label_ids']

    if args.method in ['text', 'text_ood']:

        text_data = ind_outputs['text_data']

        text_train_data = TextDataset(train_label_ids, text_data['train'])
        text_dev_data = TextDataset(dev_label_ids, text_data['dev'])

        data = {'train': text_train_data, 'dev': text_dev_data}

    else:
        text_data, video_data, audio_data = ind_outputs['text_data'], ind_outputs['video_data'], ind_outputs['audio_data']
        mm_train_data = MMDataset(train_label_ids, text_data['train'], video_data['train'], audio_data['train'])
        mm_dev_data = MMDataset(dev_label_ids, text_data['dev'], video_data['dev'], audio_data['dev'])

        data = {'train': mm_train_data, 'dev': mm_dev_data}    
        
    if args.ood:
        
        ood_data_path = os.path.join(args.data_path, 'out-of-scope', args.ood_dataset)
        ood_bm = bm['ood_data'][args.ood_dataset]
        label_map[ood_bm['ood_label']] = args.ood_label_id
        
        ood_outputs = get_data(args, logger, ood_data_path, ood_bm, label_map)

        ood_train_label_ids, ood_dev_label_ids, ood_test_label_ids = ood_outputs['train_label_ids'], ood_outputs['dev_label_ids'], ood_outputs['test_label_ids']
        test_label_ids.extend(ood_test_label_ids)

        if args.method in ['text', 'text_ood']:

            ood_text_data = ood_outputs['text_data']
            ood_text_train_data = TextDataset(ood_train_label_ids, ood_text_data['train'])
            ood_text_dev_data = TextDataset(ood_dev_label_ids, ood_text_data['dev'])

            text_data['test'].extend(ood_text_data['test'])
            text_test_data = TextDataset(test_label_ids, text_data['test'])
            
            data.update({
                'ood_train': ood_text_train_data,
                'ood_dev': ood_text_dev_data,
                'test': text_test_data
            })

        else:

            ood_text_data, ood_video_data, ood_audio_data = ood_outputs['text_data'], ood_outputs['video_data'], ood_outputs['audio_data']

            ood_mm_train_data = MMDataset(ood_train_label_ids, ood_text_data['train'], ood_video_data['train'], ood_audio_data['train'])
            ood_mm_dev_data = MMDataset(ood_dev_label_ids, ood_text_data['dev'], ood_video_data['dev'], ood_audio_data['dev'])

            text_data['test'].extend(ood_text_data['test'])

            video_data['test']['feats'].extend(ood_video_data['test']['feats'])
            video_data['test']['lengths'].extend(ood_video_data['test']['lengths'])

            audio_data['test']['feats'].extend(ood_audio_data['test']['feats'])
            audio_data['test']['lengths'].extend(ood_audio_data['test']['lengths'])

            mm_test_data = MMDataset(test_label_ids, text_data['test'], video_data['test'], audio_data['test'])

            data.update({
                'ood_train': ood_mm_train_data,
                'ood_dev': ood_mm_dev_data,
                'test': mm_test_data
            })

    return data
                     
def get_data(args, logger, data_path, bm, label_map):
    
    logger.info('Data preparation...')
    
    train_data_index, train_label_ids = get_indexes_annotations(args, bm, label_map, os.path.join(data_path, 'train.tsv'), args.data_mode)
    dev_data_index, dev_label_ids = get_indexes_annotations(args, bm, label_map, os.path.join(data_path, 'dev.tsv'), args.data_mode)
    test_data_index, test_label_ids = get_indexes_annotations(args, bm, label_map, os.path.join(data_path, 'test.tsv'), args.data_mode)
    args.num_train_examples = len(train_data_index)
    
    data_args = {
        'data_path': data_path,
        'train_data_index': train_data_index,
        'dev_data_index': dev_data_index,
        'test_data_index': test_data_index,
        'label_map': label_map
    }
    
    if args.method in ['text', 'text_ood']:

        text_data = get_t_data(args, data_args)

        outputs = {
            'text_data': text_data,
            'train_label_ids': train_label_ids,
            'dev_label_ids': dev_label_ids,
            'test_label_ids': test_label_ids
        }


    else:
        text_data = get_t_data(args, data_args)
    
        video_feats_path = os.path.join(data_args['data_path'], args.video_data_path, args.video_feats_path)
        video_data = get_v_a_data(data_args, video_feats_path, args.video_seq_len)
        audio_feats_path = os.path.join(data_args['data_path'], args.audio_data_path, args.audio_feats_path)
      
        audio_data = get_v_a_data(data_args, audio_feats_path, args.audio_seq_len)  
     
        outputs = {
            'text_data': text_data,
            'video_data': video_data,
            'audio_data': audio_data,
            'train_label_ids': train_label_ids,
            'dev_label_ids': dev_label_ids,
            'test_label_ids': test_label_ids
        }
        
    return outputs
    
def get_indexes_annotations(args, bm, label_map, read_file_path, data_mode):

    with open(read_file_path, 'r') as f:

        data = csv.reader(f, delimiter="\t")
        indexes = []
        label_ids = []

        for i, line in enumerate(data):
            if i == 0:
                continue

            if args.dataset == 'MIntRec':
                index = '_'.join([line[0], line[1], line[2]])
                
                indexes.append(index)
                label_id = label_map[line[4]]
                
            elif args.dataset in ['MIntRec2.0']:
                index = '_'.join(['dia' + str(line[0]), 'utt' + str(line[1])])
                indexes.append(index)
                label_id = label_map[line[3]]
               
            
            label_ids.append(label_id)
    
    return indexes, label_ids