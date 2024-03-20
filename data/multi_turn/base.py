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

        if args.multiturn:
            self.data = prepare_multiturn_data(args, self.logger, self.label_list, bm)

        else:
            self.data = prepare_data(args, self.logger, self.label_list, bm)

def dialogue_merge(outputs, mode, data_args, elem):

    temp = {}
    temp_utt_id = {}

    for i, (key, v) in enumerate(zip(data_args[mode + '_data_index'], elem)):

        dia_id = key.split('_')[0][3:]
        utt_id = int(key.split('_')[1][3:])

        if dia_id not in temp.keys():
            temp[dia_id] = []
            temp_utt_id[dia_id] = {}

        if utt_id not in temp_utt_id[dia_id].keys():
            temp_utt_id[dia_id][utt_id] = []

        temp_utt_id[dia_id][utt_id].append(v)

    for k in temp_utt_id.keys():
        sorted_temp = []
        for j in sorted (temp_utt_id[k]) : 
            sorted_temp.append(temp_utt_id[k][j][0])
            
        temp[k] = sorted_temp
    
    keys = list(temp.keys())

    new_keys = {keys[i]: i for i in range(len(keys))}

    return new_keys, temp

def singleturn2multiturn(args, outputs, data_args):

    modality_list = []
    speaker_ids_list = []
    label_ids_list = []

    for key in outputs.keys():

        if key in ['text_data', 'video_data', 'audio_data']:
            modality_list.append(key)
        if key.endswith('speaker_ids'):
            speaker_ids_list.append(key)
        if key.endswith('label_ids'):
            label_ids_list.append(key)

    for mode in ['train', 'dev', 'test'] :

        for modality in modality_list: 
        
            if modality == 'text_data':

                feats = outputs[modality][mode]
                keys, infos = dialogue_merge(outputs, mode, data_args, feats)
                results = {keys[k]: v for k, v in infos.items()}
                outputs[modality][mode] = results

            else:
                feats = outputs[modality][mode]['feats']
                keys, infos = dialogue_merge(outputs, mode, data_args, feats)
                results = {keys[k]: v for k, v in infos.items()}
                outputs[modality][mode]['feats'] = results

                lengths = outputs[modality][mode]['lengths']
                keys, infos = dialogue_merge(outputs, mode, data_args, lengths)
                results = {keys[k]: v for k, v in infos.items()}
                outputs[modality][mode]['lengths'] = results

    for speaker_ids_name in speaker_ids_list:

        speaker_ids = outputs[speaker_ids_name]
        keys, infos = dialogue_merge(outputs, speaker_ids_name.split('_')[0], data_args, speaker_ids)
        results = {keys[k]: v for k, v in infos.items()}

        outputs[speaker_ids_name] = results

    for label_ids_name in label_ids_list:

        label_ids = outputs[label_ids_name]
        keys, infos = dialogue_merge(outputs, label_ids_name.split('_')[0], data_args, label_ids)
        results = {keys[k]: v for k, v in infos.items()}      

        outputs[label_ids_name] = results

    return outputs

def prepare_multiturn_data(args, logger, label_list, bm): 

    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i

    ood_bm = bm['ood_data'][args.ood_dataset]
    label_map[ood_bm['ood_label']] = args.ood_label_id

    speaker_map = {}
    for i, speaker_name in enumerate(bm['speaker_list']):
        speaker_map[speaker_name] = i

    args.speaker_map = speaker_map

    total_data_path = os.path.join(args.data_path, 'total')   
    
    total_outputs = get_data(args, logger, total_data_path, ood_bm, label_map)

    train_label_ids, dev_label_ids, test_label_ids = total_outputs['train_label_ids'], total_outputs['dev_label_ids'], total_outputs['test_label_ids']
    train_speaker_ids, dev_speaker_ids, test_speaker_ids = total_outputs['train_speaker_ids'], total_outputs['dev_speaker_ids'], total_outputs['test_speaker_ids']

    if args.method in ['text']:

        text_data = total_outputs['text_data']
       
        text_train_data = TextDataset(train_label_ids, text_data['train'], speaker_ids = train_speaker_ids, multi_turn = True)
        text_dev_data = TextDataset(dev_label_ids, text_data['dev'], speaker_ids = dev_speaker_ids, multi_turn = True)
        text_test_data = TextDataset(test_label_ids, text_data['test'], speaker_ids = test_speaker_ids, multi_turn = True)

        data = {'train': text_train_data, 'dev': text_dev_data, 'test': text_test_data} 

    else:
        
        text_data = total_outputs['text_data']
        video_data = total_outputs['video_data']
        audio_data = total_outputs['audio_data']

        mm_train_data = MMDataset(train_label_ids, text_data['train'], video_data['train'], audio_data['train'], train_speaker_ids, multi_turn = True)
        mm_dev_data = MMDataset(dev_label_ids, text_data['dev'], video_data['dev'], audio_data['dev'], dev_speaker_ids, multi_turn = True)
        mm_test_data = MMDataset(test_label_ids, text_data['test'], video_data['test'], audio_data['test'], test_speaker_ids, multi_turn = True)

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
    
    train_data_index, train_label_ids, train_speaker_ids = get_indexes_annotations(args, bm, label_map, os.path.join(data_path, 'train.tsv'), args.data_mode)
    dev_data_index, dev_label_ids, dev_speaker_ids = get_indexes_annotations(args, bm, label_map, os.path.join(data_path, 'dev.tsv'), args.data_mode)
    test_data_index, test_label_ids, test_speaker_ids = get_indexes_annotations(args, bm, label_map, os.path.join(data_path, 'test.tsv'), args.data_mode)
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
    
    if args.multiturn:

        outputs.update({
            'train_speaker_ids': train_speaker_ids,
            'dev_speaker_ids': dev_speaker_ids,
            'test_speaker_ids': test_speaker_ids
        })
        outputs = singleturn2multiturn(args, outputs, data_args)

    return outputs
    
def get_indexes_annotations(args, bm, label_map, read_file_path, data_mode):

    with open(read_file_path, 'r') as f:

        data = csv.reader(f, delimiter="\t")
        indexes = []
        label_ids = []
        speaker_ids = []

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
                speaker_ids.append(args.speaker_map[line[7]])
                label_id = label_map[line[3]]
            
            label_ids.append(label_id)
    
    return indexes, label_ids, speaker_ids