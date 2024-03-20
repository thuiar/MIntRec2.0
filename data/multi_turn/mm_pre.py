from torch.utils.data import Dataset
import torch
import numpy as np

__all__ = ['MMDataset']

class MMDataset(Dataset):
        
    def __init__(self, label_ids, text_data, video_data, audio_data, speaker_ids = None, multi_turn = False):
        
        self.label_ids = label_ids
        self.text_data = text_data
        self.video_data = video_data
        self.audio_data = audio_data
        self.size = len(self.text_data)
        self.speaker_ids = speaker_ids
        self.multi_turn = multi_turn

    def __len__(self):
        return self.size

    def __getitem__(self, index):

        sample = {
            'label_ids': torch.tensor(self.label_ids[index]), 
            'text_feats': torch.tensor(self.text_data[index]),
            'video_feats': torch.tensor(np.array(self.video_data['feats'][index])),
            'video_lengths': torch.tensor(np.array(self.video_data['lengths'][index])),
            'audio_feats': torch.tensor(np.array(self.audio_data['feats'][index])),
            'audio_lengths': torch.tensor(np.array(self.audio_data['lengths'][index]))
        } 

        if self.multi_turn:
            sample.update({
                'speaker_ids': torch.tensor(np.array(self.speaker_ids[index])),
                'umask': torch.tensor(np.array([1] * len(self.label_ids[index])))
            })

        return sample