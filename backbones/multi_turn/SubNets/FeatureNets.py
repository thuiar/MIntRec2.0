
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, RobertaModel
from torch.nn.utils.rnn import pack_padded_sequence

__all__ = [ 'BERTEncoder', 'ROBERTAEncoder']

def freeze_backbone_parameters(model):
    for name, param in model.named_parameters():  
        param.requires_grad = False
        if "encoder.layer.11" in name or "pooler" in name:
            param.requires_grad = True
    return model

class BERTEncoder(nn.Module):

    def __init__(self, args):

        super(BERTEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(args.text_pretrained_model)
        if args.freeze_backbone_parameters:
            self.bert = freeze_backbone_parameters(self.bert)
    
    def forward(self, text_feats = None, embeds = None, sent_mask = None, mixup = False):
        
        if mixup:
            outputs = self.bert(inputs_embeds=embeds, attention_mask = sent_mask)
        else:
            outputs = self.bert(text_feats[:, 0], text_feats[:, 1], text_feats[:, 2])

        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states

class RoBERTaEncoder(nn.Module):
    
    def __init__(self, args):

        super(RoBERTaEncoder, self).__init__()
        self.roberta = RobertaModel.from_pretrained(args.text_pretrained_model)
        if args.freeze_backbone_parameters:
            self.roberta = freeze_backbone_parameters(self.roberta)

    def forward(self, text_feats):
        outputs = self.roberta(input_ids = text_feats[:, 0], attention_mask = text_feats[:, 1])
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states

#################
class SubNet(nn.Module):
    '''
    The subnetwork that is used in TFN for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, dropout):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(SubNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))

        return y_3

    
class AuViSubNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(AuViSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
   
        packed_sequence = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, final_states = self.rnn(packed_sequence)
      
        h = self.dropout(final_states[0].squeeze(0))
        y_1 = self.linear_1(h)
        return y_1