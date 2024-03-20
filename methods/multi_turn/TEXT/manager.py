import torch
import torch.nn.functional as F
import logging
import numpy as np

from data.multi_turn.utils import get_dataloader
from torch.utils.data import DataLoader
from torch import nn
from tqdm import trange, tqdm
from transformers import BertForSequenceClassification, RobertaForSequenceClassification
from utils.functions import restore_model, save_model, EarlyStopping
from utils.metrics import AverageMeter, Metrics, OOD_Metrics, OID_Metrics
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AdamW, get_linear_schedule_with_warmup
from ood_detection.multi_turn import ood_detection_map
import itertools
from itertools import cycle
import pandas as pd
from scipy.stats import norm as dist_model
from utils.mt import generate_context
from torch import optim
from utils.functions import softmax_cross_entropy_with_softtarget

__all__ = ['TEXT']


class TEXT:

    def __init__(self, args, data):

        self.logger = logging.getLogger(args.logger_name)
        
        if args.text_backbone.startswith('roberta'):
            self.model = RobertaForSequenceClassification.from_pretrained(args.text_pretrained_model, num_labels = args.num_labels)
        elif args.text_backbone.startswith('bert'):
            self.model = BertForSequenceClassification.from_pretrained(args.text_pretrained_model, num_labels = args.num_labels)
            
        self.optimizer, self.scheduler = self._set_optimizer(args, self.model)
  
        args.device = self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
        
        self.model.to(self.device)
        
        text_data = data.data
        text_dataloader = get_dataloader(args, text_data)
        
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            text_dataloader['train'], text_dataloader['dev'], text_dataloader['test']
    
        self.criterion = nn.CrossEntropyLoss()
        self.metrics = Metrics(args)
        self.oid_metrics = OID_Metrics(args)
        self.ood_metrics = OOD_Metrics(args)
        self.ood_detection_func = ood_detection_map[args.ood_detection_method]
        
        if args.train:
            self.best_eval_score = 0
        else:
            self.model = restore_model(self.model, args.model_output_path, self.device)

    def _set_optimizer(self, args, model):

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr = args.lr, correct_bias=False)
        
        print('num_train_examples:', args.num_train_examples)
        num_train_optimization_steps = int(args.num_train_examples / args.train_bs) * args.num_train_epochs
        print('num_train_optimization_steps:', num_train_optimization_steps)

        num_warmup_steps= int(args.num_train_examples * args.num_train_epochs * args.warmup_proportion / args.train_bs)
        print('num_warmup_steps:', num_warmup_steps)
        
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
        return optimizer, scheduler

    def _train(self, args): 
        
        early_stopping = EarlyStopping(args)

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            loss_record = AverageMeter()

            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):

                text_feats = batch['text_feats'].to(self.device)
                label_ids = batch['label_ids'].to(self.device)
                speaker_ids = batch['speaker_ids'].to(self.device)
                u_mask = batch['umask'].to(self.device)

                text_lengths = torch.sum(text_feats[:, :, 1], dim=2, keepdim=True)  

                text_feats = generate_context(args, text_feats, speaker_ids, u_mask, text_lengths)
                f1, f2 = text_feats.shape[-2], text_feats.shape[-1]
                
                input_ids, input_mask, segment_ids = text_feats.view(-1, f1, f2)[:, 0], text_feats.view(-1, f1, f2)[:, 1], text_feats.view(-1, f1, f2)[:, 2]
                label_ids = label_ids.view(-1)
                u_mask = u_mask.view(-1).bool()

                input_ids = input_ids[u_mask]
                input_mask = input_mask[u_mask]
                segment_ids = segment_ids[u_mask]
                label_ids = label_ids[u_mask]

                is_ids = torch.nonzero(label_ids != args.ood_label_id)
                oos_ids = torch.nonzero(label_ids == args.ood_label_id)

                if len(is_ids) > len(oos_ids):
                    main_e = is_ids
                    cycle_e = cycle(oos_ids)
                    
                else:
                    main_e = oos_ids
                    cycle_e = cycle(is_ids)
                
                batch_size = args.train_bs // 2
                main_e_batches = [main_e[i:i+batch_size] for i in range(0, len(main_e), batch_size)]
                # cycle_e_batches = [cycle_e[i:i+batch_size] for i in range(0, len(main_e), batch_size)]
                cycle_e_batches = [list(itertools.islice(cycle_e, batch_size)) for _ in range(batch_size)]
                cycle_e_batches = torch.stack([torch.tensor(batch) for batch in cycle_e_batches])

                for step, (m_e, c_e) in enumerate(zip(main_e_batches, cycle_e_batches)):
                    
                    m_select_input_ids = input_ids[m_e].squeeze(1) if input_ids[m_e].ndim == 3 else input_ids[m_e]
                    m_select_segment_ids = segment_ids[m_e].squeeze(1) if segment_ids[m_e].ndim == 3 else segment_ids[m_e]
                    m_select_attention_mask = input_mask[m_e].squeeze(1) if input_mask[m_e].ndim == 3 else input_mask[m_e]
                    m_select_label_ids = label_ids[m_e].squeeze(1) if label_ids[m_e].ndim == 2 else label_ids[m_e].unsqueeze(0)

                    if len(c_e) != 0:
                        c_select_input_ids = input_ids[c_e].squeeze(1) if input_ids[c_e].ndim == 3 else input_ids[c_e]
                        c_select_segment_ids = segment_ids[c_e].squeeze(1) if segment_ids[c_e].ndim == 3 else segment_ids[c_e]
                        c_select_attention_mask = input_mask[c_e].squeeze(1) if input_mask[c_e].ndim == 3 else input_mask[c_e]
                        c_select_label_ids = label_ids[c_e].squeeze(0) 

                    with torch.set_grad_enabled(True):
                        
                        m_outputs = self.model(input_ids = m_select_input_ids, token_type_ids = m_select_segment_ids, attention_mask = m_select_attention_mask)
                        m_logits = m_outputs.logits

                        if len(c_e) != 0:
                            c_outputs = self.model(input_ids = c_select_input_ids, token_type_ids = c_select_segment_ids, attention_mask = c_select_attention_mask)
                            c_logits = c_outputs.logits
                        
                        if m_select_label_ids[0] != args.ood_label_id:
                            
                            id_loss = self.criterion(m_logits, m_select_label_ids)

                            if len(c_e) != 0:
                                # smax_oe = F.log_softmax(c_logits - torch.max(c_logits, dim=1, keepdim=True)[0], dim=1)
                                # ood_loss = -1 * smax_oe.mean()
                                ood_loss = softmax_cross_entropy_with_softtarget(c_logits, args.num_labels, self.device)
                        
                        else:
                            id_loss = self.criterion(c_logits, c_select_label_ids)

                            if len(c_e) != 0:
                                # smax_oe = F.log_softmax(m_logits - torch.max(m_logits, dim=1, keepdim=True)[0], dim=1)
                                # ood_loss = -1 * smax_oe.mean()
                                ood_loss = softmax_cross_entropy_with_softtarget(m_logits, args.num_labels, self.device)

                        if len(c_e) != 0:
                            loss = id_loss + ood_loss
                        else:
                            loss = id_loss

                        self.optimizer.zero_grad()

                        loss.backward()
                        loss_record.update(loss.item(), m_select_label_ids.size(0))
                        
                        # if args.grad_clip != -1.0:
                        #     nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], args.grad_clip)

                        self.optimizer.step()
                        self.scheduler.step()    
            
            train_outputs = self._get_outputs(args, mode = 'train')      
            outputs = self._get_outputs(args, mode = 'eval')

            y_logit = outputs['y_logit']
            y_true = outputs['y_true']
            y_pred = outputs['y_pred']
 
            mu_stds = self.cal_mu_std(train_outputs['y_logit'], train_outputs['y_true'], args.num_labels)
            y_pred = self.classify_doc(args, y_logit, mu_stds)
            eval_score = self.oid_metrics(y_true, y_pred)['oid_f1']
 
            if args.use_scheduler:
                self.scheduler.step(eval_score)

            eval_results = {
                'train_loss': round(loss_record.avg, 4),
                'best_eval_score': round(early_stopping.best_score, 4),
                'eval_score': round(eval_score, 4)
            }

            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in eval_results.keys():
                self.logger.info("  %s = %s", key, str(eval_results[key]))
            
            early_stopping(eval_score, self.model)

            if early_stopping.early_stop:
                self.logger.info(f'EarlyStopping at epoch {epoch + 1}')
                break

        self.best_eval_score = early_stopping.best_score
        self.model = early_stopping.best_model

        if args.save_model:
            self.logger.info('Trained models are saved in %s', args.model_output_path)
            save_model(self.model, args.model_output_path)     
    
    def batch_iteration(self, args, dataloader):

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)
        total_features = torch.empty((0, args.text_feat_dim)).to(self.device)
       
        for batch in tqdm(dataloader, desc="Iteration"):

            text_feats = batch['text_feats'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)
            speaker_ids = batch['speaker_ids'].to(self.device)
            u_mask = batch['umask'].to(self.device)

            text_lengths = torch.sum(text_feats[:, :, 1], dim = 2, keepdim = True)  
            
            text_feats = generate_context(args, text_feats, speaker_ids, u_mask, text_lengths)
            f1, f2 = text_feats.shape[-2], text_feats.shape[-1]

            input_ids, input_mask, segment_ids = text_feats.view(-1, f1, f2)[:, 0], text_feats.view(-1, f1, f2)[:, 1], text_feats.view(-1, f1, f2)[:, 2]
            label_ids = label_ids.view(-1)
            u_mask = u_mask.view(-1).bool()

            input_ids = input_ids[u_mask]
            input_mask = input_mask[u_mask]
            segment_ids = segment_ids[u_mask]
            label_ids = label_ids[u_mask]

            select_bs = args.train_bs
            st = 0
            flag = False

            while True:
                
                ed = st + select_bs
                if ed >= u_mask.shape[0]:
                    flag = True
                    ed = u_mask.shape[0]

                select_input_ids = input_ids[st : ed]
                select_segment_ids = segment_ids[st : ed]
                select_attention_mask = input_mask[st : ed]
                
                select_label_ids = label_ids[st : ed]

                flag_id = torch.any(select_label_ids != args.ood_label_id).item()
                flag_ood = torch.any(select_label_ids == args.ood_label_id).item()

                if flag_id or flag_ood:

                    with torch.set_grad_enabled(False):
                        
                        outputs = self.model(input_ids = select_input_ids, token_type_ids = select_segment_ids, attention_mask = select_attention_mask, output_hidden_states = True)
                        logits = outputs.logits

                        features = outputs.hidden_states[-1][:, 0]

                        total_logits = torch.cat((total_logits, logits))
                        total_labels = torch.cat((total_labels, select_label_ids))
                        total_features = torch.cat((total_features, features))

                st += select_bs
                if flag:
                    break  

        return total_logits, total_labels, total_features

    def _get_outputs(self, args, mode = 'eval', show_results = False, test_ind = False):

        self.model.eval()
        
        if mode == 'eval':
            total_logits, total_labels, total_features = self.batch_iteration(args, self.eval_dataloader)

        elif mode == 'train':
            total_logits, total_labels, total_features = self.batch_iteration(args, self.train_dataloader)

        elif mode == 'test':
            total_logits, total_labels, total_features = self.batch_iteration(args, self.test_dataloader)
        
        total_probs = F.softmax(total_logits.detach(), dim=1)
        total_maxprobs, total_preds = total_probs.max(dim = 1)
        y_logit = torch.sigmoid(total_logits.detach()).cpu().numpy()

        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        y_prob = total_maxprobs.cpu().numpy()
        y_feat = total_features.cpu().numpy()
        
        if test_ind:
            outputs = self.metrics(y_true[y_true != args.ood_label_id], y_pred[y_true != args.ood_label_id])
        else:

            outputs = self.oid_metrics(y_true, y_pred, show_results = show_results)
        
        outputs.update(
            {
                'y_prob': y_prob,
                'y_logit': y_logit,
                'y_true': y_true,
                'y_pred': y_pred,
                'y_feat': y_feat
            }
        )

        return outputs
    
    def classify_doc(self, args, y_prob, mu_stds):

        thresholds = {}
        for col in range(args.num_labels):
            threshold = max(0.5, 1 - args.scale * mu_stds[col][1])
            thresholds[col] = threshold
        thresholds = np.array(thresholds)
        self.logger.info('Probability thresholds of each class: %s', thresholds)
        
        y_pred = []
        for p in y_prob:
            max_class = np.argmax(p)
            max_value = np.max(p)
            threshold = max(0.5, 1 - args.scale * mu_stds[max_class][1])

            if max_value > threshold:
                y_pred.append(max_class)
            else:
                y_pred.append(args.ood_label_id)

        return np.array(y_pred)
    
    def fit(self, prob_pos_X):
        prob_pos = [p for p in prob_pos_X] + [2 - p for p in prob_pos_X]
        pos_mu, pos_std = dist_model.fit(prob_pos)
        return pos_mu, pos_std

    def cal_mu_std(self, y_prob, trues, num_labels):

        mu_stds = []
        for i in range(num_labels):
            pos_mu, pos_std = self.fit(y_prob[trues == i, i])
            mu_stds.append([pos_mu, pos_std])

        return mu_stds

    def classify_lof(self, args, preds, train_feats, pred_feats):

        lof = LocalOutlierFactor(n_neighbors = 20, contamination = 0.05, novelty = True, n_jobs = -1)
        lof.fit(train_feats)
        y_pred_lof = pd.Series(lof.predict(pred_feats))
        preds[y_pred_lof[y_pred_lof == -1].index] = args.ood_label_id

        return preds

    def _test(self, args):
        
        test_results = {}
        
        ind_test_results = self._get_outputs(args, mode = 'test', show_results = True, test_ind = True)
        if args.train:
            test_results['best_eval_score'] = round(self.best_eval_score, 4)
        test_results.update(ind_test_results)
 
        if args.ood:
            
            ind_train_outputs = self._get_outputs(args, mode = 'train')

            train_y_logit = ind_train_outputs['y_logit']
            train_y_true = ind_train_outputs['y_true']

            y_pred = ind_test_results['y_pred']
            y_feat = ind_test_results['y_feat']
            y_true = ind_test_results['y_true']
            y_logit = ind_test_results['y_logit']     
  
            mu_stds = self.cal_mu_std(train_y_logit, train_y_true, args.num_labels)
            y_pred = self.classify_doc(args, y_logit, mu_stds)

            oid_test_results = self.oid_metrics(y_true, y_pred, show_results = True)
            test_results.update(oid_test_results)
        
        return test_results