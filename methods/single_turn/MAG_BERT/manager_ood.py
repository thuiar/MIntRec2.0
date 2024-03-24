import torch
import torch.nn.functional as F
import logging
import numpy as np
from torch import nn
from utils.functions import restore_model, save_model, EarlyStopping
from tqdm import trange, tqdm
from data.single_turn.utils import get_dataloader
from utils.metrics import AverageMeter, Metrics, OOD_Metrics, OID_Metrics
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
from scipy.stats import norm as dist_model
from itertools import cycle
from utils.functions import softmax_cross_entropy_with_softtarget

__all__ = ['MAG_BERT_OOD']

class MAG_BERT_OOD:

    def __init__(self, args, data, model):
             
        self.logger = logging.getLogger(args.logger_name)
        
        self.device, self.model = model.device, model.model
        self.optimizer, self.scheduler = self._set_optimizer(args, self.model)

        mm_data = data.data
        mm_dataloader = get_dataloader(args, mm_data)
        
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            mm_dataloader['train'], mm_dataloader['dev'], mm_dataloader['test']

        if args.ood:
            self.ood_train_dataloader, self.ood_eval_dataloader = mm_dataloader['ood_train'], mm_dataloader['ood_dev']
   
        self.args = args
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
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr = args.lr, correct_bias=False)
        
        num_train_optimization_steps = int(args.num_train_examples / args.train_batch_size) * args.num_train_epochs
        num_warmup_steps= int(args.num_train_examples * args.num_train_epochs * args.warmup_proportion / args.train_batch_size)
        
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)
        
        return optimizer, scheduler

    def _train(self, args): 
        
        early_stopping = EarlyStopping(args)
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            loss_record = AverageMeter()
            
            for step, (batch_id, batch_ood) in enumerate(tqdm(zip(self.train_dataloader, cycle(self.ood_train_dataloader)), desc="Iteration")):
                
                id_label_ids = batch_id['label_ids'].to(self.device)
                ood_label_ids = batch_ood['label_ids'].to(self.device)

                id_text_feats = batch_id['text_feats'].to(self.device)                
                ood_text_feats = batch_ood['text_feats'].to(self.device)
                
                id_video_feats = batch_id['video_feats'].to(self.device)
                ood_video_feats = batch_ood['video_feats'].to(self.device)

                id_audio_feats = batch_id['audio_feats'].to(self.device)
                ood_audio_feats = batch_ood['audio_feats'].to(self.device)

                with torch.set_grad_enabled(True):

                    id_outputs = self.model(id_text_feats, id_video_feats, id_audio_feats)
                    ood_outputs = self.model(ood_text_feats, ood_video_feats, ood_audio_feats)

                    id_loss = self.criterion(id_outputs['mm'], id_label_ids)
                    
                    smax_oe = F.log_softmax(ood_outputs['mm'] - torch.max(ood_outputs['mm'], dim=1, keepdim=True)[0], dim=1)
                    ood_loss = -1 * smax_oe.mean()

                    loss = id_loss + ood_loss

                    self.optimizer.zero_grad()

                    loss.backward()
                    loss_record.update(loss.item(), id_label_ids.size(0))

                    self.optimizer.step()
                    self.scheduler.step()
            
            outputs = self._get_outputs(args, mode = 'eval')
            eval_score = outputs['acc']
 
            eval_results = {
                'train_loss': round(loss_record.avg, 4),
                'eval_score': round(eval_score, 4),
                'best_eval_score': round(early_stopping.best_score, 4),
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
            video_feats = batch['video_feats'].to(self.device)
            audio_feats = batch['audio_feats'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)
            
            with torch.set_grad_enabled(False):
        
                outputs = self.model(text_feats, video_feats, audio_feats)
                logits, features = outputs['mm'], outputs['h'][:, 0]

                total_logits = torch.cat((total_logits, logits))
                total_labels = torch.cat((total_labels, label_ids))
                total_features = torch.cat((total_features, features))
        
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
            outputs = self.metrics(y_true, y_pred, show_results = show_results)
            
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

        lof = LocalOutlierFactor(n_neighbors = 20, contamination = 0.1, novelty = True, n_jobs = -1)
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