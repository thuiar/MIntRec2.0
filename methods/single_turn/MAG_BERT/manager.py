import torch
import torch.nn.functional as F
import logging
import numpy as np
from torch import nn
from utils.functions import restore_model, save_model, EarlyStopping
from tqdm import trange, tqdm
from data.single_turn.utils import get_dataloader
from utils.metrics import AverageMeter, Metrics, OID_Metrics
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
from scipy.stats import norm as dist_model

__all__ = ['MAG_BERT']

class MAG_BERT:

    def __init__(self, args, data, model):
             
        self.logger = logging.getLogger(args.logger_name)
        
        self.device, self.model = model.device, model.model
        self.optimizer, self.scheduler = self._set_optimizer(args, self.model)

        mm_data = data.data
        mm_dataloader = get_dataloader(args, mm_data)
        
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            mm_dataloader['train'], mm_dataloader['dev'], mm_dataloader['test']
            
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.metrics = Metrics(args)
        self.oid_metrics = OID_Metrics(args)
        
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
            
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):

                text_feats = batch['text_feats'].to(self.device)
                video_feats = batch['video_feats'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)
                label_ids = batch['label_ids'].to(self.device)

                with torch.set_grad_enabled(True):

                    outputs = self.model(text_feats, video_feats, audio_feats)

                    loss = self.criterion(outputs['mm'], label_ids)

                    self.optimizer.zero_grad()

                    loss.backward()
                    loss_record.update(loss.item(), label_ids.size(0))

                    self.optimizer.step()
                    self.scheduler.step()
            
            outputs = self._get_outputs(args, self.eval_dataloader)
            eval_score = outputs[args.eval_monitor]

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

    def _get_outputs(self, args, dataloader, show_results = False, test_ind = False):

        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)
        total_features = torch.empty((0, args.feat_size)).to(self.device)
    
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
        
        ind_test_results = self._get_outputs(args, self.test_dataloader, show_results = True, test_ind = True)
        if args.train:
            test_results['best_eval_score'] = round(self.best_eval_score, 4)
        test_results.update(ind_test_results)
        
        if args.ood:
            
            tmp_outputs = self._get_outputs(args, self.test_dataloader)
            ind_train_outputs = self._get_outputs(args, self.train_dataloader)

            train_y_logit = ind_train_outputs['y_logit']
            train_y_true = ind_train_outputs['y_true']

            y_pred = tmp_outputs['y_pred']
            y_feat = tmp_outputs['y_feat']
            y_true = tmp_outputs['y_true']
            y_logit = tmp_outputs['y_logit']

            mu_stds = self.cal_mu_std(train_y_logit, train_y_true, args.num_labels)
            y_pred = self.classify_doc(args, y_logit, mu_stds)
  
            oid_test_results = self.oid_metrics(y_true, y_pred, show_results = True)
            test_results.update(oid_test_results)     
        
        return test_results
