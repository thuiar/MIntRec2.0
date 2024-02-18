from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, \
    precision_score, recall_score
        
import logging
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum) / self.count

class Metrics(object):
    """
    column of confusion matrix: predicted index
    row of confusion matrix: target index
    """
    def __init__(self, args):

        self.logger = logging.getLogger(args.logger_name)
        self.eval_metrics = ['acc', 'weighted_f1', 'weighted_prec',  'f1',  'prec', 'rec']

    def __call__(self, y_true, y_pred, show_results = False):

        acc_score = self._acc_score(y_true, y_pred)
        macro_f1, weighted_f1 = self._f1_score(y_true, y_pred)
        macro_prec, weighted_prec = self._precision_score(y_true, y_pred)
        macro_rec, weighted_rec = self._recall_score(y_true, y_pred)
            
        eval_results = {
            'acc': acc_score,
            'f1': macro_f1,
            'weighted_f1': weighted_f1,
            'prec': macro_prec,
            'weighted_prec': weighted_prec,
            'rec': macro_rec,
            'weighted_rec': weighted_rec
        }
        
        if show_results:
            
            self._show_confusion_matrix(y_true, y_pred)

            self.logger.info("***** In-domain Evaluation results *****")
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(round(eval_results[key], 4)))

        return eval_results

    def _acc_score(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)
    
    def _f1_score(self, y_true, y_pred):
        return f1_score(y_true, y_pred, average='macro'), f1_score(y_true, y_pred, average='weighted')
    
    def _precision_score(self, y_true, y_pred):
        return precision_score(y_true, y_pred, average='macro'), precision_score(y_true, y_pred, average='weighted')

    def _recall_score(self, y_true, y_pred):
        return recall_score(y_true, y_pred, average='macro'), recall_score(y_true, y_pred, average='weighted')

    def _show_confusion_matrix(self, y_true, y_pred):

        cm = confusion_matrix(y_true, y_pred)
        self.logger.info("***** Test: Confusion Matrix *****")
        self.logger.info("%s", str(cm))

class OID_Metrics(object):
    """
    column of confusion matrix: predicted index
    row of confusion matrix: target index
    """
    def __init__(self, args):

        self.logger = logging.getLogger(args.logger_name)
        self.eval_metrics = ['oid_acc', 'f1-known',  'f1-open', 'oid_f1']

    def __call__(self, y_true, y_pred, show_results = False):

        acc_score = self._acc_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        F_measure = self._F_measure(cm)

            
        eval_results = {
            'oid_acc': acc_score,
            'f1-known': F_measure['f1-known'],
            'f1-open': F_measure['f1-open'],
            'oid_f1': F_measure['f1-all'],
        }
        
        if show_results:
            
            self._show_confusion_matrix(cm)

            self.logger.info("***** In-domain Evaluation results *****")
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(round(eval_results[key], 4)))

        return eval_results

    def _acc_score(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)
    
    def _show_confusion_matrix(self, cm):
        self.logger.info("***** Test: Confusion Matrix *****")
        self.logger.info("%s", str(cm))
    
    def _F_measure(self, cm):
        idx = 0
        rs, ps, fs = [], [], []
        n_class = cm.shape[0]
        
        for idx in range(n_class):
            TP = cm[idx][idx]
            r = TP / cm[idx].sum() if cm[idx].sum() != 0 else 0
            p = TP / cm[:, idx].sum() if cm[:, idx].sum() != 0 else 0
            f = 2 * r * p / (r + p) if (r + p) != 0 else 0
            rs.append(r)
            ps.append(p)
            fs.append(f)
            
        f = np.mean(fs).round(4)
        f_seen = np.mean(fs[:-1]).round(4)
        f_unseen = round(fs[-1], 4)
        
        results = {}
        results['f1-known'] = f_seen
        results['f1-open'] = f_unseen
        results['f1-all'] = f
        
        return results
