from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, \
    precision_score, recall_score, roc_auc_score, average_precision_score, \
        auc, precision_recall_curve, roc_curve
        
from scipy.optimize import brentq
from scipy.interpolate import interp1d
        
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

class OOD_Metrics(object):
    """
    column of confusion matrix: predicted index
    row of confusion matrix: target index
    """
    def __init__(self, args):

        self.logger = logging.getLogger(args.logger_name)
        self.eval_metrics = ['auroc', 'aupr_in', 'aupr_out', 'fpr95', 'eer', 'der']

    def __call__(self, scores, y_true, show_results = False):

        fpr95 = self._fpr_recall(scores, y_true, 0.95)
        auroc, aupr_in, aupr_out = self._auc(scores, y_true)
        eer = self._calculate_eer(scores, y_true)
        
        der = self._detection_error(scores, y_true)
        
        eval_results = {
            'auroc': auroc,
            'aupr_in': aupr_in,
            'aupr_out': aupr_out,
            'fpr95': fpr95,
            'eer': eer,
            'der': der
        }
        
        if show_results:

            self.logger.info("***** Out-of-domain Evaluation results *****")
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(round(eval_results[key], 4)))
                
        return eval_results


    def _fpr_recall(self, conf, label, tpr):

        sorted_indices = np.argsort(conf)[::-1]
        y_true_sorted = label[sorted_indices]
        n_samples = len(y_true_sorted)
        n_positive_samples = np.sum(y_true_sorted)
        fpr = 0.0
        n_false_positive = 0
        n_true_positive = 0

        for i in range(n_samples):
            if y_true_sorted[i] == 0:
                n_false_positive += 1
            else:
                n_true_positive += 1

            if n_true_positive / n_positive_samples >= tpr:
                break

        fpr = n_false_positive / (n_samples - n_positive_samples)
        return fpr

  
    def _auc(self, conf, label):

        ind_indicator = np.zeros_like(label)
        ind_indicator[label != 0] = 1

        fpr, tpr, thresholds = roc_curve(ind_indicator, conf)

        precision_in, recall_in, thresholds_in \
            = precision_recall_curve(ind_indicator, conf)

        precision_out, recall_out, thresholds_out \
            = precision_recall_curve(1 - ind_indicator, 1 - conf)

        auroc = auc(fpr, tpr)
        aupr_in = auc(recall_in, precision_in)
        aupr_out = auc(recall_out, precision_out)

        return auroc, aupr_in, aupr_out
    
    def _calculate_eer(self, y_score, y_true):
        fpr, tpr, thresholds = roc_curve(y_true, y_score)

        eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
        return eer
    
    def _detection_error(self, preds, labels, pos_label=1):
        """Return the misclassification probability when TPR is 95%.
            
        preds: array, shape = [n_samples]
            Target normality scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
            i.e.: an high value means sample predicted "normal", belonging to the positive class
            
        labels: array, shape = [n_samples]
                True binary labels in range {0, 1} or {-1, 1}.
        pos_label: label of the positive class (1 by default)
        """
        fpr, tpr, _ = roc_curve(labels, preds, pos_label=pos_label)

        # Get ratios of positives to negatives
        pos_ratio = sum(np.array(labels) == pos_label) / len(labels)
        neg_ratio = 1 - pos_ratio

        # Get indexes of all TPR >= 95%
        idxs = [i for i, x in enumerate(tpr) if x >= 0.95]

        # Calc error for a given threshold (i.e. idx)
        # Calc is the (# of negatives * FNR) + (# of positives * FPR)
        _detection_error = lambda idx: neg_ratio * (1 - tpr[idx]) + pos_ratio * fpr[idx]

        # Return the minimum detection error such that TPR >= 0.95
        return min(map(_detection_error, idxs))

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