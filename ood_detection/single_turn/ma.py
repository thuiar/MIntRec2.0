import numpy as np
from sklearn.covariance import EmpiricalCovariance

def cal_ma_dis(mean, prec, features):
    
    ma_score = -np.array([(((f - mean)@prec) * (f - mean)).sum(-1).min() for f in features])
    
    return ma_score
            
def func(args, inputs):
    
    train_feats = inputs['train_feats']
    train_labels = inputs['train_labels']
    train_means = []
    train_dis = []
    
    for l in range(args.num_labels):
        fs = train_feats[train_labels == l]
        m = fs.mean(axis = 0)
        train_means.append(m)
        train_dis.extend(fs - m)
    
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(np.array(train_dis).astype(np.float64))

    mean = np.array(train_means)
    prec = ec.precision_  
    
    features = inputs['y_feat']
    scores = cal_ma_dis(mean, prec, features)
    
    return scores