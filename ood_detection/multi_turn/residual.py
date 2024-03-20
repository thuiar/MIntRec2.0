import numpy as np
from sklearn.covariance import EmpiricalCovariance
from numpy.linalg import norm, pinv
    
def func(args, inputs):
    
    w = inputs['w']
    w = w.data.cpu().numpy()
    b = inputs['b']
    b = b.data.cpu().numpy()
    
    train_feats = inputs['train_feats']
 
    u = -np.matmul(pinv(w), b)
            
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(train_feats - u)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[args.num_labels:]]).T)
 
    features = inputs['y_feat']
    scores = -norm(np.matmul(features - u, NS), axis=-1)

    return scores