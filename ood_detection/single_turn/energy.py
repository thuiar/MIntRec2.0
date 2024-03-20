from scipy.special import logsumexp

    
def func(args, inputs):

    logits = inputs['y_logit']
    scores = logsumexp(logits, axis = -1)
    
    return scores