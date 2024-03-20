
def func(args, inputs):

    logits = inputs['y_logit']
    scores = logits.max(1)
    
    return scores