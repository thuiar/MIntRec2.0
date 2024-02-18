import torch
import torch.nn.functional as F

def generate_context(args, feats, qmask, umask, lengths, context_len = 1, modality = 'text'):
    
    bs = feats.shape[0]
    dia_len = feats.shape[1]

    tensor_shape = feats.shape
    tensor_shape = list(tensor_shape)

    if modality == 'text':    
        max_seq_len = tensor_shape[-1] = round((context_len + 1) * tensor_shape[-1])
    else:
        max_seq_len = tensor_shape[-2] = round((context_len + 1) * tensor_shape[-2])
        max_feat_dim = tensor_shape[-1]


    tensor_shape = tuple(tensor_shape)
    results = torch.zeros(tensor_shape, dtype=feats.dtype, device=feats.device)

    for i, (fe, qm, le) in enumerate(zip(feats, qmask, lengths)):
        q0 = {}

        for j, (fe_batch, qm_batch, le_batch) in enumerate(zip(fe, qm, le)):
            
            qm_batch = int(qm_batch.cpu().numpy())

            if le_batch == 0:
                results[i][j] = torch.zeros((tensor_shape[-2], tensor_shape[-1]))
                continue

            if qm_batch in q0.keys():

                history_fe = q0[qm_batch]

                if modality == 'text':

                    history_length = torch.sum(history_fe[1])
                                                                                                                                                                                                                                                                                                   
                    history_fe[2][: history_length] = 1
                    q_feat = updated_feat = torch.cat((fe_batch[:, :le_batch], history_fe[:, 1:]), dim = -1)

                else:
                    cur_feature_identifier = torch.zeros((1, max_feat_dim)).to(fe_batch.device)
                    cur_feats = torch.cat([fe_batch[:le_batch, :], cur_feature_identifier], dim = -2)
        
                    q_feat = torch.cat((fe_batch[:le_batch, :], history_fe), dim = -2)
                    updated_feat = torch.cat((cur_feats, history_fe), dim = -2)

            else:
                updated_feat = q_feat = fe_batch[:le_batch]

            q0[qm_batch] = q_feat
            cur_feat = updated_feat

            if modality == 'text':
                cur_len = cur_feat.shape[-1]

                if cur_len >= max_seq_len:
                    cur_feat = cur_feat[:, :max_seq_len]
                    final_feat = cur_feat
                else:
                    pad_length = max_seq_len - cur_len
                    padding = (0, pad_length)
                    final_feat = F.pad(cur_feat, padding, mode='constant', value=0)

            else:
                cur_len = cur_feat.shape[-2]

                if cur_len >= max_seq_len:
                    cur_feat = cur_feat[:max_seq_len, :]
                    final_feat = cur_feat
                else:
                    pad_length = max_seq_len - cur_len
                    padding = (0, 0, 0, pad_length)
                    final_feat = F.pad(cur_feat, padding, mode='constant', value=0)

            results[i][j] = final_feat
          
    return results
    





