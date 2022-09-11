import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import ce_loss

class Get_Scalar:
    def __init__(self, value):
        self.value = value
        
    def get_value(self, iter):
        return self.value
    
    def __call__(self, iter):
        return self.value

def consistency_loss(logits_x_ulb_w_reverse, logits_x_ulb_s_reverse, logits_w, logits_s, k, name='ce', T=1.0, p_cutoff=0.0, use_hard_labels=True):

    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()
    logits_x_ulb_w_reverse = logits_x_ulb_w_reverse.detach()

    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')
    
    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(p_cutoff).float()
        mask_dis = max_probs.lt(p_cutoff).float()

        pseudo_label_reverse = torch.softmax(logits_x_ulb_w_reverse, dim=-1)

        if k==0 or k==pseudo_label.size(1):
            pass
        else:
            filter_value = float(0)
            indices_to_remove = pseudo_label_reverse < torch.topk(pseudo_label_reverse, k)[0][..., -1, None]
            pseudo_label_reverse[indices_to_remove] = filter_value
            logits_x_ulb_s_reverse[indices_to_remove] = filter_value 

        if use_hard_labels:
            masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
            masked_reverse_loss = ce_loss(logits_x_ulb_s_reverse, pseudo_label_reverse, use_hard_labels = False, reduction='none') * mask_dis 
            # masked_reverse_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask_dis
        else:
            pseudo_label = torch.softmax(logits_w/T, dim=-1)
            masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
        return masked_loss.mean(), masked_reverse_loss.mean(), mask.mean()

    else:
        assert Exception('Not Implemented consistency_loss')

