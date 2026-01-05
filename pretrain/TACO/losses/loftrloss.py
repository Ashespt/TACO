import torch
import torch.nn as nn


class LoFTRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_pos_w = 1
        self.c_neg_w = 1
    def compute_coarse_loss(self, conf, conf_gt, weight=None):
        """ Point-wise CE / Focal Loss with 0 / 1 confidence as gt.
        Args:
            conf (torch.Tensor): (N, HW0, HW1) / (N, HW0+1, HW1+1)
            conf_gt (torch.Tensor): (N, HW0, HW1)
            weight (torch.Tensor): (N, HW0, HW1)
        """
        pos_mask, neg_mask = conf_gt == 1, conf_gt == 0
        c_pos_w, c_neg_w = self.c_pos_w, self.c_neg_w
        # corner case: no gt coarse-level match at all
        if not pos_mask.any():  # assign a wrong gt
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_neg_w = 0.

        
        conf = torch.clamp(conf, 1e-6, 1-1e-6)
        loss_pos = - torch.log(conf[pos_mask])
        loss_neg = - torch.log(1 - conf[neg_mask])
        if weight is not None:
            loss_pos = loss_pos * weight[pos_mask]
            loss_neg = loss_neg * weight[neg_mask]
        pos_loss = c_pos_w * loss_pos.mean() 
        neg_loss = c_neg_w * loss_neg.mean()
        return pos_loss+neg_loss,pos_loss,neg_loss
    
    # @torch.no_grad()
    # def compute_c_weight(self, data):
    #     """ compute element-wise weights for computing coarse-level loss. """
    #     if 'mask0' in data:
    #         c_weight = (data['mask0'].flatten(-2)[..., None] * data['mask1'].flatten(-2)[:, None]).float()
    #     else:
    #         c_weight = None
    #     return c_weight

    def forward(self, predict,gt):
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        # 0. compute element-wise loss weight
        # c_weight = self.compute_c_weight(data)

        # 1. coarse-level loss
        loss_c = self.compute_coarse_loss(
                predict,gt)
        return loss_c