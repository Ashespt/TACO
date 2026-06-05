# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import random
import numpy as np
from PIL import Image
from monai.networks.nets.swin_unetr import *
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.utils import ensure_tuple_rep
import argparse
import torch.nn.functional as F
from utils.superglue import log_optimal_transport
from losses.coderating import MaximalCodingRateReduction
from losses.loss import GeCoLoss
from einops import rearrange
from models.swin import Swin
from models.modules import CrossBlock,SelfBlock,TransformerLayer,LearnableFourierPositionalEncoding, normalize_keypoints
from models.decoder import MAEDecoder3D
from models.decoder_head import DecoderHead
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from utils.util import sinkhorn_distance
from losses.superglue import log_optimal_transport
import torch.nn.functional as F
import math

def cos_dist(x, y, eps=1e-8):
    # 计算余弦距离（1 - 余弦相似度）
    x_norm = F.normalize(x, dim=-1)
    y_norm = F.normalize(y, dim=-1)
    cos_sim = (x_norm * y_norm).sum(dim=-1)
    cos_sim = cos_sim.clamp(-1 + eps, 1 - eps)  # 避免数值不稳定
    cos_dist = 1 - cos_sim
    return cos_dist

def neighborhood_ranking_crossbrain(z1, z2, match_idx, k=5, margin=0.05, c=1.0):
    """
    跨脑的局部拓扑保持（邻域排序一致性，向量化快速实现）
    """
    N = z1.size(0)
    D1 = torch.cdist(z1, z1)                 # 这里用欧氏近邻定义；也可换 hyp_dist
    neigh = D1.argsort(dim=-1)[:, 1:k+1]     # (N,k)

    i = torch.arange(N, device=z1.device).unsqueeze(1).expand_as(neigh)
    j = neigh
    i2 = match_idx[i]                        # (N,k)
    j2 = match_idx[j]

    # ------------------------------
    # 构造“非近邻”负样本
    # ------------------------------
    neg = torch.empty_like(neigh)
    for n in range(N):
        # 当前样本的禁止集合（自己 + 邻居）
        exclude = set(neigh[n].tolist() + [n])
        # 可选负样本集合
        candidates = list(set(range(N)) - exclude)
        # 随机采样 k 个负样本
        neg[n] = torch.tensor(
            random.sample(candidates, k),
            device=z1.device,
            dtype=torch.long
        )

    n2 = match_idx[neg]

    # ------------------------------
    # 对比损失计算
    # ------------------------------
    d_pos = cos_dist(z2[i2], z2[j2])  # 正样本相似度
    d_neg = cos_dist(z2[i2], z2[n2])  # 负样本相似度
    loss = F.relu(d_pos - d_neg + margin).mean()
    return loss


def pairwise_dist(x, y=None, metric="cosine"):
    """
    x: (T, C), y: (T, C) or None -> (T, T)
    metric: 'cosine' or 'euclidean'
    """
    if y is None: y = x
    if metric == "euclidean":
        return torch.cdist(x, y)  # (T, T)
    elif metric == "cosine":
        x_n = F.normalize(x, dim=-1)
        y_n = F.normalize(y, dim=-1)
        sim = x_n @ y_n.T
        return 1.0 - sim  
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
def partial_matches_mnn(A, B, metric="cosine"):
    D = pairwise_dist(A, B, metric=metric)      # (T, T)
    # p->q 最近邻
    b_of_a = D.argmin(dim=1)                    # (T,)
    # q->p 最近邻
    a_of_b = D.argmin(dim=0)                    # (T,)

    # 互为NN（Mutual NN）
    ar = torch.arange(A.size(0), device=A.device)
    is_mutual = ar == a_of_b[b_of_a]            # (T,)
    idxA_m = ar[is_mutual]
    idxB_m = b_of_a[is_mutual]
    return idxA_m, idxB_m


def nrc_on_partial(A, B, idxA_m, idxB_m, k=5, margin=0.1, metric="cosine"):
    if idxA_m.numel() == 0:
        return A.new_tensor(0.)

    DA = pairwise_dist(A, metric=metric)                    # (T, T)
    neigh = DA.argsort(dim=-1)[:, 1:k+1]                    # (T, k)

    DB = pairwise_dist(B, metric=metric)                    # (T, T)
    
    is_matched_B = torch.zeros(B.size(0), dtype=torch.bool, device=B.device)
    is_matched_B[idxB_m] = True

    idxB_of_A = torch.full((A.size(0),), -1, device=A.device, dtype=torch.long)
    idxB_of_A[idxA_m] = idxB_m

    loss_sum = A.new_tensor(0.)
    count = 0

    for iA in idxA_m.tolist():
        uB = idxB_of_A[iA].item()  
        if uB < 0:    
            continue

        neigh_i = neigh[iA]                    # (k,)
        neigh_i_matched_mask = idxB_of_A[neigh_i] >= 0
        if not neigh_i_matched_mask.any():
            continue
        neigh_i_posA = neigh_i[neigh_i_matched_mask]        
        posB_idx = idxB_of_A[neigh_i_posA]                 

        neg_pool = (~is_matched_B).nonzero(as_tuple=False).flatten() 
        if neg_pool.numel() == 0:
            all_pool = torch.arange(B.size(0), device=B.device)
            exclude = torch.cat([posB_idx, torch.tensor([uB], device=B.device)])
            mask = torch.ones(B.size(0), dtype=torch.bool, device=B.device)
            mask[exclude] = False
            neg_pool = all_pool[mask]
            if neg_pool.numel() == 0:
                continue

        M_pos = posB_idx.numel()
        if neg_pool.numel() >= M_pos:
            rand_idx = torch.randint(0, neg_pool.numel(), (M_pos,), device=B.device)
            negB_idx = neg_pool[rand_idx]
        else:
            rand_idx = torch.randint(0, neg_pool.numel(), (M_pos,), device=B.device)
            negB_idx = neg_pool[rand_idx]

        d_pos = DB[uB, posB_idx]   # (M_pos,)
        d_neg = DB[uB, negB_idx]   # (M_pos,)
        loss = F.relu(d_pos - d_neg + margin).mean()

        loss_sum = loss_sum + loss
        count += 1

    if count == 0:
        return A.new_tensor(0.)
    return loss_sum / count

def partial_nrc_loss(z1, z2, k=5, margin=0.1, metric="cosine", max_pairs=None):
    
    B, T, C = z1.shape
    device = z1.device
    pairs = []

    for p in range(B):
        for q in range(B):
            if p == q: 
                continue
            pairs.append((p, q))

    if (max_pairs is not None) and (len(pairs) > max_pairs):
        idx = torch.randperm(len(pairs), device=device)[:max_pairs].tolist()
        pairs = [pairs[i] for i in idx]

    total_loss = z1.new_tensor(0.)
    num_terms = 0

    for (p, q) in pairs:
        A = z1[p]  # (T, C)
        B_ = z2[q] # (T, C)

        idxA_m, idxB_m = partial_matches_mnn(A, B_, metric=metric)
        loss_pq = nrc_on_partial(A, B_, idxA_m, idxB_m, k=k, margin=margin, metric=metric)

        total_loss = total_loss + loss_pq
        num_terms += 1

    if num_terms == 0:
        return z1.new_tensor(0.)
    return total_loss / num_terms



class Taco(nn.Module):
    def __init__(self, args,writer=None,exp=200,dim=768,num_patch_side=4,spatial_dims=3,loss_function=GeCoLoss(),norm_pix_loss=True):
        super(Taco, self).__init__()
        self.vit = Swin(args)
        
        self.args = args
        
        self.iter = 0
        self.writer = writer
        self.norm_pix_loss = norm_pix_loss
        self.decoder = DecoderHead(args.in_channels,upsample="vae", dim=768)
        self.k = 5
        self.l = 1
        self.margin = 0.4


    def patchify_3d(self, imgs):
        """
        imgs: (B, 1, H, W, D)
        x: (B, L, patch_size**3 * 1)
        """
        p = 32  # Patch size 16x16x16
        assert imgs.shape[2] == imgs.shape[3] == imgs.shape[4] and imgs.shape[2] % p == 0

        h = w = d = imgs.shape[2] // p  # Number of patches along each dimension
        # Reshape to create patches
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p, d, p))  # (B, 1, h, p, w, p, d, p)
        # Rearrange the axes to create a patch-like structure
        # x = torch.einsum('bnhpwqdc->bnhwdpq',x)  # Permute dimensions
        # Flatten patches into (B, L, patch_size**3)
        x = x.reshape(shape=(imgs.shape[0], h * w * d, p**3))  # (B, L, patch_size^3)

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        
        target = self.patchify_3d(imgs)
        pred = self.patchify_3d(pred)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1).mean()  # [N, L], mean loss per patch
        # loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    
    # def forward(self, img:dict,visual_flag:bool=False,step:int=0):
    def forward(self, x_in,visual_flag:bool=False,step:int=0):
        src_paths = x_in.get('src_path') if isinstance(x_in, dict) else None
        target_paths = x_in.get('target_path') if isinstance(x_in, dict) else None
        x = torch.cat([x_in['src'],x_in['target']])
        hidden_states_out,last_out = self.vit(x) 
        B = hidden_states_out[0].shape[0]
        # single modality
        pred,mask = self.decoder(last_out=last_out)
        loss_rec = self.forward_loss(x,pred,mask)
        # intra instance
        loss2 = 0
        loss3 = 0
        valid_pairs = 0
        for hidden in hidden_states_out[-self.l:]:
            C = hidden.shape[1]
            z_hyp1 = hidden[:B//2,...].reshape(B//2,-1,C)
            z_hyp2 = hidden[B//2:,...].reshape(B//2,-1,C)
            # loss1 = simclr_hyp(z_hyp1,z_hyp2)
            for i in range(B//2):
                same_image = False
                if src_paths is not None and target_paths is not None:
                    same_image = src_paths[i] == target_paths[i]
                if same_image:
                    continue
                match_idx = torch.arange(z_hyp1.shape[1], device=z_hyp1.device)
                loss2 += neighborhood_ranking_crossbrain(z_hyp1[i],z_hyp2[i],match_idx,k=self.k)
                loss2 += neighborhood_ranking_crossbrain(z_hyp2[i],z_hyp1[i],match_idx,k=self.k)
                valid_pairs += 1
            loss3 += partial_nrc_loss(z_hyp1, z_hyp2, k=self.k, margin=self.margin, metric="cosine", max_pairs=6)
            loss3 += partial_nrc_loss(z_hyp2, z_hyp1, k=self.k, margin=self.margin, metric="cosine", max_pairs=6)

        if valid_pairs > 0:
            loss2 = loss2 / valid_pairs
        else:
            loss2 = loss_rec.new_zeros(())
        loss3 = loss3/self.l
        return 10*loss2+loss_rec+10*loss3, 10*loss3, loss_rec
        # return 10*loss2+loss_rec, 10*loss2, loss_rec

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
