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
from utils.visualization import pca_1d_visual
import math
from ripser import ripser
from persim import wasserstein, bottleneck
from torchph.pershom import vr_persistence_l1, vr_persistence
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from utils.util import sinkhorn_distance
from losses.superglue import log_optimal_transport
from geomloss import SamplesLoss
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
    返回的是“距离”（越小越近）
    """
    if y is None: y = x
    if metric == "euclidean":
        return torch.cdist(x, y)  # (T, T)
    elif metric == "cosine":
        x_n = F.normalize(x, dim=-1)
        y_n = F.normalize(y, dim=-1)
        sim = x_n @ y_n.T
        return 1.0 - sim  # 1 - cos = 距离
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
def partial_matches_mnn(A, B, metric="cosine"):
    """
    A: (T, C), B: (T, C)
    返回：
      idxA_m: (M,)  A中被匹配的token索引
      idxB_m: (M,)  B中对应的token索引（单射近似，可能有重复会被MNN过滤）
    """
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
    """
    A, B: (T, C)   —— 两个实例的token特征
    idxA_m, idxB_m: (M,)  —— 部分匹配子集（A[i] ↔ B[j]）
    在 A 的局部邻域里，对匹配到的邻居做 Ranking Hinge：
        d_B(u, v_pos) + margin < d_B(u, v_neg)
    未匹配token不参与
    """
    if idxA_m.numel() == 0:
        return A.new_tensor(0.)

    # A内邻域（用A自身的局部结构来定义拓扑）
    DA = pairwise_dist(A, metric=metric)                    # (T, T)
    neigh = DA.argsort(dim=-1)[:, 1:k+1]                    # (T, k)

    # B内距离（用于计算排名）
    DB = pairwise_dist(B, metric=metric)                    # (T, T)

    # 标记B中哪些token在匹配集中
    is_matched_B = torch.zeros(B.size(0), dtype=torch.bool, device=B.device)
    is_matched_B[idxB_m] = True

    # A中匹配token的快速映射：A[i] -> B[idxB_of_A[i]]（未匹配用-1）
    idxB_of_A = torch.full((A.size(0),), -1, device=A.device, dtype=torch.long)
    idxB_of_A[idxA_m] = idxB_m

    loss_sum = A.new_tensor(0.)
    count = 0

    for iA in idxA_m.tolist():
        uB = idxB_of_A[iA].item()  # iA 在 B 中的对应 uB
        if uB < 0:    # 未匹配，跳过
            continue

        # A中iA的k个邻居
        neigh_i = neigh[iA]                    # (k,)
        # 仅保留“也在匹配子集”的邻居（即这些邻居在B里也有对应）
        neigh_i_matched_mask = idxB_of_A[neigh_i] >= 0
        if not neigh_i_matched_mask.any():
            continue
        neigh_i_posA = neigh_i[neigh_i_matched_mask]        # A中的正样本邻居
        posB_idx = idxB_of_A[neigh_i_posA]                  # 对应到 B 的索引 (M_pos,)

        # 负样本：从B中未匹配的token里采样，数量与正样本相同
        neg_pool = (~is_matched_B).nonzero(as_tuple=False).flatten()  # 未匹配集合
        if neg_pool.numel() == 0:
            # 如果B里全部被匹配了，就从非正样本里采
            all_pool = torch.arange(B.size(0), device=B.device)
            exclude = torch.cat([posB_idx, torch.tensor([uB], device=B.device)])
            mask = torch.ones(B.size(0), dtype=torch.bool, device=B.device)
            mask[exclude] = False
            neg_pool = all_pool[mask]
            if neg_pool.numel() == 0:
                continue

        # 按需采样相同数量的负样本
        M_pos = posB_idx.numel()
        if neg_pool.numel() >= M_pos:
            rand_idx = torch.randint(0, neg_pool.numel(), (M_pos,), device=B.device)
            negB_idx = neg_pool[rand_idx]
        else:
            # 候选不足就有放回采样
            rand_idx = torch.randint(0, neg_pool.numel(), (M_pos,), device=B.device)
            negB_idx = neg_pool[rand_idx]

        # 计算排名hinge：d(uB, pos) vs d(uB, neg)
        d_pos = DB[uB, posB_idx]   # (M_pos,)
        d_neg = DB[uB, negB_idx]   # (M_pos,)
        loss = F.relu(d_pos - d_neg + margin).mean()

        loss_sum = loss_sum + loss
        count += 1

    if count == 0:
        return A.new_tensor(0.)
    return loss_sum / count

def partial_nrc_loss(z1, z2, k=5, margin=0.1, metric="cosine", max_pairs=None):
    """
    z1, z2: (B, T, C)
      - 同一 batch index 的 (z1[b], z2[b]) 是同一个 instance（不参与下面的跨实例约束）
      - 我们对 “不同 b != b' 的实例对” 做 partial NRC
    单射+未知对应：用 Mutual Nearest Neighbors 得到“部分匹配子集”，再做邻域排名一致性
    """
    B, T, C = z1.shape
    device = z1.device
    pairs = []

    # 生成所有跨实例对 (p, q), p != q
    for p in range(B):
        for q in range(B):
            if p == q: 
                continue
            pairs.append((p, q))

    # 如需减小计算量，可随机子采样若干对
    if (max_pairs is not None) and (len(pairs) > max_pairs):
        idx = torch.randperm(len(pairs), device=device)[:max_pairs].tolist()
        pairs = [pairs[i] for i in idx]

    total_loss = z1.new_tensor(0.)
    num_terms = 0

    for (p, q) in pairs:
        A = z1[p]  # (T, C)
        B_ = z2[q] # (T, C)

        # 1) 构造部分匹配（MNN）
        idxA_m, idxB_m = partial_matches_mnn(A, B_, metric=metric)

        # 2) 在这些部分匹配上做邻域排名一致性
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
        self.loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
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
        # x_in = torch.cat([img['src'],img['target']])
        x_in = torch.cat([x_in,x_in])
        hidden_states_out,last_out = self.vit(x_in) 
        B = hidden_states_out[0].shape[0]
        # single modality
        pred,mask = self.decoder(last_out=last_out)
        loss_rec = self.forward_loss(x_in,pred,mask)
        # return loss_rec, loss_rec, loss_rec
        # intra instance
        loss2 = 0
        loss3 = 0
        for hidden in hidden_states_out[-self.l:]:
            C = hidden.shape[1]
            z_hyp1 = hidden[:B//2,...].reshape(B//2,-1,C)
            z_hyp2 = hidden[B//2:,...].reshape(B//2,-1,C)
            # loss1 = simclr_hyp(z_hyp1,z_hyp2)
            for i in range(B//2):
                match_idx = torch.arange(z_hyp1.shape[1]).cuda()
                loss2 += neighborhood_ranking_crossbrain(z_hyp1[i],z_hyp2[i],match_idx,k=self.k)
                loss2 +=neighborhood_ranking_crossbrain(z_hyp2[i],z_hyp1[i],match_idx,k=self.k)
            loss3 += partial_nrc_loss(z_hyp1, z_hyp2, k=self.k, margin=self.margin, metric="cosine", max_pairs=6)
            loss3 += partial_nrc_loss(z_hyp2, z_hyp1, k=self.k, margin=self.margin, metric="cosine", max_pairs=6)

        loss2 = loss2/(self.l*B//2)
        loss3 = loss3/self.l
        return 10*loss2+loss_rec+10*loss3, 10*loss3, loss_rec

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