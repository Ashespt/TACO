import torch
import torch.nn as nn
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
from models.decoder_head import DecoderHeadUni
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
from models.uniformer_blocks import uniformer_small, uniformer_base

# ---- Poincaré ball basic ops ----
def proj(x, eps=1e-5):
    # 投影到球内，避免数值越界
    norm = x.norm(dim=-1, keepdim=True)
    max_norm = 1 - eps
    scale = torch.clamp(max_norm / (norm + 1e-12), max=1.0)
    return x * scale

def exp_map_euclid_to_poincare(x, c=1.0, eps=1e-5):
    # 欧氏 -> 双曲（Poincaré）指数映射，默认以 0 为切点
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    return proj(torch.tanh(math.sqrt(c) * norm) * x / (math.sqrt(c) * norm))

def hyp_dist(x, y, c=1.0, eps=1e-5):
    # Poincaré 双曲距离
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    diff2 = ((x - y) * (x - y)).sum(dim=-1, keepdim=True)
    num = 2 * diff2
    den = (1 - x2) * (1 - y2)
    z = 1 + num / (den + eps)
    return torch.acosh(z.clamp_min(1 + 1e-6)).squeeze(-1)

def cos_dist(x, y, eps=1e-8):
    # 计算余弦距离（1 - 余弦相似度）
    x_norm = F.normalize(x, dim=-1)
    y_norm = F.normalize(y, dim=-1)
    cos_sim = (x_norm * y_norm).sum(dim=-1)
    cos_sim = cos_sim.clamp(-1 + eps, 1 - eps)  # 避免数值不稳定
    cos_dist = 1 - cos_sim
    return cos_dist

def mobius_add(x, y, eps=1e-5):
    xy = (x * y).sum(dim=-1, keepdim=True)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    num = (1 + 2*xy + y2) * x + (1 - x2) * y
    den = 1 + 2*xy + x2 * y2
    return proj(num / (den + eps))

def mobius_scalar_mul(a, x):
    # a ⊙ x （标量 a 的 Möbius 缩放）
    ax_norm = (a * x).norm(dim=-1, keepdim=True)
    x_norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return proj(torch.tanh(ax_norm / x_norm) * x / torch.tanh(x_norm))

def poincare_barycenter(points, weights=None, iters=5):
    # 简化/近似的双曲重心（Frechet mean）——足够用于原型计算
    if weights is None:
        weights = torch.ones(points.size(0), device=points.device)
    weights = (weights / (weights.sum() + 1e-12)).view(-1, 1)
    y = proj((points * weights).sum(dim=0, keepdim=True))  # init (欧氏均值后投影)
    for _ in range(iters):
        # 近似更新：Möbius 加权平均
        upd = None
        for i in range(points.size(0)):
            term = mobius_scalar_mul(weights[i, 0], points[i:i+1])
            upd = term if upd is None else mobius_add(upd, term)
        y = proj(upd)
    return y.squeeze(0)

def build_radial_targets(levels, r_min=0.10, r_max=0.85):
    """
    目标半径：高层靠内、底层靠外，线性或对数插值均可。
    """
    L = levels.max().item()
    L = max(L, 1)
    t = levels.float() / L  # 0..1
    # 可以换成指数映射：r = r_min * (r_max/r_min) ** t
    r = r_min + (r_max - r_min) * t
    return r

def radial_loss(z, levels, r_target, w=1.0):
    r = z.norm(dim=-1).clamp_max(0.999)
    return w * F.smooth_l1_loss(r, r_target)

def parent_child_loss(z, parent_idx, margin=0.05, c=1.0):
    """
    约束：子-父距离 < 子-叔距离；并鼓励子更靠外（半径更大一点）
    """
    N = z.size(0)
    mask_child = parent_idx >= 0
    idx_child = torch.nonzero(mask_child).squeeze(-1)
    if idx_child.numel() == 0:
        return z.new_tensor(0.)
    p = parent_idx[idx_child]
    d_ip = hyp_dist(z[idx_child], z[p], c=c)
    # 随机选取一个“叔叔/其他父”的候选作为负样本
    neg = torch.randint(0, N, (idx_child.numel(),), device=z.device)
    d_in = hyp_dist(z[idx_child], z[neg], c=c)
    loss_rank = F.relu(d_ip - d_in + margin).mean()
    # 半径单调：子离边界更近（半径更大）
    r_child = z[idx_child].norm(dim=-1)
    r_parent = z[p].norm(dim=-1)
    loss_radius_mono = F.relu(r_parent + 0.01 - r_child).mean()
    return loss_rank + 0.5 * loss_radius_mono

def sibling_separation_loss(z, levels, branch_id, tau=0.15, c=1.0):
    """
    同层不同分支要分开：用“原型角度/方向”或直接原型间距。
    这里用每个分支的双曲重心作原型，最大化不同分支原型之间的间隔；
    同时让 token 靠近自己分支的原型（pull）。
    """
    loss = z.new_tensor(0.)
    for L in torch.unique(levels):
        idx = torch.nonzero(levels == L).squeeze(-1)
        b_ids = torch.unique(branch_id[idx])
        if b_ids.numel() <= 1: 
            continue
        # 原型
        protos, proto_list = [], []
        for b in b_ids:
            j = idx[branch_id[idx] == b]
            if j.numel() == 0: 
                continue
            proto = poincare_barycenter(z[j])
            protos.append(proto)
            proto_list.append((b.item(), proto))
        if len(protos) <= 1:
            continue
        P = torch.stack(protos, dim=0)  # (#branches, D)
        # 拉近：token → 自己原型
        pull = []
        for b, proto in proto_list:
            j = idx[branch_id[idx] == b]
            pull.append(hyp_dist(z[j], proto.unsqueeze(0)).mean())
        pull = torch.stack(pull).mean()
        # 推开：原型之间作为 N-pair（InfoNCE on prototypes）
        Dpp = hyp_dist(P.unsqueeze(1), P.unsqueeze(0))  # (B,B)
        logits = -Dpp / tau
        labels = torch.arange(P.size(0), device=z.device)
        push = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) * 0.5
        loss = loss + pull + 0.3 * push
    return loss

def neighborhood_ranking_crossbrain(z1, z2, match_idx, k=5, margin=0.05, c=1.0):
    """
    跨脑的局部拓扑保持（邻域排序一致性，双曲版，向量化快速实现）
    """
    N = z1.size(0)
    D1 = torch.cdist(z1, z1)                 # 这里用欧氏近邻定义；也可换 hyp_dist
    neigh = D1.argsort(dim=-1)[:, 1:k+1]     # (N,k)

    i = torch.arange(N, device=z1.device).unsqueeze(1).expand_as(neigh)
    j = neigh
    i2 = match_idx[i]                        # (N,k)
    j2 = match_idx[j]

    # 负样本随机选（可替换为“非近邻中采样”）
    neg = torch.randint(0, N, (N, k), device=z1.device)
    n2 = match_idx[neg]

    d_pos = cos_dist(z2[i2], z2[j2])
    d_neg = cos_dist(z2[i2], z2[n2])
    # d_pos = hyp_dist(z2[i2], z2[j2], c=c)
    # d_neg = hyp_dist(z2[i2], z2[n2], c=c)
    return F.relu(d_pos - d_neg + margin).mean()

def simclr_hyp(z_a, z_b, temperature=0.2, c=1.0):
    # 双曲 InfoNCE：用 -d_H 作为相似度
    D = hyp_dist(z_a.unsqueeze(1), z_b.unsqueeze(0), c=c)  # (N,N)
    logits = -D / temperature
    labels = torch.arange(z_a.size(0), device=z_a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

def hyperbolic_hierarchy_step(z_euclid, levels, branch_id,
                              match_idx=None, z_euclid_peer=None,
                              w_rad=1.0, w_pc=1.0, w_sib=1.0, w_rank=1.0, w_ssl=1.0,
                              c=1.0):
    """
    z_euclid: (N,D) 当前脑的 token 特征（欧氏）
    z_euclid_peer: (N,D) 对应另一脑的 token（用于跨脑约束/InfoNCE）
    match_idx: (N,)  z_euclid[i] ↔ z_euclid_peer[match_idx[i]]
    """
    # 欧氏->双曲
    z = exp_map_euclid_to_poincare(z_euclid, c=c)
    loss = z.new_tensor(0.)

    # 目标半径（层级单调）
    r_tgt = build_radial_targets(levels).to(z.device)
    loss += w_rad * radial_loss(z, levels, r_tgt)

    # 父子约束（最近父，半径单调）
    # loss += w_pc * parent_child_loss(z, parent_idx, margin=0.05, c=c)

    # 同层分支：拉近本分支原型，推开兄弟分支原型
    loss += w_sib * sibling_separation_loss(z, levels, branch_id, tau=0.15, c=c)

    # 跨脑邻域排序（可选）
    if (match_idx is not None) and (z_euclid_peer is not None):
        z_peer = exp_map_euclid_to_poincare(z_euclid_peer, c=c)
        loss += w_rank * neighborhood_ranking_crossbrain(z, z_peer, match_idx, k=5, margin=0.05, c=c)
        # 也可以再加一个跨脑的 InfoNCE
        loss += w_ssl * simclr_hyp(z, z_peer, temperature=0.2, c=c)

    return loss

def rank_correlation_loss(z1, z2, match_matrix=None, c=1.0):
    # z1, z2: 双曲特征
    D1 = hyp_dist(z1.unsqueeze(1), z1.unsqueeze(0), c=c)
    D2 = hyp_dist(z2.unsqueeze(1), z2.unsqueeze(0), c=c)
    D1_rank = D1.argsort(dim=-1)
    D2_rank = D2.argsort(dim=-1)

    loss = 0.0
    N = z1.size(0)
    for i in range(N):
        # 选取z1[i]对应的z2的加权索引（若有匹配矩阵）
        if match_matrix is not None:
            j = match_matrix[i].argmax()
        else:
            j = i
        # Spearman correlation approximation
        diff = (D1_rank[i].float() - D2_rank[j].float()).abs()
        loss += diff.mean()
    return loss / N


class projection_head(nn.Module):
    def __init__(self, in_dim:int=768, hidden_dim:int=2048, out_dim:int=2048):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, affine=False, track_running_stats=False),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, affine=False, track_running_stats=False),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x): 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


# def pairwise_dist(X):
#     # X: [N, d]
#     dist = torch.cdist(X, X, p=2)  # GPU 加速
#     return dist

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



class Topo(nn.Module):
    def __init__(self, args,writer=None,exp=200,dim=768,num_patch_side=4,spatial_dims=3,loss_function=GeCoLoss(),norm_pix_loss=True):
        super(Topo, self).__init__()
        self.vit = uniformer_small(in_chans=1)
        
        self.args = args
        
        self.iter = 0
        self.writer = writer
        self.norm_pix_loss = norm_pix_loss
        self.decoder = DecoderHeadUni(args.in_channels,upsample="vae")
        self.loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)


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
        # if self.norm_pix_loss:
        #     mean = target.mean(dim=-1, keepdim=True)
        #     var = target.var(dim=-1, keepdim=True)
        #     target = (target - mean) / (var + 1.e-6)**.5
        # if self.norm_pix_loss:
        #     mean = target.mean(dim=-1, keepdim=True)
        #     var = target.var(dim=-1, keepdim=True)
        #     target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1).mean()  # [N, L], mean loss per patch
        # loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    
    def forward(self, img:dict,visual_flag:bool=False,step:int=0):
        
        x_in = torch.cat([img['src'],img['target']])
        hidden_states_out = self.vit(x_in)
        last_out = hidden_states_out[-1]
        # reconstruction loss
        B = last_out.shape[0]
        pred,mask = self.decoder(last_out)
        loss_rec = self.forward_loss(x_in,pred,mask)

        # intra instance
        loss2 = 0
        for hidden in hidden_states_out[-2:]:
            C = hidden.shape[1]
            z_hyp1 = hidden[:B//2,...].reshape(B//2,-1,C)
            z_hyp2 = hidden[B//2:,...].reshape(B//2,-1,C)
            # loss1 = simclr_hyp(z_hyp1,z_hyp2)
            for i in range(B//2):
                match_idx = torch.arange(z_hyp1.shape[1]).cuda()
                loss2 += neighborhood_ranking_crossbrain(z_hyp1[i],z_hyp2[i],match_idx)

        loss2 = loss2/(B//2)
        # inter instance

        loss3 = partial_nrc_loss(z_hyp1, z_hyp2, k=5, margin=0.1, metric="cosine", max_pairs=6)
        return 10*loss2+loss_rec+10*loss3, 10*loss2, loss_rec
        
        # loss_total, loss_radial, loss_parent_child, loss_sibling, loss_crossbrain = 0,0,0,0,0
        # for i in range(B//2):
        #     total_tokens = sum(h[i].shape[1]*h[i].shape[2]*h[i].shape[3] for h in hidden1)
        #     match_idx = torch.arange(total_tokens).cuda()
        #     tokens0, levels0 = [], []
        #     for level, feat in enumerate(hidden0[::-1]):  # 最高层level=0
        #         B, C, D, H, W = feat.shape
        #         t = feat.permute(0, 2, 3, 4, 1).reshape(B, -1, C)  # [B, N, C]
        #         tokens0.append(t)
        #         levels0.append(torch.full((B, t.size(1)), level, device=feat.device))
        #     tokens1, levels1 = [], []
        #     for level, feat in enumerate(hidden1[::-1]):  # 最高层level=0
        #         B, C, D, H, W = feat.shape
        #         t = feat.permute(0, 2, 3, 4, 1).reshape(B, -1, C)  # [B, N, C]
        #         tokens1.append(t)
        #         levels1.append(torch.full((B, t.size(1)), level, device=feat.device))

        #     hyperbolic_hierarchy_step(tokens0,levels1,)

        
        # return loss_total/(B//2)+loss_rec, loss_radial/(B//2), loss_rec

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