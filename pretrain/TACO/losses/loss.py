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
from torch.nn import functional as F
from .coderating import MaximalCodingRateReduction
from .loftrloss import LoFTRLoss
import random
import numpy as np
import torch.nn as nn
from .superglue import log_optimal_transport

class GeCoContrast(object):
    def __init__(self, temperature=0.9):
        super().__init__()
        self.temp = temperature
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
    
    def __call__(self, q,all_k):
        N = q.shape[0]
        # q = F.normalize(q,dim=1)
        # all_k = F.normalize(all_k,dim=1)
        # sim = torch.einsum("nc,kc->nk",[q,all_k])
        # sim = (1+torch.einsum("nc,kc->nk",[q,all_k]))/2
        sim = (1+F.cosine_similarity(q, all_k, dim=1).unsqueeze(0))/2
        # sim = F.relu(sim)
        sim = torch.clamp(sim, 1e-6, 1-1e-6)
        l_pos = torch.diag(sim).unsqueeze(-1)  # positive logits Nx1
        l_neg = sim[:,N:] # negative logits Nxque_k_num
        # infonce
        # logits = torch.cat([l_pos,l_neg],dim=1)
        # logits /= self.temp
        # labels = torch.zeros(logits.shape[0],dtype=torch.long).cuda()
        # loss = self.criterion(logits, labels)
        # return loss
        # ce
        loss = - torch.log(l_pos)
        loss_neg = - torch.log(1 - l_neg)
        return (torch.mean(loss)+torch.mean(loss_neg))/2

class Loss(torch.nn.Module):
    def __init__(self, batch_size, args):
        super().__init__()
        #self.recon_loss = torch.nn.L1Loss().cuda()
        self.contrast_loss = Contrast(args, batch_size)
        self.rate_loss = MaximalCodingRateReduction(eps=0.5)
        self.adv_loss = torch.nn.NLLLoss()
        self.alpha1 = 1.0
        self.alpha2 = 1.0
        # self.alpha3 = 
        self.alpha_domain = args.alpha_domain
        self.alpha_adv = args.alpha_adv

    def __call__(self, output_contrastive,target_contrastive,domain,num_cls=2,output_adv=None):
        contrast_loss = self.alpha2 * self.contrast_loss(output_contrastive, target_contrastive)
        #recon_loss = self.alpha3 * self.recon_loss(output_recons, target_recons)
        ratedistortion_loss = self.alpha_domain*0.5*(self.rate_loss(output_contrastive,domain,num_classes=2)[0]+self.rate_loss(target_contrastive,domain,num_classes=2)[0])
        if output_adv is not None:
            adv_loss = self.alpha_adv*self.adv_loss(output_adv[0],output_adv[1])
            total_loss =  contrast_loss + ratedistortion_loss + adv_loss
            return total_loss, (contrast_loss,ratedistortion_loss,adv_loss)
        else:
            total_loss = contrast_loss + ratedistortion_loss

            return total_loss, (contrast_loss,ratedistortion_loss,0)



# def save_fig(mask):
#     pl
class SharpnessLoss(nn.Module):
    def __init__(self,temperature=0.9):
        super().__init__()
        self.temp = temperature

    def forward(self,q,k,labels,step=0): # q: src k: aug
        # q,k Bx16xdim
        B_q,_,_ = q.shape
        loss = []
        # labels = torch.zeros(all_k.shape[0], dtype=torch.long).cuda()
        # labels[:num_patch]=1
        losses = []
        labels = labels.cuda()
        sharpness_list = []
        for i in range(B_q):
            label = labels[i]
            sim_matrix = (1+torch.einsum("lc,sc->ls", q[i], k[i]))/2
            sharpness1 = F.softmax((torch.max(sim_matrix,dim=0)[0]-torch.mean(sim_matrix, dim=0))/torch.std(sim_matrix, dim=0),dim=0)
            # sharpness1 = (torch.max(sim_matrix,dim=0)[0]-torch.mean(sim_matrix, dim=0))/torch.std(sim_matrix, dim=0)
            sharpness_list.append(sharpness1)
            sharpness2 = F.softmax((torch.max(sim_matrix, dim=1)[0]-torch.mean(sim_matrix, dim=1))/torch.std(sim_matrix, dim=1),dim=0)
            pos_mask = label == 1
            neg_mask = label == 0
            sim_matrix = torch.clamp(sim_matrix, 1e-6, 1-1e-6)
            loss = - torch.log(sim_matrix)
            loss_neg = - torch.log(1 - sim_matrix[neg_mask])
            loss1 = loss*sharpness1[None,...]
            loss2 = loss*sharpness2[...,None]
            # import pdb;pdb.set_trace()
            # save_image(loss,(loss1+loss2)/2,"visual_local1_global0_sharp0",step)
            loss1 = torch.sum(loss1[pos_mask])
            loss2 = torch.sum(loss2[pos_mask])
            loss3 = torch.mean(loss_neg)
            losses.append((loss1+loss2)/2+loss3)

        return sharpness_list,torch.mean(torch.stack(losses))


import numpy as np
import numpy as np
import matplotlib.pyplot as plt
def save_sharp(sim,ssim,logdir,step):
    sim=sim.cpu().detach().numpy()
    ssim=ssim.cpu().detach().numpy()
    # plt.figure(figsize=(8, 8))
    # plt.imshow(pred, cmap='viridis', aspect='auto')
    # plt.title('Heatmap of Geco')
    # plt.savefig(f'{logdir}/heatmap_wo{str(step)}.png', dpi=300, bbox_inches='tight')
    # plt.close()
    # plt.figure(figsize=(8, 8))
    # plt.imshow(wpred, cmap='viridis', aspect='auto')
    # plt.title('Heatmap of GT')
    # plt.savefig(f'{logdir}/heatmap_w{str(step)}.png', dpi=300, bbox_inches='tight')
    # plt.close()

class GeometricLoss(object):
    def __init__(self, temperature=0.07,sinkhorn=False):
        super().__init__()
        self.temp = temperature
        self.loss = LoFTRLoss().cuda()
        self.sinkhorn = sinkhorn
        self.bin_score=torch.nn.Parameter(
                torch.tensor(1.0, requires_grad=True)).cuda()
        self.skh_iters=100

    def __call__(self,q,k,labels): # q: src k: aug
        # q,k Bx16xdim
        B_q,_,_ = q.shape
        loss = []
        # labels = torch.zeros(all_k.shape[0], dtype=torch.long).cuda()
        # labels[:num_patch]=1
        losses = []
        pos_losses = []
        neg_losses = []
        for i in range(B_q):
            label = labels[i]
            all_k = k[i][None,...]#(1xnum_patchxdim
            # sinkhorn
            if self.sinkhorn:
                sim_matrix = torch.einsum("nlc,nsc->nls", q[i][None,...], all_k) / self.temp
                log_assign_matrix = log_optimal_transport(sim_matrix,self.bin_score,self.skh_iters)
                assign_matrix = log_assign_matrix.exp()[:,:-1,:-1]
                gt = torch.zeros(assign_matrix.shape).cuda()
                gt[0,:,:]=label
                # loss = F.cross_entropy(assign_matrix, gt)
                loss,pos_loss,neg_loss = self.loss(assign_matrix, gt)
            else:
                # dual_softmax
                sim_matrix = torch.einsum("nlc,nsc->nls", q[i][None,...], all_k)/self.temp
                conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)
                gt = torch.zeros(conf_matrix.shape).cuda()
                gt[0,:,:]=label
                loss,pos_loss,neg_loss = self.loss(conf_matrix, gt)
            losses.append(loss)
            pos_losses.append(pos_loss)
            neg_losses.append(neg_loss)
        return torch.mean(torch.stack(losses)),torch.mean(torch.stack(pos_losses)),torch.mean(torch.stack(neg_losses))

class GeCoLoss(torch.nn.Module):
    def __init__(self,weight_local=0.5,weight_sharp=0.5,sinkhorn=False):
        super().__init__()
        self.contrast_loss = GeCoContrast()
        self.geco_loss = GeometricLoss(sinkhorn=sinkhorn)
        self.sharp_loss = SharpnessLoss().cuda()
        self.weight_local = weight_local
        self.weight_sharp = weight_sharp
    def __call__(self,q_geo,q_cl,geo_pos=None,cl_pos=None,que_k_geo=None,que_k_cl=None,label=None,cl_pos_save=None,step=0):
        if geo_pos !=None:
            loss_geo,geo_pos_loss,geo_neg_loss = self.geco_loss(q_geo,geo_pos,label)#src:q_geo x aug:geo_pos
            sharpness,loss_sharp = self.sharp_loss(q_geo,geo_pos,label,step)
            loss_geo = ((1-self.weight_sharp)*loss_geo+self.weight_sharp*loss_sharp)*self.weight_local
            # loss_geo = loss_sharp
            # geo_neg_loss = torch.tensor(0).cuda()
            # geo_pos_loss = torch.tensor(0).cuda()
        else:
            loss_geo = torch.tensor(0).cuda()
            geo_pos_loss,geo_neg_loss,loss_sharp=loss_geo,loss_geo,loss_geo
        if cl_pos!=None:
            N,P,_ = q_cl.shape
            idx = random.sample(range(P),N)
            
            q_cl = torch.cat([q_cl[i,j,...][None,...] for i,j in zip(range(N),idx)])
            
            if que_k_cl is not None:
                cl_pos_save = torch.cat([cl_pos_save[i,j,...][None,...] for i,j in zip(range(N),idx)])
                for i,j in zip(range(N),idx):
                    tmp = cl_pos[i,0,:].clone()
                    cl_pos[i,0,:] = cl_pos[i,j,:]
                    cl_pos[i,j,:] = tmp
                # cl_pos = torch.cat([cl_pos[i,j,...][None,...] for i,j in zip(range(N),idx)])
                cl_pos = torch.cat([cl_pos[i,0,...][None,...] for i in range(N)]+[cl_pos[i,1:,...] for i in range(N)])
                cl_pos = torch.cat([cl_pos,que_k_cl])
            else:
                for i,j in zip(range(N),idx):
                    tmp = cl_pos[i,0,:].clone()
                    cl_pos[i,0,:] = cl_pos[i,j,:]
                    cl_pos[i,j,:] = tmp
                cl_pos_save = torch.cat([cl_pos[i,0,...][None,...] for i in range(N)])
                cl_pos = torch.cat([cl_pos[i,0,...][None,...] for i in range(N)]+[cl_pos[i,1:,...] for i in range(N)])
            loss_cl = self.contrast_loss(q_cl,cl_pos)
        else:
            loss_cl = torch.tensor(0).cuda()
            cl_pos_save = torch.tensor(0).cuda()
        loss_cl = loss_cl*(1-self.weight_local)
        return  sharpness,cl_pos_save,loss_geo+loss_cl,geo_pos_loss,geo_neg_loss,loss_sharp,loss_cl