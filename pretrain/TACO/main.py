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

import argparse
import os
from time import time

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from losses.loss import GeCoLoss

# from models.geco_head_token import GeCoTokenHead
from optimizers.lr_scheduler import WarmupCosineSchedule
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from utils.data_utils import get_loader
from utils.ops import aug_rand, rot_rand, monai_aug,img_monai_aug
from monai.utils import set_determinism
from utils.visualization import tsne_visual
from utils.util import AverageMeter, distributed_all_gather
from tqdm import tqdm
from utils.ops import *
import matplotlib.pyplot as plt

import random
from typing import Dict, List, Any

def select_random_keys(batch: Dict[str, Any], num_keys: int = 2, exclude_pattern: str = 'meta') -> List[str]:
    """
    从batch中随机选择不包含特定模式的key
    
    Args:
        batch: 输入的数据字典
        num_keys: 要选择的key数量
        exclude_pattern: 要排除的模式字符串
    
    Returns:
        随机选择的key列表
    """
    # 过滤出不包含排除模式的key
    filtered_keys = [key for key in batch.keys() if exclude_pattern not in key.lower()]
    
    # 检查是否有足够的key
    if len(filtered_keys) == 0:
        return []

    if len(filtered_keys) < num_keys:
        return [filtered_keys[0],filtered_keys[0]]
        # raise ValueError(f"只有 {len(filtered_keys)} 个可用key，但需要选择 {num_keys} 个")
    
    # 随机选择指定数量的key
    selected_keys = random.sample(filtered_keys, num_keys)
    return selected_keys

def main():
    def save_ckp(state, checkpoint_dir):
        torch.save(state, checkpoint_dir)

    def train(args, global_step, train_loader, val_best, scaler):
        # img_transforms = monai_aug(args)
        model.train()
        run_loss = AverageMeter()
        topo_avg, rec_avg = AverageMeter(), AverageMeter()

        for step, batch in enumerate(train_loader):
            t1 = time()
            # import pdb;pdb.set_trace()
            keys = ['modal1','modal2']
            # keys = select_random_keys(batch)
            # if len(keys) == 0:
            #     print('bad data!')
            #     continue
            src = batch[keys[0]].cuda()
            target = batch[keys[1]].cuda()
            loss, loss_topo, loss_rec = model({'src':src,'target':target})
            
            if torch.isnan(loss).any() or torch.isinf(loss).any() or loss.item() > 1e5:
                print(f"⚠️ [NaN/Inf/Overflow] loss={loss.item():.3e}")
                torch.cuda.empty_cache()
                continue


            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.lrdecay:
                scheduler.step()
            optimizer.zero_grad()

            run_loss.update(loss.item(), n=args.batch_size)
            topo_avg.update(loss_topo.item(), n=args.batch_size)
            rec_avg.update(loss_rec.item(), n=args.batch_size)
            
            lr = optimizer.param_groups[0]["lr"]

            if args.distributed:
                if dist.get_rank() == 0:
                    print("Step:{}/{}, Loss:{:.4f}, Loss_topo:{:.4f}, Loss_rec:{:.4f} "
                      "lr:{:.8f}, Time:{:.4f}".format(global_step, args.num_steps,
                                                               run_loss.avg,topo_avg.avg,rec_avg.avg,
                                                               lr, time() - t1))
                    if global_step % 100 == 0:
                        writer.add_scalar("train/loss_total", scalar_value=run_loss.avg, global_step=global_step)
            else:
                print("Step:{}/{}, Loss:{:.4f}, Loss_topo:{:.4f}, Loss_rec:{:.4f} "
                      "lr:{:.8f}, Time:{:.4f}".format(global_step, args.num_steps,
                                                               run_loss.avg,topo_avg.avg,rec_avg.avg,
                                                               lr, time() - t1))
            

            if global_step % args.eval_num == 0:
                torch.cuda.empty_cache()

            global_step += 1
            if args.distributed:
                val_cond = (dist.get_rank() == 0) and (global_step % args.eval_num == 0)
            else:
                val_cond = global_step % args.eval_num == 0

            if val_cond:
                checkpoint = {
                        "global_step": global_step,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                save_ckp(checkpoint, logdir + f"/model_step{str(global_step)}.pt")
                # print(f"{'*'*5}start to validate {'*'*5}")
                # validation(args, test_loader,os.path.join(logdir,'val_feat.npz'))
                # tsne_visual(os.path.join(logdir,'val_feat.npz'),os.path.join(logdir,'distribution.png'))
                # image = plt.imread(os.path.join(logdir,'distribution.png'))
                # writer.add_image('t-SNE visualization', image, global_step=global_step, dataformats='HWC')
                print('saving checkpoint')
                model.train()
                writer.add_scalar("train/loss_total", scalar_value=run_loss.avg, global_step=global_step)
                writer.add_scalar("train/loss_topo", scalar_value=topo_avg.avg, global_step=global_step)
                writer.add_scalar("train/loss_rec", scalar_value=rec_avg.avg, global_step=global_step)
        return global_step, loss


    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
    parser.add_argument("--model_name", default="swin", type=str, help="directory to save the tensorboard logs")
    parser.add_argument("--epochs", default=100, type=int, help="number of training epochs")
    # parser.add_argument("--local_rank", default=0, type=int, help="number of training epochs")
    parser.add_argument("--num_steps", default=100000, type=int, help="number of training iterations")
    parser.add_argument("--adversarial", action="store_true", help="adversarial loss")
    parser.add_argument("--visual_steps", default=100, type=int, help="number of training iterations")
    parser.add_argument("--eval_num", default=100, type=int, help="evaluation frequency")
    parser.add_argument("--warmup_steps", default=500, type=int, help="warmup steps")
    parser.add_argument("--feature_dim", default=512, type=int, help="warmup steps")
    parser.add_argument("--num_domains", default=2, type=int, help="domain numbers")
    parser.add_argument("--queue_num", default=200, type=int, help="domain numbers")
    parser.add_argument("--crop_foreground", action="store_true", help="use monai Dataset class")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--modality", default="PET_CT", type=str, help="PET/CT/PET_CT")
    parser.add_argument("--feature_size", default=48, type=int, help="embedding size")
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
    parser.add_argument("--data_type", default="data_1k", type=str,)
    parser.add_argument("--use_last_layer", action="store_true")
    parser.add_argument("--use_geo", action="store_true")
    parser.add_argument("--use_cl", action="store_true")
    parser.add_argument("--use_sharp", action="store_true")
    parser.add_argument("--random_seed", default=20, type=int, help="random seed")
    parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
    parser.add_argument("--a_min", default=-1000, type=float, help="a_min in ScaleIntensityRanged")
    parser.add_argument("--a_max", default=1000, type=float, help="a_max in ScaleIntensityRanged")
    parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
    parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
    parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
    parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
    parser.add_argument("--space_z", default=1.5, type=float, help="spacing in z direction")
    parser.add_argument("--alpha_domain", default=1, type=float, help="factor of domain loss")
    parser.add_argument("--alpha_adv", default=1, type=float, help="factor of adversarial loss")
    parser.add_argument("--roi_large", default=288, type=int, help="roi size in x direction")
    parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
    parser.add_argument("--batch_size", default=2, type=int, help="number of batch size")
    parser.add_argument("--sw_batch_size", default=2, type=int, help="number of sliding window batch size")
    parser.add_argument("--num_geo_layer", default=2, type=int, help="which layer to apply ugco")
    parser.add_argument("--lr", default=4e-4, type=float, help="learning rate")
    parser.add_argument("--decay", default=0.1, type=float, help="decay rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--lrdecay", action="store_true", help="enable learning rate decay")
    parser.add_argument("--token_head", action="store_true", help="without teacher")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="maximum gradient norm")
    parser.add_argument("--loss_type", default="SSL", type=str)
    parser.add_argument("--opt", default="adamw", type=str, help="optimization algorithm")
    parser.add_argument("--lr_schedule", default="warmup_cosine", type=str)
    parser.add_argument("--resume", default=None, type=str, help="resume training")
    parser.add_argument("--local-rank", type=int, default=0, help="local rank")
    parser.add_argument("--grad_clip", action="store_true", help="gradient clip")
    parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--smartcache_dataset", action="store_true", help="use monai smartcache Dataset")
    parser.add_argument("--cache_dataset", action="store_true", help="use monai cache Dataset")
    parser.add_argument("--data_dir", default="/dataset/dataset0/", type=str, help="dataset directory")
    parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")

    args = parser.parse_args()
    logdir = args.logdir
    args.amp = not args.noamp
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)
    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        print(f'WORLD_SIZE {int(os.environ["WORLD_SIZE"])}')
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
        args.wdsize = int(os.environ["WORLD_SIZE"])
    args.epochs = args.num_steps / (args.batch_size*args.wdsize)
    args.device = "cuda:0"
    args.world_size = 1
    args.rank = 0

    if args.distributed:
        args.device = "cuda:%d" % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method=args.dist_url)
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        print(
            "Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."
            % (args.rank, args.world_size)
        )
    else:
        print("Training with a single process on 1 GPUs.")
    assert args.rank >= 0

    if args.rank == 0:
        os.makedirs(logdir, exist_ok=True)
        writer = SummaryWriter(logdir)
    else:
        writer = None


    random_seed = args.random_seed
    if random_seed is not None and (isinstance(random_seed, int) or isinstance(random_seed, float)):
        set_determinism(seed=random_seed)
    
    if args.model_name == 'swin':
        from models.taco import Taco
        model = Taco(args,writer)
    elif args.model_name == 'uniformer':
        from models.taco_uni import Taco
        model = Taco(args,writer)
    
    model.cuda()

    if args.opt == "adam":
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.decay)

    elif args.opt == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.decay)

    elif args.opt == "sgd":
        optimizer = optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)
    global_step = 0
    if args.resume:
        model_pth = args.resume
        model_dict = torch.load(model_pth)
        state_dict = model_dict["state_dict"]
        if "module." in list(state_dict.keys())[0]:
            print("Tag 'module.' found in state dict - fixing!")
            for key in list(state_dict.keys()):
                state_dict[key.replace("module.", "")] = state_dict.pop(key)
        model.load_state_dict(state_dict,strict=True)
        global_step = model_dict['global_step']

    if args.lrdecay:
        if args.lr_schedule == "warmup_cosine":
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps)

        elif args.lr_schedule == "poly":

            def lambdas(step):
                return (1 - float(step/(args.batch_size*args.wdsize)) / float(args.epochs)) ** 0.9

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)
        else:
            scheduler = None
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[args.local_rank],find_unused_parameters=True)
    
    train_loader = get_loader(args)

    
    best_val = 1e8
    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None
    # from ptflops import get_model_complexity_info
    # macs, params = get_model_complexity_info(model, (1, 96, 96, 96), as_strings=True, 
    #                                                     print_per_layer_stat=True)

    while global_step < args.num_steps:
        global_step, loss = train(args, global_step, train_loader, best_val, scaler)
    checkpoint = {"epoch": args.epochs, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}

    if args.distributed:
        if dist.get_rank() == 0:
            torch.save(model.state_dict(), logdir + "/final_model.pth")
        dist.destroy_process_group()
    else:
        torch.save(model.state_dict(), logdir + "/final_model.pth")
    save_ckp(checkpoint, logdir + "/model_final_epoch.pt")


if __name__ == "__main__":
    main()
