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

import numpy as np
import scipy.ndimage as ndimage
import torch


def sinkhorn_distance(x, y, epsilon=0.01, n_iters=100, p=2):
    """
    Compute Sinkhorn approximation of the Wasserstein distance between two point clouds.

    Args:
        x, y: [N, D] and [M, D] torch tensors
        epsilon: regularization coefficient (entropy weight)
        n_iters: number of Sinkhorn iterations
        p: norm degree (default=2 for L2 distance)
    
    Returns:
        sinkhorn_distance: scalar (approximate Wasserstein distance)
    """
    # Number of samples
    n, m = x.size(0), y.size(0)

    # Compute cost matrix C[i,j] = ||x_i - y_j||_p^p
    C = torch.cdist(x, y, p=p) ** p

    # Uniform marginals
    a = torch.full((n,), 1.0 / n, device=x.device)
    b = torch.full((m,), 1.0 / m, device=y.device)

    # Kernel matrix K = exp(-C / epsilon)
    K = torch.exp(-C / epsilon)

    # Initialize scaling vectors
    u = torch.ones_like(a)
    v = torch.ones_like(b)

    # Sinkhorn iterations
    for _ in range(n_iters):
        u = a / (K @ v)
        v = b / (K.t() @ u)

    # Transport plan
    T = torch.diag(u) @ K @ torch.diag(v)

    # Sinkhorn distance (expected cost)
    dist = torch.sum(T * C)

    return dist.sqrt()

def resample_3d(img, target_size):
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = (float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
    img_resampled = ndimage.zoom(img, zoom_ratio, order=0, prefilter=False)
    return img_resampled


def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def distributed_all_gather(
    tensor_list, valid_batch_size=None, out_numpy=False, world_size=None, no_barrier=False, is_valid=None
):
    if world_size is None:
        world_size = torch.distributed.get_world_size()
    if valid_batch_size is not None:
        valid_batch_size = min(valid_batch_size, world_size)
    elif is_valid is not None:
        is_valid = torch.tensor(bool(is_valid), dtype=torch.bool, device=tensor_list[0].device)
    if not no_barrier:
        torch.distributed.barrier()
    tensor_list_out = []
    with torch.no_grad():
        if is_valid is not None:
            is_valid_list = [torch.zeros_like(is_valid) for _ in range(world_size)]
            torch.distributed.all_gather(is_valid_list, is_valid)
            is_valid = [x.item() for x in is_valid_list]
        for tensor in tensor_list:
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gather_list, tensor)
            if valid_batch_size is not None:
                gather_list = gather_list[:valid_batch_size]
            elif is_valid is not None:
                gather_list = [g for g, v in zip(gather_list, is_valid_list) if v]
            if out_numpy:
                gather_list = [t.cpu().numpy() for t in gather_list]
            tensor_list_out.append(gather_list)
    return tensor_list_out