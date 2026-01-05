import torch
import torch.nn as nn

from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep


def random_masking(x: torch.Tensor, mask_ratio: float = 0.75):
    """
    x: (B, T, C) tokens
    return: masked_x, mask, ids_restore
    """
    B, T, C = x.shape
    len_keep = int(T * (1 - mask_ratio))

    noise = torch.rand(B, T, device=x.device)  # 每个token一个噪声
    ids_shuffle = torch.argsort(noise, dim=1)  # 从小到大排序
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, C))

    # 生成 mask：0 表示保留，1 表示mask
    mask = torch.ones([B, T], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore

class DecoderHead(nn.Module):
    def __init__(self, in_channels, upsample="vae", dim=768):
        super(DecoderHead, self).__init__()
        if upsample == "large_kernel_deconv":
            self.conv = nn.ConvTranspose3d(dim, in_channels, kernel_size=(32, 32, 32), stride=(32, 32, 32))
        elif upsample == "deconv":
            self.conv = nn.Sequential(
                nn.ConvTranspose3d(dim, dim // 2, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 2, dim // 4, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 4, dim // 8, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 8, dim // 16, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 16, in_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            )
        elif upsample == "vae":
            self.conv = nn.Sequential(
                nn.Conv3d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 2),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 2, dim // 4, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 4),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 4, dim // 8, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 8),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 8, dim // 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 16, dim // 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 16, in_channels, kernel_size=1, stride=1),
            )
        self.swap_ratio = 0.3
        self.mask_ratio = 0.0
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))

        
    def swap_tokens_same_idx(self,last_out: torch.Tensor, swap_ratio: float):
        """
        last_out: (B, T, C)
        swap_ratio: 0~1，交换的比例
        """
        B, T, C = last_out.shape
        assert B % 2 == 0
        k = int(round(swap_ratio * T))
        if k == 0:
            return last_out  # 不交换
        
        # 随机挑 k 个时间步（全 batch 共享）
        idx = torch.randperm(T, device=last_out.device)[:k]
        
        out = last_out.clone()                 # 避免原地修改带来 autograd 问题
        x1, x2 = out[:B//2], out[B//2:]        # (B/2, T, C)
        
        tmp = x1[:, idx, :].clone()
        x1[:, idx, :] = x2[:, idx, :]
        x2[:, idx, :] = tmp
        return out

    def forward(self, last_out):
        x_rec = self.conv(last_out)
        return x_rec, None

    def no_weight_decay(self):
        """Disable weight_decay on specific weights."""
        nwd = {'swinViT.absolute_pos_embed'}
        for n, _ in self.named_parameters():
            if 'relative_position_bias_table' in n:
                nwd.add(n)
        return nwd


class DecoderHeadUni(nn.Module):
    def __init__(self, in_channels=1, embed_dims=[64, 128, 320, 512], upsample="vae"):
        super(DecoderHeadUni, self).__init__()
        dim = embed_dims[-1]

        if upsample == "vae":
            # 与 UniFormer 的 4-stage 下采样对应：每层上采样一倍
            self.conv = nn.Sequential(
                # 对应 Stage4 -> Stage3
                nn.Conv3d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 2),
                nn.LeakyReLU(inplace=True),
                nn.Upsample(scale_factor=(2, 2, 2), mode="trilinear", align_corners=False),

                # 对应 Stage3 -> Stage2
                nn.Conv3d(dim // 2, dim // 4, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 4),
                nn.LeakyReLU(inplace=True),
                nn.Upsample(scale_factor=(2, 2, 2), mode="trilinear", align_corners=False),  # 深度方向缩放保持一致

                # 对应 Stage2 -> Stage1
                nn.Conv3d(dim // 4, dim // 8, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 8),
                nn.LeakyReLU(inplace=True),
                nn.Upsample(scale_factor=(2, 2, 2), mode="trilinear", align_corners=False),

                # 输出恢复到原始输入尺度
                nn.Conv3d(dim // 8, dim // 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 16),
                nn.LeakyReLU(inplace=True),
                nn.Upsample(scale_factor=(2, 2, 2), mode="trilinear", align_corners=False),

                # 输出最终重建层
                nn.Conv3d(dim // 16, in_channels, kernel_size=1, stride=1)
            )

        elif upsample == "deconv":
            self.conv = nn.Sequential(
                nn.ConvTranspose3d(dim, dim // 2, kernel_size=2, stride=2),
                nn.ConvTranspose3d(dim // 2, dim // 4, kernel_size=2, stride=2),
                nn.ConvTranspose3d(dim // 4, dim // 8, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
                nn.ConvTranspose3d(dim // 8, dim // 16, kernel_size=2, stride=2),
                nn.Conv3d(dim // 16, in_channels, kernel_size=1, stride=1)
            )

        else:
            raise ValueError(f"Unsupported upsample mode: {upsample}")

        # optional masking & swapping
        self.swap_ratio = 0.3
        self.mask_ratio = 0.0
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        """
        x: [B, 512, D', H', W']  # output of encoder
        """
        x_rec = self.conv(x)
        return x_rec, None