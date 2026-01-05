import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Dict, Tuple


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

# --- 工具：tokens → (B,C,D,H,W) 网格 ---
def _tokens_to_grid(
    x: torch.Tensor,                        # (B, T, C)
    grid_size: Tuple[int, int, int],        # (8,8,8) for 128/16
    unmerge_fn: Optional[Callable] = None,  # 把 T 恢复到 512 的函数
) -> torch.Tensor:
    B, T, C = x.shape
    D, H, W = grid_size
    target = D * H * W  # 512

    if unmerge_fn is not None:
        x = unmerge_fn(x)                   # -> (B, 512, C)
        B, T, C = x.shape

    if T != target:
        raise RuntimeError(f"Token count mismatch: got {T}, expected {target}. Provide a proper unmerge_fn to restore merged tokens.")

    return x.transpose(1, 2).contiguous().view(B, C, D, H, W)

# --- 小的积木 ---
class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, norm="instance"):
        super().__init__()
        Norm = nn.InstanceNorm3d if norm == "instance" else nn.BatchNorm3d
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            Norm(out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            Norm(out_ch),
            nn.GELU(),
        )
    def forward(self, x): return self.block(x)

class UpOnly3D(nn.Module):
    def __init__(self, in_ch, out_ch, norm="instance"):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock3D(out_ch, out_ch, norm=norm)
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

# --- 只用 last_out 的 MAE 风格 3D 解码器（无 skip） ---
class MAEDecoder3D(nn.Module):
    """
    输入: last_out (B, T_last, C) + unmerge_fns["last"]
    流程: last_out --unmerge--> 512 tokens -> (B, C, 8, 8, 8) -> 8→16→32→64→128 -> 1×1×1 输出
    """
    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 16,
        embed_dim: int = 768,               # 必须等于 ViT hidden_size
        out_channels: int = 1,
        channels=(256, 128, 64, 32, 16, 8),    # 8→16→32→64→128 的通道序列
        norm: str = "instance",
        lazy_infer: bool = False,           # True 时用 LazyConv3d 自动适配 embed_dim
        mask_ratio=0.75
    ):
        super().__init__()
        assert img_size % patch_size == 0
        self.grid = (img_size // patch_size,) * 3  # (8,8,8)

        if lazy_infer:
            self.proj = nn.LazyConv3d(channels[0], kernel_size=1, bias=False)
        else:
            self.proj = nn.Conv3d(embed_dim, channels[0], kernel_size=1, bias=False)

        self.up1 = UpOnly3D(channels[0], channels[1], norm=norm)  # 8→16
        self.up2 = UpOnly3D(channels[1], channels[2], norm=norm)  # 16→32
        self.up3 = UpOnly3D(channels[2], channels[3], norm=norm)  # 32→64
        self.up4 = UpOnly3D(channels[3], channels[4], norm=norm)  # 64→128
        self.up5 = UpOnly3D(channels[4], channels[5], norm=norm)  # 64→128
        self.out_head = nn.Conv3d(channels[5], out_channels, kernel_size=1)
        self.swap_ratio = 0.3
        self.mask_ratio = mask_ratio

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

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

    def forward(
        self,
        last_out: torch.Tensor
    ) -> torch.Tensor:
        
        # last: B, 212, 768
        B_, C_, W, H, D = last_out.shape
        last_out = last_out.reshape(B_,-1,C_)
        B, T, C = last_out.shape
        
        last_out = self.swap_tokens_same_idx(last_out,self.swap_ratio)
        
        # --- token masking ---
        x_masked, mask, ids_restore = random_masking(last_out, self.mask_ratio)
        
        # # 补回 mask token
        mask_tokens = self.mask_token.repeat(B, T - x_masked.shape[1], 1)
        x_ = torch.cat([x_masked, mask_tokens], dim=1)  # (B, T, C)
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, C))
        x = x.reshape(B_, C_, W, H, D)
        # --- 解码 ---
        x = self.proj(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        pred = self.out_head(x)  # (B, out_channels, 128,128,128)


        return pred, mask
