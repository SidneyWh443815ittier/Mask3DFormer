# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn
import numpy as np


# from utils.pc_util import shift_scale_points
# 将 3D 空间中的坐标转换为位置嵌入（Position Embeddings）

def shift_scale_points(pred_xyz, src_range, dst_range=None):
    # 从范围 src_range 映射到范围 dst_range
    """
    pred_xyz: B x N x 3
    src_range: [[B x 3], [B x 3]] - min and max XYZ coords
    dst_range: [[B x 3], [B x 3]] - min and max XYZ coords
    """
    if dst_range is None:
        dst_range = [
            torch.zeros(
                (src_range[0].shape[0], 3), device=src_range[0].device
            ),
            torch.ones((src_range[0].shape[0], 3), device=src_range[0].device),
        ]

    if pred_xyz.ndim == 4:
        src_range = [x[:, None] for x in src_range]
        dst_range = [x[:, None] for x in dst_range]

    assert src_range[0].shape[0] == pred_xyz.shape[0]
    assert dst_range[0].shape[0] == pred_xyz.shape[0]
    assert src_range[0].shape[-1] == pred_xyz.shape[-1]
    assert src_range[0].shape == src_range[1].shape
    assert dst_range[0].shape == dst_range[1].shape
    assert src_range[0].shape == dst_range[1].shape

    src_diff = src_range[1][:, None, :] - src_range[0][:, None, :]
    dst_diff = dst_range[1][:, None, :] - dst_range[0][:, None, :]
    prop_xyz = (
                       ((pred_xyz - src_range[0][:, None, :]) * dst_diff) / src_diff
               ) + dst_range[0][:, None, :]
    return prop_xyz


class PositionEmbeddingCoordsSine(nn.Module):
    def __init__(
            self,
            # temperature: 控制正弦编码中的频率缩放。值越高，嵌入频率越低。
            temperature=10000,
            normalize=False,
            # 用于调整正弦嵌入的缩放因子
            scale=None,
            # pos_type: 位置嵌入的类型，可以是 'sine' 或 'fourier'
            pos_type="fourier",
            # 生成的嵌入维度
            d_pos=None,
            # 输入坐标的维度
            d_in=3,
            # 用于傅里叶嵌入中高斯矩阵的缩放因子，影响嵌入空间的尺度
            gauss_scale=1.0,
    ):
        super().__init__()
        self.d_pos = d_pos
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        assert pos_type in ["sine", "fourier"]
        self.pos_type = pos_type
        self.scale = scale
        if pos_type == "fourier":
            assert d_pos is not None
            assert d_pos % 2 == 0
            # define a gaussian matrix input_ch -> output_ch
            B = torch.empty((d_in, d_pos // 2)).normal_()
            B *= gauss_scale
            self.register_buffer("gauss_B", B)
            self.d_pos = d_pos

    # 正弦嵌入（Sine Embeddings）
    def get_sine_embeddings(self, xyz, num_channels, input_range):
        num_channels = self.d_pos
        # clone coords so that shift/scale operations do not affect original tensor
        orig_xyz = xyz
        xyz = orig_xyz.clone()

        ncoords = xyz.shape[1]
        if self.normalize:
            # 归一化到指定范围
            xyz = shift_scale_points(xyz, src_range=input_range)
        # ndim 计算每个坐标维度所需的编码维数。xyz.shape[2] 是 3，代表 x, y, z 三个维度。
        ndim = num_channels // xyz.shape[2]
        if ndim % 2 != 0:
            ndim -= 1
        # automatically handle remainder by assiging it to the first dim
        rems = num_channels - (ndim * xyz.shape[2])

        assert (
                ndim % 2 == 0
        ), f"Cannot handle odd sized ndim={ndim} where num_channels={num_channels} and xyz={xyz.shape}"

        # 构建位置编码
        final_embeds = []
        prev_dim = 0

        for d in range(xyz.shape[2]):
            cdim = ndim
            if rems > 0:
                # add remainder in increments of two to maintain even size
                cdim += 2
                rems -= 2

            if cdim != prev_dim:
                dim_t = torch.arange(
                    cdim, dtype=torch.float32, device=xyz.device
                )
                dim_t = self.temperature ** (2 * (dim_t // 2) / cdim)

            # create batch x cdim x nccords embedding
            # raw_pos 提取 xyz 的第 d 维坐标（即 x, y, z 中的一个）。
            raw_pos = xyz[:, :, d]
            if self.scale:
                raw_pos *= self.scale
            # 计算 pos，即将 raw_pos 除以 dim_t，并将其扩展维度以便后续的正弦和余弦计算。
            pos = raw_pos[:, :, None] / dim_t
            # 创建正弦位置编码，通过对位置的正弦和余弦变换生成编码
            pos = torch.stack(
                (pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3
            ).flatten(2)
            final_embeds.append(pos)
            prev_dim = cdim
        # 将 final_embeds 中所有维度的编码按第三维度（n_coords）拼接，
        # 最后得到形状为 (B, num_channels, N) 的张量，表示每个点的最终位置编码。
        final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
        return final_embeds

    def get_fourier_embeddings(self, xyz, num_channels=None, input_range=None):
        # Follows - https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
        # Fourier嵌入
        if num_channels is None:
            num_channels = self.gauss_B.shape[1] * 2

        # bsize 和 npoints 分别表示批量大小和点的数量。
        bsize, npoints = xyz.shape[0], xyz.shape[1]
        assert num_channels > 0 and num_channels % 2 == 0
        # d_in 表示 xyz 的最后一维度的大小（这里为 3，因为是 3D 坐标）。
        d_in, max_d_out = self.gauss_B.shape[0], self.gauss_B.shape[1]
        # d_out 是一半的 num_channels，
        d_out = num_channels // 2
        # 检查 d_out 是否小于或等于 max_d_out，保证生成的特征维度不会超过预定义的最大值。
        assert d_out <= max_d_out
        assert d_in == xyz.shape[-1]

        # clone coords so that shift/scale operations do not affect original tensor
        orig_xyz = xyz
        xyz = orig_xyz.clone()

        ncoords = xyz.shape[1]
        if self.normalize:
            xyz = shift_scale_points(xyz, src_range=input_range)
        # 将坐标转换到正弦和余弦空间
        xyz *= 2 * np.pi
        # self.gauss_B 是一个高斯随机矩阵，用于将 3D 坐标 xyz 投影到更高的特征空间。
        xyz_proj = torch.mm(xyz.view(-1, d_in), self.gauss_B[:, :d_out]).view(
            bsize, npoints, d_out
        )
        # 对投影结果分别应用 sin 和 cos 函数，以生成互补的正弦和余弦特征
        final_embeds = [xyz_proj.sin(), xyz_proj.cos()]

        # return batch x d_pos x npoints embedding
        final_embeds = torch.cat(final_embeds, dim=2).permute(0, 2, 1)
        return final_embeds

    def forward(self, xyz, num_channels=None, input_range=None):
        assert isinstance(xyz, torch.Tensor)
        assert xyz.ndim == 3
        # xyz is batch x npoints x 3
        if self.pos_type == "sine":
            with torch.no_grad():
                out = self.get_sine_embeddings(xyz, num_channels, input_range)
        elif self.pos_type == "fourier":
            with torch.no_grad():
                out = self.get_fourier_embeddings(
                    xyz, num_channels, input_range
                )
        else:
            raise ValueError(f"Unknown {self.pos_type}")

        return out

    def extra_repr(self):
        st = f"type={self.pos_type}, scale={self.scale}, normalize={self.normalize}"
        if hasattr(self, "gauss_B"):
            st += f", gaussB={self.gauss_B.shape}, gaussBsum={self.gauss_B.sum().item()}"
        return st
