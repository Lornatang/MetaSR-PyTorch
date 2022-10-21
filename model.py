# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import math
from typing import Any

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F_torch

__all__ = [
    "MetaSR_RDN",
    "metasr_rdn",
]


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, growth_channels: int) -> None:
        super(_ResidualBlock, self).__init__()
        self.rb = nn.Sequential(
            nn.Conv2d(channels, growth_channels, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.rb(x)
        out = torch.cat([identity, out], 1)

        return out


class _ResidualDenseBlock(nn.Module):
    def __init__(self, channels: int, growth_channels: int, layers: int) -> None:
        super(_ResidualDenseBlock, self).__init__()
        rdb = []
        for index in range(layers):
            rdb.append(_ResidualBlock(channels + index * growth_channels, growth_channels))
        self.rdb = nn.Sequential(*rdb)

        # Local Feature Fusion layer
        self.local_feature_fusion = nn.Conv2d(channels + layers * growth_channels, channels, (1, 1), (1, 1), (0, 0))

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.rdb(x)
        out = self.local_feature_fusion(out)

        out = torch.add(out, identity)

        return out


class _PosToWeight(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(_PosToWeight, self).__init__()
        self.pos_to_weight = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(True),
            nn.Linear(256, 3 * 3 * in_channels * out_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.pos_to_weight(x)

        return out


def repeat(x: Tensor, upscale_factor: int) -> Tensor:
    batch_size, channels, height, width = x.size()
    out = x.view(batch_size, channels, height, 1, width, 1)

    upscale_factor = math.ceil(upscale_factor)
    out = torch.cat([out] * upscale_factor, 3)
    out = torch.cat([out] * upscale_factor, 5).permute(0, 3, 5, 1, 2, 4)

    out = out.contiguous().view(-1, channels, height, width)

    return out


class MetaSR_RDN(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            channels: int = 64,
            num_rdb: int = 16,
            num_rb: int = 8,
            growth_channels: int = 64,
    ) -> None:
        super(MetaSR_RDN, self).__init__()
        self.num_rdb = num_rdb

        # First layer
        self.conv1 = nn.Conv2d(in_channels, channels, (3, 3), (1, 1), (1, 1))

        # Second layer
        self.conv2 = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1))

        # Residual Dense Blocks
        trunk = []
        for _ in range(num_rdb):
            trunk.append(_ResidualDenseBlock(channels, growth_channels, num_rb))
        self.trunk = nn.Sequential(*trunk)

        # Global Feature Fusion
        self.global_feature_fusion = nn.Sequential(
            nn.Conv2d(int(num_rdb * channels), channels, (1, 1), (1, 1), (0, 0)),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
        )

        # Position to weight
        self.pos_to_weight = _PosToWeight(channels, out_channels)

    def forward(self, x: Tensor, pos_matrix: Tensor, upscale_factor: int) -> Tensor:
        return self._forward_impl(x, pos_matrix, upscale_factor)

    # Support torch.script function.
    def _forward_impl(self, x: Tensor, pos_matrix: Tensor, upscale_factor: int) -> Tensor:
        out1 = self.conv1(x)
        out = self.conv2(out1)

        outs = []
        for i in range(self.num_rdb):
            out = self.trunk[i](out)
            outs.append(out)

        out = torch.cat(outs, 1)
        out = self.global_feature_fusion(out)
        out = torch.add(out1, out)

        pos_matrix = pos_matrix.view(pos_matrix.size(1), -1)
        local_weight = self.pos_to_weight(pos_matrix)

        repeat_out = repeat(out, upscale_factor)
        cols = F_torch.unfold(repeat_out, (3, 3), padding=1)

        upscale_factor = math.ceil(upscale_factor)
        cols = cols.contiguous().view(cols.size(0) // (upscale_factor ** 2),
                                      upscale_factor ** 2,
                                      cols.size(1),
                                      cols.size(2), 1).permute(0, 1, 3, 4, 2).contiguous()

        local_weight = local_weight.contiguous().view(out.size(2),
                                                      upscale_factor,
                                                      out.size(3),
                                                      upscale_factor,
                                                      -1,
                                                      3).permute(1, 3, 0, 2, 4, 5).contiguous()
        local_weight = local_weight.contiguous().view(upscale_factor ** 2,
                                                      out.size(2) * out.size(3),
                                                      -1,
                                                      3)

        outs = torch.matmul(cols, local_weight).permute(0, 1, 4, 2, 3)
        outs = outs.contiguous().view(out.size(0),
                                      upscale_factor,
                                      upscale_factor,
                                      3,
                                      out.size(2),
                                      out.size(3)).permute(0, 3, 4, 1, 5, 2)
        outs = outs.contiguous().view(out.size(0),
                                      3,
                                      upscale_factor * out.size(2),
                                      upscale_factor * out.size(3))

        outs = outs.clamp_(0.0, 1.0)

        return outs


def metasr_rdn(**kwargs: Any) -> MetaSR_RDN:
    model = MetaSR_RDN(num_rdb=16, num_rb=8, growth_channels=64, **kwargs)

    return model
