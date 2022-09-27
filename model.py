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
from torch import nn
from torch.nn import functional as F
from torch import Tensor

__all__ = [
    "MetaRDN",
    "meta_rdn",
]


class _ResidualDenseConvBlock(nn.Module):
    def __init__(self, channels: int, growth_channels: int) -> None:
        super(_ResidualDenseConvBlock, self).__init__()
        self.rdb_conv = nn.Sequential(
            nn.Conv2d(channels, growth_channels, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.rdb_conv(x)
        out = torch.cat([identity, out], 1)

        return out


class _ResidualDenseBlock(nn.Module):
    def __init__(self, channels: int, growth_channels: int, layers: int) -> None:
        super(_ResidualDenseBlock, self).__init__()
        rdb = []
        for index in range(layers):
            rdb.append(_ResidualDenseConvBlock(channels + index * growth_channels, growth_channels))
        self.rdb = nn.Sequential(*rdb)

        # Local Feature Fusion layer
        self.local_feature_fusion = nn.Conv2d(channels + layers * growth_channels, growth_channels, (1, 1), (1, 1),
                                              (0, 0))

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


class MetaRDN(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            channels: int = 64,
            growth_channels: int = 64,
            conv_layers: int = 8,
            num_blocks: int = 16,
    ) -> None:
        super(MetaRDN, self).__init__()
        # First layer
        self.conv1 = nn.Conv2d(in_channels, channels, (3, 3), (1, 1), (1, 1))

        # Second layer
        self.conv2 = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1))

        # Residual Dense Blocks
        trunk = []
        for _ in range(num_blocks):
            trunk.append(_ResidualDenseBlock(channels, growth_channels, conv_layers))
        self.trunk = nn.Sequential(*trunk)

        # Global Feature Fusion
        self.global_feature_fusion = nn.Sequential(
            nn.Conv2d(1024, channels, (1, 1), (1, 1), (0, 0)),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
        )

        # Position to weight
        self.pos_to_weight = _PosToWeight(channels, out_channels)

        # Initialize all layer
        self._initialize_weights()

    def forward(self, x: Tensor, pos_matrix: Tensor, upscale_factor: int) -> Tensor:
        return self._forward_impl(x, pos_matrix, upscale_factor)

    # Support torch.script function.
    def _forward_impl(self, x: Tensor, pos_matrix: Tensor, upscale_factor: int) -> Tensor:
        out1 = self.conv1(x)
        out = self.conv2(out1)

        trunks_out = []
        for index in range(16):
            out = self.trunk[index](out)
            trunks_out.append(out)

        out = torch.cat(trunks_out, 1)
        out = self.global_feature_fusion(out)
        out = torch.add(out, out1)

        pos_matrix = pos_matrix.view(pos_matrix.size(1), -1)
        local_weight = self.pos_to_weight(pos_matrix)

        repeat_out = repeat(out, upscale_factor)
        cols = F.unfold(repeat_out, 3, padding=1)

        upscale_factor = math.ceil(upscale_factor)
        cols = cols.contiguous().view(cols.size(0) // (upscale_factor ** 2),
                                      upscale_factor ** 2,
                                      cols.size(1),
                                      cols.size(2), 1).permute(0, 1, 3, 4, 2).contiguous()

        local_weight = local_weight.contiguous().view(out.size(2), upscale_factor, out.size(3), upscale_factor, -1,
                                                      3).permute(1, 3, 0, 2, 4, 5).contiguous()
        local_weight = local_weight.contiguous().view(upscale_factor ** 2, out.size(2) * out.size(3), -1, 3)

        outs = torch.matmul(cols, local_weight).permute(0, 1, 4, 2, 3)
        outs = outs.contiguous().view(out.size(0),
                                      upscale_factor,
                                      upscale_factor,
                                      3,
                                      out.size(2),
                                      out.size(3)).permute(0, 3, 4, 1, 5, 2)
        outs = outs.contiguous().view(out.size(0), 3, upscale_factor * out.size(2), upscale_factor * out.size(3))

        return outs

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)


def meta_rdn(**kwargs: Any) -> MetaRDN:
    model = MetaRDN(**kwargs)

    return model
