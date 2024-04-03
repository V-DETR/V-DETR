import torch
import torch.nn as nn

import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
    
    
class MinkResNet(nn.Module):
    r"""Minkowski ResNet backbone. See `4D Spatio-Temporal ConvNets
    <https://arxiv.org/abs/1904.08755>`_ for more details.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (ont): Number of input channels, 3 for RGB.
        num_stages (int, optional): Resnet stages. Default: 4.
        pool (bool, optional): Add max pooling after first conv if True.
            Default: True.
    """
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self, depth, in_channels, inplanes=64, num_stages=4, stem_bn=False):
        super(MinkResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        assert 4 >= num_stages >= 1
        block, stage_blocks = self.arch_settings[depth]
        stage_blocks = stage_blocks[:num_stages]
        self.num_stages = num_stages

        self.inplanes = inplanes
        
        self.conv1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=3, stride=2, dimension=3)
        # May be BatchNorm is better, but we follow original implementation.
        self.norm1 = ME.MinkowskiBatchNorm(self.inplanes) if stem_bn else ME.MinkowskiInstanceNorm(self.inplanes)
        self.relu = ME.MinkowskiReLU(inplace=False)
        
        for i, num_blocks in enumerate(stage_blocks):
            setattr(
                self, f'layer{i + 1}',
                self._make_layer(block, inplanes * 2**i, stage_blocks[i], stride=2))
            
        self.init_weights()
            
    def init_weights(self):
        print('random init backbone')
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(
                    m.kernel, mode='fan_out', nonlinearity='relu')

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(self, block, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    dimension=3),
                ME.MinkowskiBatchNorm(planes * block.expansion))
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                dimension=3))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, dimension=3))
        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass of ResNet.

        Args:
            x (ME.SparseTensor): Input sparse tensor.

        Returns:
            list[ME.SparseTensor]: Output sparse tensors.
        """
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        outs = []
        for i in range(self.num_stages):
            x = getattr(self, f'layer{i + 1}')(x)
            outs.append(x)
        return outs