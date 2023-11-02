# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np
from mmcv.cnn import ConvModule, build_conv_layer
from mmengine.model import BaseModule
from torch import nn as nn

from mmdet3d.registry import MODELS


def fill_up_weights(up):
    """Simulated bilinear upsampling kernel.

    Args:
        up (nn.Module): ConvTranspose2d module.
    """
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class IDAUpsample(BaseModule):
    """Iterative Deep Aggregation (IDA) Upsampling module to upsample features
    of different scales to a similar scale.

    Args:
        out_channels (int): Number of output channels for DeformConv.
        in_channels (List[int]): List of input channels of multi-scale
            feature maps.
        kernel_sizes (List[int]): List of size of the convolving
            kernel of different scales.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        use_dcn (bool, optional): If True, use DCNv2. Default: True.
    """

    def __init__(
        self,
        out_channels,
        in_channels,
        kernel_sizes,
        norm_cfg=None,
        use_dcn=True,
        init_cfg=None,
    ):
        super(IDAUpsample, self).__init__(init_cfg)
        self.use_dcn = use_dcn
        self.projs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.nodes = nn.ModuleList()

        for i in range(1, len(in_channels)):
            in_channel = in_channels[i]
            up_kernel_size = int(kernel_sizes[i])
            proj = ConvModule(
                in_channel,
                out_channels,
                3,
                padding=1,
                bias=True,
                conv_cfg=dict(type='DCNv2') if self.use_dcn else None,
                norm_cfg=norm_cfg)
            node = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                bias=True,
                conv_cfg=dict(type='DCNv2') if self.use_dcn else None,
                norm_cfg=norm_cfg)
            up = build_conv_layer(
                dict(type='deconv'),
                out_channels,
                out_channels,
                up_kernel_size * 2,
                stride=up_kernel_size,
                padding=up_kernel_size // 2,
                output_padding=0,
                groups=out_channels,
                bias=False)

            self.projs.append(proj)
            self.ups.append(up)
            self.nodes.append(node)

    def forward(self, mlvl_features, start_level, end_level):
        """Forward function.

        Args:
            mlvl_features (list[torch.Tensor]): Features from multiple layers.
            start_level (int): Start layer for feature upsampling.
            end_level (int): End layer for feature upsampling.
        """
        """

        """
        for i in range(start_level, end_level - 1):
            upsample = self.ups[i - start_level]
            project = self.projs[i - start_level]
            mlvl_features[i + 1] = upsample(project(mlvl_features[i + 1]))
            node = self.nodes[i - start_level]
            mlvl_features[i + 1] = node(mlvl_features[i + 1] +
                                        mlvl_features[i])


class DLAUpsample(BaseModule):
    """Deep Layer Aggregation (DLA) Upsampling module for different scales
    feature extraction, upsampling and fusion, It consists of groups of
    IDAupsample modules.

    Args:
        start_level (int): The start layer.
        channels (List[int]): List of input channels of multi-scale
            feature maps.
        scales(List[int]): List of scale of different layers' feature.
        in_channels (NoneType, optional): List of input channels of
            different scales. Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        use_dcn (bool, optional): Whether to use dcn in IDAup module.
            Default: True.
    """

    def __init__(self,
                 start_level,
                 channels,
                 scales,
                 in_channels=None,
                 norm_cfg=None,
                 use_dcn=True,
                 init_cfg=None):
        super(DLAUpsample, self).__init__(init_cfg)
        self.start_level = start_level
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(
                self, 'ida_{}'.format(i),
                IDAUpsample(channels[j], in_channels[j:],
                            scales[j:] // scales[j], norm_cfg, use_dcn))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, mlvl_features):
        """Forward function.

        Args:
            mlvl_features(list[torch.Tensor]): Features from multi-scale
                layers.

        Returns:
            tuple[torch.Tensor]: Up-sampled features of different layers.
        """
        """
        SMOKE
        首先将level5的输出加入到outs中, 尺寸为[batch_size, 512, 12, 40]
        总共要进行三次ida, 分别记为ida0, ida1, ida2
        ida0: 
            输入是level5, 尺寸为[batch_size, 512, 12, 40]
                 level4, 尺寸为[batch_size, 256, 24, 80]
            DCNv2对卷积核的计算偏移进行学习, 除此之外还学习偏移的权重, 具体可以参考https://zhuanlan.zhihu.com/p/395200094
            将level5送入DCNv2(512->256), GN, ReLU中, 得到尺寸为[batch_size, 256, 12, 40]的张量
            将上面的张量送到转置卷积中, 得到尺寸为[batch_size, 256, 24, 80]的张量
            将上面的张量和level4相加
            送到DCNv2(256->256),  GN, ReLU中, 得到尺寸为[batch_size, 256, 24, 80]的张量, 作为新的level5
            将新的level5数据前插到outs中
        ida1:
            输入是level3, 尺寸为[batch_size, 128, 48, 160]
                 level4, 尺寸为[batch_size, 256, 24, 80]
                 level5, 尺寸为[batch_size, 256, 24, 80]
            先将level4送入DCNv2(256->128), GN, ReLU中, 得到尺寸为[batch_size, 128, 24, 80]的张量
            将上面的张量送入转置卷积中, 得到尺寸为[batch_size, 128, 48, 160]的张量
            将上面的张量和level3相加, 送入DCNv2(128->128), GN, ReLU中, 得到尺寸为[batch_size, 128, 48, 160]的张量
            作为新的level4
            
            将level5送入DCNv2(256->128), GN, ReLU中, 得到尺寸为[batch_size, 128, 24, 80]的张量
            将上面的张量送入转置卷积中, 得到尺寸为[batch_size, 128, 48, 160]的张量
            将上面的张量和level4相加, 送入DCNv2(128->128), GN, ReLU中, 得到尺寸为[batch_size, 128, 48, 160]的张量
            作为新的level5
            将新的level5数据前插到outs中
        ida2:
            输入是level2, 尺寸为[batch_size, 64, 96, 320]
                 level3, 尺寸为[batch_size, 128, 48, 160]
                 level4, 尺寸为[batch_size, 128, 48, 160]
                 level5, 尺寸为[batch_size, 128, 48, 160]
            先将level3送入DCNv2(128->64), GN, ReLU中, 得到尺寸为[batch_size, 64, 48, 160]的张量
            将上面的张量送入转置卷积中, 得到尺寸为[batch_size, 64, 96, 320]的张量
            将上面的张量和level2相加, 送入DCNv2(64->64), GN, ReLU中, 得到尺寸为[batch_size, 64, 96, 320]的张量
            作为新的level3
            
            先将level4送入DCNv2(128->64), GN, ReLU中, 得到尺寸为[batch_size, 64, 48, 160]的张量
            将上面的张量送入转置卷积中, 得到尺寸为[batch_size, 64, 96, 320]的张量
            将上面的张量和level3相加, 送入DCNv2(64->64), GN, ReLU中, 得到尺寸为[batch_size, 64, 96, 320]的张量
            作为新的level4
            
            将level5送入DCNv2(128->64), GN, ReLU中, 得到尺寸为[batch_size, 64, 48, 160]的张量
            将上面的张量送入转置卷积中, 得到尺寸为[batch_size, 64, 96, 320]的张量
            将上面的张量和level4相加, 送入DCNv2(64->64), GN, ReLU中, 得到尺寸为[batch_size, 64, 96, 320]的张量
            作为新的level5
            将新的level5数据前插到outs中
        outs中有四个元素
            0: 尺寸为[batch_size, 64, 96, 320]
            1: 尺寸为[batch_size, 128, 48, 160]
            2: 尺寸为[batch_size, 256, 24, 80]
            3: 尺寸为[batch_size, 512, 12, 40]
        """
        outs = [mlvl_features[-1]]
        for i in range(len(mlvl_features) - self.start_level - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(mlvl_features, len(mlvl_features) - i - 2, len(mlvl_features))
            outs.insert(0, mlvl_features[-1])
        return outs


@MODELS.register_module()
class DLANeck(BaseModule):
    """DLA Neck.

    Args:
        in_channels (list[int], optional): List of input channels
            of multi-scale feature map.
        start_level (int, optional): The scale level where upsampling
            starts. Default: 2.
        end_level (int, optional): The scale level where upsampling
            ends. Default: 5.
        norm_cfg (dict, optional): Config dict for normalization
            layer. Default: None.
        use_dcn (bool, optional): Whether to use dcn in IDAup module.
            Default: True.
    """

    def __init__(self,
                 in_channels=[16, 32, 64, 128, 256, 512],
                 start_level=2,
                 end_level=5,
                 norm_cfg=None,
                 use_dcn=True,
                 init_cfg=None):
        super(DLANeck, self).__init__(init_cfg)
        self.start_level = start_level
        self.end_level = end_level
        scales = [2**i for i in range(len(in_channels[self.start_level:]))]
        self.dla_up = DLAUpsample(
            start_level=self.start_level,
            channels=in_channels[self.start_level:],
            scales=scales,
            norm_cfg=norm_cfg,
            use_dcn=use_dcn)
        self.ida_up = IDAUpsample(
            in_channels[self.start_level],
            in_channels[self.start_level:self.end_level],
            [2**i for i in range(self.end_level - self.start_level)], norm_cfg,
            use_dcn)

    def forward(self, x):
        """
        SMOKE: 
        输入是一个元组, 包含多个层级的特征, 总共有6个特征
        0: [batch_size, 16, 384, 1280]
        1: [batch_size, 32, 192, 640]
        2: [batch_size, 64, 96, 320]
        3: [batch_size, 128, 48, 160]
        4: [batch_size, 256, 24, 80]
        5: [batch_size, 512, 12, 40]
        """
        #* 将元组转化成列表
        mlvl_features = [x[i] for i in range(len(x))]
        """
        mlvl_features中有四个元素
            0: 尺寸为[batch_size, 64, 96, 320]
            1: 尺寸为[batch_size, 128, 48, 160]
            2: 尺寸为[batch_size, 256, 24, 80]
            3: 尺寸为[batch_size, 512, 12, 40]
        """
        mlvl_features = self.dla_up(mlvl_features)
        outs = []
        """
        outs中有mlvl_features中的前三个元素
            0: 尺寸为[batch_size, 64, 96, 320]
            1: 尺寸为[batch_size, 128, 48, 160]
            2: 尺寸为[batch_size, 256, 24, 80]
        """
        for i in range(self.end_level - self.start_level):
            outs.append(mlvl_features[i].clone())
            
        """
        SMOKE
        输入为三个元素
            f0: 尺寸为[batch_size, 64, 96, 320]
            f1: 尺寸为[batch_size, 128, 48, 160]
            f2: 尺寸为[batch_size, 256, 24, 80]
        先将f1送入DCNv2(128->64), GN, ReLU中, 得到尺寸为[batch_size, 64, 48, 160]的张量
        将上面的张量送入转置卷积中, 得到尺寸为[batch_size, 64, 96, 320]的张量
        将上面的张量和f0相加, 送入DCNv2(64->64), GN, ReLU中, 得到尺寸为[batch_size, 64, 96, 320]的张量
        作为新的f1
        
        将f2送入DCNv2(256->64), GN, ReLU中, 得到尺寸为[batch_size, 64, 24, 80]的张量
        将上面的张量送入转置卷积中, 得到尺寸为[batch_size, 64, 96, 320]的张量
        将上面的张量和f1相加, 送入DCNv2(64->64), GN, ReLU中, 得到尺寸为[batch_size, 64, 96, 320]的张量
        作为新的f2
        最后返回[f2], 尺寸为[batch_size, 64, 96, 320]
        """
        self.ida_up(outs, 0, len(outs))
        return [outs[-1]]

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                # In order to be consistent with the source code,
                # reset the ConvTranspose2d initialization parameters
                m.reset_parameters()
                # Simulated bilinear upsampling kernel
                fill_up_weights(m)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                # In order to be consistent with the source code,
                # reset the Conv2d initialization parameters
                m.reset_parameters()
