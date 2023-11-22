# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

from mmdet.models.detectors.single_stage import SingleStageDetector
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import OptInstanceList


@MODELS.register_module()
class SingleStageMono3DDetector(SingleStageDetector):
    """Base class for monocular 3D single-stage detectors.

    Monocular 3D single-stage detectors directly and densely predict bounding
    boxes on the output features of the backbone+neck.
    """

    def add_pred_to_datasample(
        self,
        data_samples: SampleList,
        data_instances_3d: OptInstanceList = None,
        data_instances_2d: OptInstanceList = None,
    ) -> SampleList:
        """Convert results list to `Det3DDataSample`.

        Args:
            data_samples (list[:obj:`Det3DDataSample`]): The input data.
            data_instances_3d (list[:obj:`InstanceData`], optional): 3D
                Detection results of each image. Defaults to None.
            data_instances_2d (list[:obj:`InstanceData`], optional): 2D
                Detection results of each image. Defaults to None.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input. Each Det3DDataSample usually contains
            'pred_instances_3d'. And the ``pred_instances_3d`` normally
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels_3d (Tensor): Labels of 3D bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (Tensor): Contains a tensor with shape
              (num_instances, C) where C >=7.

            When there are 2D prediction in some models, it should
            contains  `pred_instances`, And the ``pred_instances`` normally
            contains following keys.

            - scores (Tensor): Classification scores of image, has a shape
              (num_instance, )
            - labels (Tensor): Predict Labels of 2D bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Contains a tensor with shape
              (num_instances, 4).
        """

        assert (data_instances_2d is not None) or \
               (data_instances_3d is not None),\
               'please pass at least one type of data_samples'

        if data_instances_2d is None:
            data_instances_2d = [
                InstanceData() for _ in range(len(data_instances_3d))
            ]
        if data_instances_3d is None:
            data_instances_3d = [
                InstanceData() for _ in range(len(data_instances_2d))
            ]

        for i, data_sample in enumerate(data_samples):
            data_sample.pred_instances_3d = data_instances_3d[i]
            data_sample.pred_instances = data_instances_2d[i]
        return data_samples

    def extract_feat(self, batch_inputs_dict: dict) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs_dict (dict): Contains 'img' key
                with image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        batch_imgs = batch_inputs_dict['imgs']
        #* 只将batch_inputs_dict['imgs']送入backbone里面
        """
        #! FCOS3D Backbone:
            batch_imgs大小为[batch_size, 3, 928, 1600]
            backbone用的是ResNet101
            首先经过一个7*7, 步长为2的卷积(3->64), BatchNorm2d, ReLU
            使用3*3的步长为2的Maxpooling进行下采样, 输出的张量尺寸为[batch_size, 64, 232, 400]

            reslayer0:
                BottleNeck0:
                    输入是x
                    x经过一个1*1的步长为1的卷积(64->64), 一个BatchNorm2d, ReLU, 得到out
                    out经过一个3*3的步长为1的卷积(64->64), 一个BatchNorm2d, ReLU, 更新out
                    out经过一个1*1的步长为1的卷积(64->256), 一个BatchNorm2d, 更新out
                    
                    x经过一个1*1步长为1的卷积(64->256), 一个Batchnorm2d, 得到indentity
                    将identity和out相加, 送入ReLU中, 得到最终的out
                BottleNeck1:
                    输入是上一层的输出, 记为x
                    x经过一个1*1的步长为1的卷积(256->64), 一个Batchnorm2d, 一个ReLU, 得到out
                    out经过一个3*3的步长为1的卷积(64->64), 一个Batchnorm2d, ReLU, 更新out
                    out经过一个3*3的步长为1的卷积(64->256), 一个Batchnorm2d, 更新out
                    将out+x送入ReLU中, 得到最终的out
                BottleNeck2:
                    同BottleNeck1
                最后的输出是BottleNeck2的输出, 尺寸为[batch_size, 256, 232, 400]
            reslayer1:
                BottleNeck0:
                    输入是x, 尺寸为[batch_size, 256, 232, 400]
                    x经过一个1*1的步长为2的卷积(256->128), 一个BatchNorm2d, ReLU, 得到out, 尺寸为[batch_size, 512, 116, 200]
                    out经过一个3*3的步长为1的卷积(128->128), 一个BatchNorm2d, ReLU, 更新out
                    out经过一个1*1的步长为1的卷积(128->512), 一个BatchNorm2d, 更新out
                    
                    x经过一个1*1步长为2的卷积(256->512), 一个Batchnorm2d, 得到indentity, 尺寸为[batch_size, 512, 116, 200]
                    将identity和out相加, 送入ReLU中, 得到最终的out, 尺寸为[batch_size, 512, 116, 200]
                BottleNeck1:
                    输入是上一层的输出, 尺寸为[batch_size, 512, 116, 200], 记为x
                    x经过一个1*1的步长为1的卷积(512->128), 一个Batchnorm2d, 一个ReLU, 得到out
                    out经过一个3*3的步长为1的卷积(128->128), 一个Batchnorm2d, 一个ReLU, 更新out
                    out经过一个1*1的步长为1的卷积(128->512), 一个Batchnorm2d, 一个ReLU, 更新out
                    将out+x送入ReLU中, 得到最终的out
                BottleNeck2:
                    同BottleNeck1
                BottleNeck3:
                    同BottleNeck1
                最后的输出是BottleNeck3的输出, 尺寸为[batch_size, 512, 116, 200]
            reslayer2:
                BottleNeck0:
                    输入是x, 尺寸为[batch_size, 1024, 116, 200]
                    x经过一个1*1的步长为2的卷积(512->256), 一个BatchNorm2d, ReLU, 得到out, 尺寸为[batch_size, 256, 58, 100]
                    out经过一个3*3的步长为1的DCNv2(256->256), 一个BatchNorm2d, ReLU, 更新out
                    out经过一个1*1的步长为1的卷积(256->1024), 一个BatchNorm2d, 更新out
                    
                    x经过一个1*1步长为2的卷积(512->1024), 一个Batchnorm2d, 得到indentity, 尺寸为[batch_size, 1024, 58, 100]
                    将identity和out相加, 送入ReLU中, 得到最终的out, 尺寸为[batch_size, 1024, 58, 100]
                BottleNeck1:
                    输入是上一层的输出, 尺寸为[batch_size, 1024, 58, 100]
                    x经过一个1*1的步长为1的卷积(1024->256), 一个BatchNorm2d, ReLU, 得到out
                    out经过一个3*3的步长为1的DCNv2(256->256), 一个BatchNorm2d, ReLU, 更新out
                    out经过一个1*1的步长为1的卷积(256->1024), 一个BatchNorm2d, 更新out
                    将out+x送入ReLU中, 得到最终的out
                BottleNeck2~BottleNeck22:
                    同BottleNeck1
            resplayer3:
                BottleNeck0:
                    输入是x, 尺寸为[batch_size, 1024, 58, 100]
                    x经过一个1*1的步长为2的卷积(1024->512), 一个BatchNorm2d, ReLU, 得到out, 尺寸为[batch_size, 512, 29, 50]
                    out经过一个3*3的步长为1的DCNv2(512->512), 一个BatchNorm2d, ReLU, 更新out
                    out经过一个1*1的步长为1的卷积(512->2048), 一个BatchNorm2d, 更新out
                    
                    x经过一个1*1步长为2的卷积(1024->2048), 一个Batchnorm2d, 得到indentity, 尺寸为[batch_size, 2048, 29, 50]
                    将identity和out相加, 送入ReLU中, 得到最终的out, 尺寸为[batch_size, 2048, 29, 50]
                BottleNeck1:
                    输入是上一层的输出, 尺寸为[batch_size, 2048, 29, 50], 记为x
                    x经过一个1*1的步长为1的卷积(2048->512), 一个Batchnorm2d, 一个ReLU, 得到out
                    out经过一个3*3的步长为1的卷积(512->512), 一个Batchnorm2d, 一个ReLU, 更新out
                    out经过一个1*1的步长为1的卷积(512->2048), 一个Batchnorm2d, 一个ReLU, 更新out
                    将out+x送入ReLU中, 得到最终的out
                BottleNeck2:
                    同BottleNeck1
            
            最后的输出是一个程度为4的元素, 分别存放reslayer0~reslayer3的输出
            尺寸分别为[batch_size, 256, 232, 400]
                    [batch_size, 512, 116, 200]
                    [batch_size, 1024, 58, 100]
                    [batch_size, 2048, 29, 50]
        """
        """
        #! FCOS Neck: 输入是backbone的输出, 尺寸如下:
        尺寸分别为[batch_size, 256, 232, 400], 记为inputs0
                [batch_size, 512, 116, 200], 记为inputs1
                [batch_size, 1024, 58, 100], 记为inputs2
                [batch_size, 2048, 29, 50], 记为inputs3
        将inputs1送入1*1步长为1的卷积(512->256)中, 记为x1
        将inputs2送入1*1步长为1的卷积(1024->256)中, 记为x2
        将inputs3送入1*1步长为1的卷积(2048->256)中, 记为x3
        将x3通过插值变成x2的形状, 再和x2相加变成新的x2
        将新的x2通过插值变成x1的形状, 再和x1相加变成新的x1
        
        用三个权值不共享的3*3步长为1的卷积处理x1, x2, x3
        
        将x3的副本用3*3步长为2的卷积(256->256)处理, 得到x4, 尺寸为[batch_size, 256, 15, 25]
        将ReLU, 3*3步长为2的卷积(256->256)作用于x4, 得到x5, 尺寸为[batch_size, 256, 8, 13]
        最后的输出是一个元组, 元组中的元素尺寸如下
            [batch_size, 256, 116, 200]
            [batch_size, 256, 58, 100]
            [batch_size, 256, 29, 50]
            [batch_size, 256, 15, 25]
            [batch_size, 256, 8, 13]
        """
        """
        #! PGD backbone KITTI:
            backbone是ResNet101, 输入的尺寸为[batch_size, 3, 384, 1248]
            先经过一个7*7的步长为2的卷积(3->64), BatchNorm2d, ReLU, 输出尺寸为[batch_size, 64, 192, 624]
            再经过一个3*3的步长为2的maxpooling, 尺寸为[batch_size, 64, 96, 312]
            总共有四层Reslayer:
            Reslayer0:
                BottleNeck0:
                    输入是x, 尺寸为[batch_size, 64, 96, 312]
                    x经过一个1*1的步长为1的卷积(64->64), 一个BatchNorm2d, 一个ReLU, 得到out, 尺寸为[batch_size, 64, 96, 312]
                    out经过一个3*3的步长为1的卷积(64->64), 一个BatchNorm2d, 一个ReLU, 更新out, 尺寸为[batch_size, 64, 96, 312]
                    out经过一个1*1的步长为1的卷积(64->256), 一个BatchNorm2d, 更新out, 尺寸为[batch_size, 256, 96, 312]
                    
                    x经过一个1*1的步长为1的卷积(64->256), 一个BatchNorm2d, 得到identity, 尺寸为[batch_size, 256, 96, 312]
                    将out和identity相加, 送入Relu中, 得到最终的out, 尺寸为[batch_size, 256, 96, 312]
                BottleNeck1:
                    输入是x, 尺寸为[batch_size, 256, 96, 312]
                    x经过一个1*1的步长为1的卷积(256->64), 一个BatchNorm2d, 一个ReLU, 得到out, 尺寸为[batch_size, 64, 96, 312]
                    out经过一个3*3的步长为1的卷积(64->64), 一个BatchNorm2d, 一个ReLU, 更新out, 尺寸为[batch_size, 64, 96, 312]
                    out经过一个1*1的步长为1的卷积(64->256), 一个BatchNorm2d, 一个ReLU, 更新out, 尺寸为[batch_size, 256, 96, 312]
                    
                    out和x相加, 送入ReLU中, 得到最终的out, 尺寸为[batch_size, 256, 96, 312]
                BottleNeck2:
                    同BottleNeck1             
            Reslayer1:
                BottleNeck0:
                    输入是x, 尺寸为[batch_size, 256, 96, 312]
                    x经过一个1*1的步长为2的卷积(256->128), 一个BatchNorm2d, 一个ReLU, 得到out, 尺寸为[batch_size, 128, 48, 156]
                    out经过一个3*3的步长为1的卷积(128->128), 一个BatchNorm2d, 一个ReLU, 更新out, 尺寸为[batch_size, 128, 48, 156]
                    out经过一个1*1的步长为1的卷积(128->512), 一个BacthNorm2d, 一个ReLU, 更新out, 尺寸为[batch_size, 512, 48, 156]
                    
                    x经过一个1*1的步长为2的卷积(256->512), 一个BatchNorm2d, 得到identity, 尺寸为[batch_size, 512, 48, 156]
                    将out和identity相加, 送入ReLU中, 得到最终的out, 尺寸为[batch_size, 512, 48, 156]
                BottleNeck1:
                    输入是x, 尺寸为[batch_size, 512, 48, 156]
                    x经过一个1*1的步长为2的卷积(512->128), 一个BatchNorm2d, 一个ReLU, 得到out, 尺寸为[batch_size, 128, 48, 156]
                    out经过一个3*3的步长为1的卷积(128->128), 一个BatchNorm2d, 一个ReLU, 更新out, 尺寸为[batch_size, 128, 48, 156]
                    out经过一个1*1的步长为1的卷积(128->512), 一个BacthNorm2d, 一个ReLU, 更新out, 尺寸为[batch_size, 512, 48, 156]
                    
                    将out和x相加, 送入ReLU中, 得到最终的out, 尺寸为[batch_size, 512, 48, 156]
                BottleNeck2~BottleNeck3:
                    同BottleNeck1
            Reslayer2:
                BottleNeck0:
                    输入是x, 尺寸为[batch_size, 512, 48, 156]
                    x经过一个1*1的步长为2的卷积(512->256), 一个BatchNorm2d, 一个ReLU, 得到out, 尺寸为[batch_size, 256, 24, 78]
                    out经过一个3*3的步长为1的卷积(256->256), 一个BatchNorm2d, 一个ReLU, 更新out, 尺寸为[batch_size, 256, 24, 78]
                    out经过一个1*1的步长为1的卷积(256->1024), 一个BatchNorm2d, 一个ReLU, 更新out, 尺寸为[batch_size, 1024, 24, 78]
                    
                    x经过一个1*1的步长为2的卷积(512->1024), 一个BatchNorm2d, 得到identity, 尺寸为[batch_size, 1024, 24, 78]
                    将out和identity相加, 送入ReLU中, 得到最终的out, 尺寸为[batch_size, 1024, 24, 78]
                BottleNeck1:
                    输入是x, 尺寸为[batch_size, 1024, 24, 78]
                    x经过一个1*1的步长为2的卷积(1024->256), 一个BatchNorm2d, 一个ReLU, 得到out, 尺寸为[batch_size, 256, 24, 78]
                    out经过一个3*3的步长为1的卷积(256->256), 一个BatchNorm2d, 一个ReLU, 更新out, 尺寸为[batch_size, 256, 24, 78]
                    out经过一个1*1的步长为1的卷积(256->1024), 一个BacthNorm2d, 一个ReLU, 更新out, 尺寸为[batch_size, 1024, 24, 78]
                    
                    将out和x相加, 送入ReLU中, 得到最终的out, 尺寸为[batch_size, 1024, 24, 78]
                BottleNeck2~BottleNeck22:
                    同BottleNeck1
            Reslayer3:
                BottleNeck0:
                    输入是x, 尺寸为[batch_size, 1024, 24, 78]
                    x经过一个1*1的步长为2的卷积(1024->512), 一个BatchNorm2d, 一个ReLU, 得到out, 尺寸为[batch_size, 512, 12, 39]
                    out经过一个3*3的步长为1的卷积(512->512), 一个BatchNorm2d, 一个ReLU, 更新out, 尺寸为[batch_size, 512, 12, 39]
                    out经过一个1*1的步长为1的卷积(512>2048), 一个BatchNorm2d, 一个ReLU, 更新out, 尺寸为[batch_size, 2048, 12, 39]
                    
                    x经过一个1*1的步长为2的卷积(1024->2048), 一个BatchNorm2d, 得到identity, 尺寸为[batch_size, 2048, 12, 39]
                    将out和x相加, 送入ReLU中, 得到最终的out, 尺寸为[batch_size, 2048, 12, 39]
                BottleNeck1:
                    输入是x, 尺寸为[batch_size, 2048, 12, 39]
                    x经过一个1*1的步长为2的卷积(2048->512), 一个BatchNorm2d, 一个ReLU, 得到out, 尺寸为[batch_size, 512, 12, 39]
                    out经过一个3*3的步长为1的卷积(512->512), 一个BatchNorm2d, 一个ReLU, 更新out, 尺寸为[batch_size, 512, 12, 39]
                    out经过一个1*1的步长为1的卷积(512->2048), 一个BacthNorm2d, 一个ReLU, 更新out, 尺寸为[batch_size, 2048, 12, 39]
                    
                    将out和x相加, 送入ReLU中, 得到最终的out, 尺寸为[batch_size, 2048, 12, 39]
                BottleNeck2:
                    同BottleNeck1
            最后的输出是一个长度为4的元组, 元组里面的元素尺寸分别是:
                [batch_size, 256, 96, 312]
                [batch_size, 512, 48, 156]
                [batch_size, 1024, 24, 78]
                [batch_size, 2048, 12, 39]
        """
        """
        #! PGD Neck KITTI:
            Neck是FPN, 输入是一个长度为4的元组inputs, 元组里面的元素尺寸分别是:
                [batch_size, 256, 96, 312]
                [batch_size, 512, 48, 156]
                [batch_size, 1024, 24, 78]
                [batch_size, 2048, 12, 39]
            对inputs[0]使用1*1的步长为1的卷积(256->256), 记为x0, 尺寸为[batch_size, 256, 96, 312]
            对inputs[1]使用1*1的步长为1的卷积(512->256), 记为x1, 尺寸为[batch_size, 256, 48, 156]
            对inputs[2]使用1*1的步长为1的卷积(1024->256), 记为x2, 尺寸为[batch_size, 256, 24, 78]
            对inputs[3]使用1*1的步长为1的卷积(2048->256), 记为x3, 尺寸为[batch_size, 256, 12, 39]
            将x3通过插值变成x2的形状, 再和x2相加变成新的x2
            将新的x2通过插值变成x1的形状, 再和x1相加变成新的x1
            将新的x1通过插值变成x0的形状, 再和x0相加变成新的x0
            
            用四个权值不共享的3*3步长为1的卷积处理x0, x1, x2, x3
            最后的输出是一个元组, 元组中的元素尺寸如下
                [batch_size, 256, 96, 312]
                [batch_size, 256, 48, 156]
                [batch_size, 256, 24, 78]
                [batch_size, 256, 12, 39]
        """
        """
        #! PGD Backbone Nusc:
            backbone是ResNet101, 输入的尺寸是[batch_size, 3, 928, 1600]
            先经过一个7*7的步长为2的卷积(3->64), BatchNorm2d, ReLU, 输出尺寸为[batch_size, 64, 464, 800]
            再经过一个3*3的步长为2的maxpooling, 尺寸为[batch_size, 64, 232, 400]
            总共有四层Reslayer:
            Reslayer0:  这一层是不用训练, 参数冻结了
                BottleNeck0:
                    输入是x, 尺寸为[batch_size, 64, 232, 400]
                    x经过一个1*1的步长为1的卷积(64->64), 一个BatchNorm2d, 一个ReLU, 得到out, 尺寸为[batch_size, 64, 232, 400]
                    out经过一个3*3的步长为1的卷积(64->64), 一个BatchNorm2d, 一个ReLU, 更新out, 尺寸为[batch_size, 64, 232, 400]
                    out经过一个1*1的步长为1的卷积(64->256), 一个BatchNorm2d, 更新out, 尺寸为[batch_size, 256, 232, 400]
                    
                    x经过一个1*1的步长为1的卷积(64->256), 一个BatchNorm2d, 得到identity, 尺寸为[batch_size, 256, 232, 400]
                    将out和identity相加, 送入Relu中, 得到最终的out, 尺寸为[batch_size, 256, 232, 400]
                BottleNeck1:
                    输入是x, 尺寸为[batch_size, 256, 232, 400]
                    x经过一个1*1的步长为1的卷积(256->64), 一个BatchNorm2d, 一个ReLU, 得到out, 尺寸为[batch_size, 64, 232, 400]
                    out经过一个3*3的步长为1的卷积(64->64), 一个BatchNorm2d, 一个ReLU, 更新out, 尺寸为[batch_size, 64, 232, 400]
                    out经过一个1*1的步长为1的卷积(64->256), 一个BatchNorm2d, 一个ReLU, 更新out, 尺寸为[batch_size, 256, 232, 400]
                    
                    out和x相加, 送入ReLU中, 得到最终的out, 尺寸为[batch_size, 256, 232, 400]
                BottleNeck2:
                    同BottleNeck1  
            Reslayer1:
                BottleNeck0:
                    输入是x, 尺寸为[batch_size, 256, 232, 400]
                    x经过一个1*1的步长为2的卷积(256->128), 一个BatchNorm2d, 一个ReLU, 得到out, 尺寸为[batch_size, 128, 116, 200]
                    out经过一个3*3的步长为1的卷积(128->128), 一个BatchNorm2d, 一个ReLU, 更新out, 尺寸为[batch_size, 128, 116, 200]
                    out经过一个1*1的步长为1的卷积(128->512), 一个BacthNorm2d, 一个ReLU, 更新out, 尺寸为[batch_size, 512, 116, 200]
                    
                    x经过一个1*1的步长为2的卷积(256->512), 一个BatchNorm2d, 得到identity, 尺寸为[batch_size, 512, 116, 200]
                    将out和identity相加, 送入ReLU中, 得到最终的out, 尺寸为[batch_size, 512, 116, 200]
                BottleNeck1:
                    输入是x, 尺寸为[batch_size, 512, 116, 200]
                    x经过一个1*1的步长为2的卷积(512->128), 一个BatchNorm2d, 一个ReLU, 得到out, 尺寸为[batch_size, 128, 116, 200]
                    out经过一个3*3的步长为1的卷积(128->128), 一个BatchNorm2d, 一个ReLU, 更新out, 尺寸为[batch_size, 128, 116, 200]
                    out经过一个1*1的步长为1的卷积(128->512), 一个BacthNorm2d, 一个ReLU, 更新out, 尺寸为[batch_size, 512, 116, 200]
                    
                    将out和x相加, 送入ReLU中, 得到最终的out, 尺寸为[batch_size, 512, 116, 200]
                BottleNeck2~BottleNeck3:
                    同BottleNeck1
            Reslayer2:
                BottleNeck0:
                    输入是x, 尺寸为[batch_size, 512, 116, 200]
                    x经过一个1*1的步长为2的卷积(512->256), 一个BatchNorm2d, 一个ReLU, 得到out, 尺寸为[batch_size, 256, 58, 100]
                    out经过一个3*3的步长为1的卷积(256->256), 一个BatchNorm2d, 一个ReLU, 更新out, 尺寸为[batch_size, 256, 58, 100]
                    out经过一个1*1的步长为1的卷积(256->1024), 一个BatchNorm2d, 一个ReLU, 更新out, 尺寸为[batch_size, 1024, 58, 100]
                    
                    x经过一个1*1的步长为2的卷积(512->1024), 一个BatchNorm2d, 得到identity, 尺寸为[batch_size, 1024, 58, 100]
                    将out和identity相加, 送入ReLU中, 得到最终的out, 尺寸为[batch_size, 1024, 58, 100]
                BottleNeck1:
                    输入是x, 尺寸为[batch_size, 1024, 58, 100]
                    x经过一个1*1的步长为2的卷积(1024->256), 一个BatchNorm2d, 一个ReLU, 得到out, 尺寸为[batch_size, 256, 58, 100]
                    out经过一个3*3的步长为1的DCNv2(256->256), 一个BatchNorm2d, 一个ReLU, 更新out, 尺寸为[batch_size, 256, 58, 100]
                    out经过一个1*1的步长为1的卷积(256->1024), 一个BacthNorm2d, 一个ReLU, 更新out, 尺寸为[batch_size, 1024, 58, 100]
                    
                    将out和x相加, 送入ReLU中, 得到最终的out, 尺寸为[batch_size, 1024, 58, 100]
                BottleNeck2~BottleNeck22:
                    同BottleNeck1
            Reslayer3:
                BottleNeck0:
                    输入是x, 尺寸为[batch_size, 1024, 58, 100]
                    x经过一个1*1的步长为2的卷积(1024->512), 一个BatchNorm2d, 一个ReLU, 得到out, 尺寸为[batch_size, 512, 29, 50]
                    out经过一个3*3的步长为1的卷积(512->512), 一个BatchNorm2d, 一个ReLU, 更新out, 尺寸为[batch_size, 512, 29, 50]
                    out经过一个1*1的步长为1的卷积(512>2048), 一个BatchNorm2d, 一个ReLU, 更新out, 尺寸为[batch_size, 2048, 29, 50]
                    
                    x经过一个1*1的步长为2的卷积(1024->2048), 一个BatchNorm2d, 得到identity, 尺寸为[batch_size, 2048, 29, 50]
                    将out和x相加, 送入ReLU中, 得到最终的out, 尺寸为[batch_size, 2048, 29, 50]
                BottleNeck1:
                    输入是x, 尺寸为[batch_size, 2048, 29, 50]
                    x经过一个1*1的步长为2的卷积(2048->512), 一个BatchNorm2d, 一个ReLU, 得到out, 尺寸为[batch_size, 512, 29, 50]
                    out经过一个3*3的步长为1的DCNv2(512->512), 一个BatchNorm2d, 一个ReLU, 更新out, 尺寸为[batch_size, 512, 29, 50]
                    out经过一个1*1的步长为1的卷积(512->2048), 一个BacthNorm2d, 一个ReLU, 更新out, 尺寸为[batch_size, 2048, 29, 50]
                    
                    将out和x相加, 送入ReLU中, 得到最终的out, 尺寸为[batch_size, 2048, 29, 50]
                BottleNeck2:
                    同BottleNeck1
            最后的输出是一个长度为4的元组, 元组里面的元素尺寸分别是:
                [batch_size, 256, 232, 400]
                [batch_size, 512, 116, 200]
                [batch_size, 1024, 58, 100]
                [batch_size, 2048, 29, 50]
        """
        """
        #! PGD Neck Nusc:
            Neck是FPN, 输入是一个长度为4的元组inputs, 元组里面的元素尺寸分别是:
                [batch_size, 256, 232, 400]
                [batch_size, 512, 116, 200]
                [batch_size, 1024, 58, 100]
                [batch_size, 2048, 29, 50]
            对inputs[1]使用1*1的步长为1的卷积(512->256), 记为x1, 尺寸为[batch_size, 256, 116, 200]
            对inputs[2]使用1*1的步长为1的卷积(1024->256), 记为x2, 尺寸为[batch_size, 256, 58, 100]
            对inputs[3]使用1*1的步长为1的卷积(2048->256), 记为x3, 尺寸为[batch_size, 256, 29, 50]
            将x3通过插值变成x2的形状, 再和x2相加变成新的x2
            将新的x2通过插值变成x1的形状, 再和x1相加变成新的x1
            
            用四个权值不共享的3*3步长为1的卷积处理x1, x2, x3
            将x3的副本用3*3步长为2的卷积(256->256)处理, 得到x4, 尺寸为[batch_size, 256, 15, 25]
            将ReLU, 3*3步长为2的卷积(256->256)作用于x4, 得到x5, 尺寸为[batch_size, 256, 8, 13]
            最后的输出是一个元组, 元组中的元素尺寸如下
                [batch_size, 256, 116, 200]
                [batch_size, 256, 58, 100]
                [batch_size, 256, 29, 50]
                [batch_size, 256, 15, 25]
                [batch_size, 256, 8, 13]
        """
        x = self.backbone(batch_imgs)
        if self.with_neck:
            x = self.neck(x)
        #* 在SMOKE中输出为[batch_size, 64, 96, 320]的张量
        return x

    # TODO: Support test time augmentation
    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation."""
        pass

