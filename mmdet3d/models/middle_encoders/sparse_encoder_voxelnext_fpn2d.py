# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
from mmcv.ops import points_in_boxes_all, three_interpolate, three_nn
from mmdet.models.losses import sigmoid_focal_loss, smooth_l1_loss
from mmengine.runner import amp
from torch import Tensor
from torch import nn as nn

from mmdet3d.models.layers import SparseBasicBlock, make_sparse_convmodule
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
from mmdet3d.registry import MODELS
from mmdet3d.structures import BaseInstance3DBoxes
from mmdet3d.models.middle_encoders import SparseEncoder

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor, SparseSequential, SparseConv2d
else:
    from mmcv.ops import SparseConvTensor, SparseSequential

TwoTupleIntType = Tuple[Tuple[int]]


@MODELS.register_module()
class SparseEncoderVoxelNeXtFPN2D(SparseEncoder):
    r"""Sparse encoder for VoxelNeXt.

    Args:
        in_channels (int): The number of input channels.
        sparse_shape (list[int]): The sparse shape of input tensor.
        order (tuple[str], optional): Order of conv module.
            Defaults to ('conv', 'norm', 'act').
        norm_cfg (dict, optional): Config of normalization layer. Defaults to
            dict(type='BN1d', eps=1e-3, momentum=0.01).
        base_channels (int, optional): Out channels for conv_input layer.
            Defaults to 16.
        output_channels (int, optional): Out channels for conv_out layer.
            Defaults to 128.
        encoder_channels (tuple[tuple[int]], optional):
            Convolutional channels of each encode block.
            Defaults to ((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)).
        encoder_paddings (tuple[tuple[int]], optional):
            Paddings of each encode block.
            Defaults to ((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)).
        block_type (str, optional): Type of the block to use.
            Defaults to 'conv_module'.
        return_middle_feats (bool): Whether output middle features.
            Default to False.
    """

    def __init__(
            self,
            in_channels: int,
            sparse_shape: List[int],
            order: Optional[Tuple[str]] = ('conv', 'norm', 'act'),
            norm_cfg: Optional[dict] = dict(
                type='BN1d', eps=1e-3, momentum=0.01),
            base_channels: Optional[int] = 16,
            output_channels: Optional[int] = 128,
            encoder_channels: Optional[TwoTupleIntType] = ((16, ), (32, 32,
                                                                    32),
                                                           (64, 64,
                                                            64), (64, 64, 64)),
            encoder_paddings: Optional[TwoTupleIntType] = ((1, ), (1, 1, 1),
                                                           (1, 1, 1),
                                                           ((0, 1, 1), 1, 1)),
            block_type: Optional[str] = 'conv_module',
            return_middle_feats: Optional[bool] = False,
            dense_out: Optional[bool] = False):
        super().__init__(in_channels,
                        sparse_shape,
                        order,
                        norm_cfg,
                        base_channels,
                        output_channels,
                        encoder_channels,
                        encoder_paddings,
                        block_type,
                        return_middle_feats)
        self.conv_out = make_sparse_convmodule(
            encoder_channels[-1][-1],
            self.output_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            norm_cfg=norm_cfg,
            padding=1,
            indice_key='spconv_down2',
            conv_type='SparseConv2d')

        self.shared_conv = make_sparse_convmodule(
            encoder_channels[-1][-1],
            self.output_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            norm_cfg=norm_cfg,
            padding=1,
            conv_type='SubMConv2d')
        
        in_channels = 128
        for i in range(3, 5):
            blocks_list = []
            blocks = (128, 128, 128)
            for j, out_channels in enumerate(tuple(blocks)):
                    padding = 1
                    if j == 0:
                        blocks_list.append(
                            make_sparse_convmodule(
                                in_channels,
                                out_channels,
                                3,
                                norm_cfg=norm_cfg,
                                stride=2,
                                padding=padding,
                                indice_key=f'spconv2d{i + 1}',
                                conv_type='SparseConv2d'))
                    else:
                        conv_cfg = dict(type='SubMConv2d', indice_key=f'subm{i+2}')
                        blocks_list.append(
                            SparseBasicBlock(
                                out_channels,
                                out_channels,
                                norm_cfg=norm_cfg,
                                conv_cfg=conv_cfg))
                    in_channels = out_channels
            self.__setattr__(f'conv{i+1}', SparseSequential(*blocks_list))
        self.dense_out = dense_out

        self.lateral_convs = nn.ModuleList()
        self.interp_convs = nn.ModuleList()
        for i in range(3):
            l_conv = make_sparse_convmodule(
                128,
                128,
                kernel_size=1,
                stride=1,
                norm_cfg=norm_cfg,
                padding=0,
                indice_key=f'spconv_lateral{i+1}',
                conv_type='SparseConv2d')
            self.lateral_convs.append(l_conv)
        for i in range(2):
            interp_conv = make_sparse_convmodule(
                128,
                128,
                kernel_size=3,
                stride=1,
                indice_key=f'spconv2d{i+4}',
                conv_type='SparseInverseConv2d'
            )
            self.interp_convs.append(interp_conv)
        self.fpn_conv_out = make_sparse_convmodule(
                128,
                128,
                kernel_size=3,
                stride=1,
                indice_key=f'fpn_conv_out',
                conv_type='SubMConv2d'
            )

    def make_encoder_layers2d(
        self,
        make_block: nn.Module,
        norm_cfg: Dict,
        in_channels: int,
        block_type: Optional[str] = 'conv_module',
        conv_cfg: Optional[dict] = dict(type='SubMConv2d')
    ) -> int:
        """make encoder layers using sparse convs.

        Args:
            make_block (method): A bounded function to build blocks.
            norm_cfg (dict[str]): Config of normalization layer.
            in_channels (int): The number of encoder input channels.
            block_type (str, optional): Type of the block to use.
                Defaults to 'conv_module'.
            conv_cfg (dict, optional): Config of conv layer. Defaults to
                dict(type='SubMConv3d').

        Returns:
            int: The number of encoder output channels.
        """
        assert block_type in ['conv_module', 'basicblock']
        self.encoder_layers = SparseSequential()

        for i, blocks in enumerate(self.encoder_channels):
            blocks_list = []
            for j, out_channels in enumerate(tuple(blocks)):
                padding = tuple(self.encoder_paddings[i])[j]
                # each stage started with a spconv layer
                # except the first stage
                if i != 0 and j == 0 and block_type == 'conv_module':
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            stride=2,
                            padding=padding,
                            indice_key=f'spconv{i + 1}',
                            conv_type='SparseConv2d'))
                elif block_type == 'basicblock':
                    if j == len(blocks) - 1 and i != len(
                            self.encoder_channels) - 1:
                        blocks_list.append(
                            make_block(
                                in_channels,
                                out_channels,
                                3,
                                norm_cfg=norm_cfg,
                                stride=2,
                                padding=padding,
                                indice_key=f'spconv{i + 1}',
                                conv_type='SparseConv2d'))
                    else:
                        assert conv_cfg['type'] == 'SubMConv2d'
                        conv_cfg['indice_key'] = f'subm{i + 1}'
                        blocks_list.append(
                            SparseBasicBlock(
                                out_channels,
                                out_channels,
                                norm_cfg=norm_cfg,
                                conv_cfg=conv_cfg))
                else:
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            padding=padding,
                            indice_key=f'subm{i + 1}',
                            conv_type='SubMConv2d'))
                in_channels = out_channels
            stage_name = f'encoder_layer{i + 1}'
            stage_layers = SparseSequential(*blocks_list)
            self.encoder_layers.add_module(stage_name, stage_layers)
        return out_channels
    
    def bev_out(self, x_conv):
        features_cat = x_conv.features
        indices_cat = x_conv.indices[:, [0, 2, 3]]
        spatial_shape = x_conv.spatial_shape[1:]

        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, _inv, features_cat)

        x_out = SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=spatial_shape,
            batch_size=x_conv.batch_size
        )
        return x_out

    @amp.autocast(enabled=False)
    def forward(self, voxel_features: Tensor, coors: Tensor,
                batch_size: int) -> Union[Tensor, Tuple[Tensor, list]]:
        """Forward of SparseEncoder.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, C).
            coors (torch.Tensor): Coordinates in shape (N, 4),
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.

        Returns:
            torch.Tensor | tuple[torch.Tensor, list]: Return spatial features
                include:

            - spatial_features (torch.Tensor): Spatial features are out from
                the last layer.
            - encode_features (List[SparseConvTensor], optional): Middle layer
                output features. When self.return_middle_feats is True, the
                module returns middle features.
        """
        coors = coors.int()
        input_sp_tensor = SparseConvTensor(voxel_features, coors,
                                           self.sparse_shape, batch_size)
        x = self.conv_input(input_sp_tensor)

        encode_features = []
        for encoder_layer in self.encoder_layers:
            if len(encoder_layer) == 3:
                for i, block in enumerate(encoder_layer):
                    x = block(x)
                    if i == 1:
                        encode_features.append(x)
            elif len(encoder_layer) == 2:
                x = encoder_layer(x)
                encode_features.append(x)
            else:
                raise RuntimeError
#            x = encoder_layer(x)
#            encode_features.append(x)

        x_conv4 = encode_features[-1]
        x_conv4 = self.bev_out(x_conv4)

        x_conv5 = self.conv4(x_conv4)
        x_conv6 = self.conv5(x_conv5)

        inputs = [x_conv4, x_conv5, x_conv6]
        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            #laterals[i - 1] = laterals[i - 1] + self.interp_convs[i - 1](laterals[i])
            interp_out = self.interp_convs[i - 1](laterals[i])
            laterals[i - 1] = laterals[i - 1].replace_feature(laterals[i - 1].features+interp_out.features)

        # build outputs
        # part 1: from original levels
        out = self.fpn_conv_out(laterals[0])

        out = self.conv_out(out)
        if self.dense_out:
            spatial_features = self.shared_conv(out).dense()
        else:
            spatial_features = self.shared_conv(out)
        if self.return_middle_feats:
            return spatial_features, encode_features
        else:
            return spatial_features
