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
class SparseEncoderVoxelNeXt(SparseEncoder):
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
            dense_out: Optional[bool] = False,
            return_bev: Optional[bool] = False,
            **kwargs):
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
        self.return_bev = return_bev
        if not self.return_bev:
            self.conv_out = make_sparse_convmodule(
                encoder_channels[-1][-1],
                self.output_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='spconv_down2',
                conv_type='SparseConv2d')
            # self.conv_out = make_sparse_convmodule(
            #     encoder_channels[-1][-1],
            #     self.output_channels,
            #     kernel_size=(5, 5),
            #     stride=(1, 1),
            #     norm_cfg=norm_cfg,
            #     padding=2,
            #     indice_key='subm_down2',
            #     conv_type='SubMConv2d')

            self.shared_conv = make_sparse_convmodule(
                encoder_channels[-1][-1],
                self.output_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                #norm_cfg=norm_cfg,
                norm_cfg=dict(
                    type='BN1d', eps=1e-5, momentum=0.1),
                padding=1,
                conv_type='SubMConv2d',
                use_bias_before_norm=True)
        else:
            self.conv_out = None
            self.shared_conv = None
        self.dense_out = dense_out
        
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
                batch_size: int, **kwargs) -> Union[Tensor, Tuple[Tensor, list]]:
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

        x_conv6 = encode_features[-1]
        x_conv5 = encode_features[-2]
        x_conv4 = encode_features[-3]

        debug = False
        if debug:
            x_conv4_dense = x_conv4.dense().detach()[0].sum(dim=(0,1))
            x_conv5_dense = x_conv5.dense().detach()[0].sum(dim=(0,1))
            x_conv6_dense = x_conv6.dense().detach()[0].sum(dim=(0,1))
            import matplotlib.pyplot as plt
            plt.subplot(3,1,1)
            plt.imshow(x_conv4_dense.cpu().numpy())
            plt.subplot(3,1,2)
            plt.imshow(x_conv5_dense.cpu().numpy())
            plt.subplot(3,1,3)
            plt.imshow(x_conv6_dense.cpu().numpy())
            plt.savefig('a.png')
            plt.clf()

        x_conv5.indices[:, 1:] *= 2
        x_conv6.indices[:, 1:] *= 4

        x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features, x_conv6.features]))
        x_conv4.indices = torch.cat([x_conv4.indices, x_conv5.indices, x_conv6.indices])

        out_bev = self.bev_out(x_conv4)
        if self.return_bev: 
            return out_bev
        out = self.conv_out(out_bev)
        if self.dense_out:
            spatial_features = self.shared_conv(out).dense()
        else:
            spatial_features = self.shared_conv(out)

        debug = False
        if debug:
            from mmdet3d.visualization import Det3DLocalVisualizer
            from mmengine.visualization import Visualizer
            vis: Det3DLocalVisualizer = Visualizer.get_current_instance()
            vis.set_image(out_bev.dense().sum([0,1]).T.flip(0).flip(1).cpu().numpy())
            vis.fig_save_canvas.print_figure('./middle_vis_2/out_bev.png')
            vis.set_image(out.dense().sum([0,1]).T.flip(0).flip(1).cpu().numpy())
            vis.fig_save_canvas.print_figure('./middle_vis_2/out.png')
            vis.set_image(spatial_features.dense().sum([0,1]).T.flip(0).flip(1).cpu().numpy())
            vis.fig_save_canvas.print_figure('./middle_vis_2/spatial_features.png')

        if self.return_middle_feats:
            return spatial_features, encode_features
        else:
            return spatial_features
