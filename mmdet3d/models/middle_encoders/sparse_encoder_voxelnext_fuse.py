# Copyright (c) OpenMMLab. All rights reserved.
import os
import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from mmcv.ops import points_in_boxes_all, three_interpolate, three_nn
from mmdet.models.losses import sigmoid_focal_loss, smooth_l1_loss
from mmengine.runner import amp
from torch import Tensor
from torch import nn as nn
try:
    from torch_scatter import scatter_mean, scatter_max
except:
    import warnings
    warnings.warn("No torch_scatter found!")
from mmdet3d.models.layers import SparseBasicBlock, make_sparse_convmodule
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
from mmdet3d.registry import MODELS
from mmdet3d.structures import BaseInstance3DBoxes
from mmdet3d.models.middle_encoders import SparseEncoder
from mmdet3d.models.utils import voxel_aggregation_utils
from mmdet3d.structures.bbox_3d import get_proj_mat_by_coord_type
from mmdet3d.models.layers.fusion_layers.voxel_fusion import proj_to_img

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor, SparseSequential, SparseConv3d, SparseMaxPool3d, SparseAvgPool3d
else:
    from mmcv.ops import SparseConvTensor, SparseSequential

TwoTupleIntType = Tuple[Tuple[int]]


@MODELS.register_module()
class SparseEncoderVoxelNeXtFuse(nn.Module):
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
            num_feature_levels = 4,
            embed_dims = 256,
            encoder_channels: Optional[TwoTupleIntType] = ((16, 16), (32, 32),
                                                           (64, 64), (128, 128), (128, 128), (128, 128)),
            block_type: Optional[str] = 'conv_module',
            return_middle_feats: Optional[bool] = False,
            dense_out: Optional[bool] = False,
            seg_branch: Optional[dict] = None,
            voxel_size: Optional[List[float]] = [0.05, 0.05, 0.125],
            point_cloud_range: Optional[List[float]] = [0, -25.6, -3, 51.2, 25.6, 2],
            optional_cfg: Optional[dict] = dict(),
        ):
        super().__init__()
        assert block_type in ['conv_module', 'basicblock']
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.order = order
        #self.base_channels = base_channels
        self.base_channels = encoder_channels[0][0]
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.stage_num = len(self.encoder_channels)
        self.return_middle_feats = return_middle_feats
        self.optional_cfg = optional_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.use_seperate_radar_branch = self.optional_cfg.get('use_seperate_radar_branch', False)
        if self.use_seperate_radar_branch:
            self.radar_branch = self.optional_cfg.get('sep_radar_branch_cfg', dict(
                    type='SparseEncoderVoxelNeXt',
                    in_channels=in_channels,
                    base_channels=8,
                    sparse_shape=sparse_shape,
                    output_channels=64,
                    order=('conv', 'norm', 'act'),
                    encoder_channels=((8, 8, 16), (16, 16, 32), (32, 32, 64), (64, 64, 64), (64, 64, 64), (64, 64)),
                    encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0)),
                    block_type='basicblock',
                    return_bev=True))
            self.radar_branch = MODELS.build(self.radar_branch)
        
        self.use_seg_score_map = self.optional_cfg.get('use_seg_score_map', False)
        self.score_map_downsample_method = self.optional_cfg.get('score_map_downsample_method', 'maxpool')
        self.score_map_bev_reduce_method = self.optional_cfg.get('score_map_bev_reduce_method', 'avg')
        self.detach_score_map = self.optional_cfg.get('detach_score_map', True)
        # Spconv init all weight on its own

        assert isinstance(order, tuple) and len(order) == 3
        assert set(order) == {'conv', 'norm', 'act'}

        if self.order[0] != 'conv':  # pre activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d',
                order=('conv', ))
        else:  # post activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d')

        # 第一层 两个残差块
        self.conv1 = SparseSequential(
            #self.conv_input,
            SparseBasicBlock(encoder_channels[0][1], encoder_channels[0][1], norm_cfg=norm_cfg, conv_cfg=dict(type='SubMConv3d', indice_key='res1')),
            SparseBasicBlock(encoder_channels[0][1], encoder_channels[0][1], norm_cfg=norm_cfg, conv_cfg=dict(type='SubMConv3d', indice_key='res1')),
        )
        # 第二层(/2) 稀疏卷积下采样 后接两个残差块
        self.conv2 = SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            make_sparse_convmodule(encoder_channels[0][-1], encoder_channels[1][0], 3,norm_cfg=norm_cfg, stride=2, padding=1, indice_key=f'spconv2', conv_type='SparseConv3d'),
            SparseBasicBlock(encoder_channels[1][1], encoder_channels[1][1], norm_cfg=norm_cfg, conv_cfg=dict(type='SubMConv3d', indice_key='res2')),
            SparseBasicBlock(encoder_channels[1][1], encoder_channels[1][1], norm_cfg=norm_cfg, conv_cfg=dict(type='SubMConv3d', indice_key='res2')),
        )
        # 第三层(/4) 稀疏卷积下采样 后接两个残差块
        self.conv3 = SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            make_sparse_convmodule(encoder_channels[1][-1], encoder_channels[2][0], 3,norm_cfg=norm_cfg, stride=2, padding=1, indice_key=f'spconv3', conv_type='SparseConv3d'),
            SparseBasicBlock(encoder_channels[2][1], encoder_channels[2][1], norm_cfg=norm_cfg, conv_cfg=dict(type='SubMConv3d', indice_key='res3')),
            SparseBasicBlock(encoder_channels[2][1], encoder_channels[2][1], norm_cfg=norm_cfg, conv_cfg=dict(type='SubMConv3d', indice_key='res3')),
        )

        # 第四层(/8) 稀疏卷积下采样 后接两个残差块
        self.conv4 = SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            make_sparse_convmodule(encoder_channels[2][-1], encoder_channels[3][0], 3,norm_cfg=norm_cfg, stride=2, padding=1, indice_key=f'spconv4', conv_type='SparseConv3d'),
            SparseBasicBlock(encoder_channels[3][1], encoder_channels[3][1], norm_cfg=norm_cfg, conv_cfg=dict(type='SubMConv3d', indice_key='res4')),
            SparseBasicBlock(encoder_channels[3][1], encoder_channels[3][1], norm_cfg=norm_cfg, conv_cfg=dict(type='SubMConv3d', indice_key='res4')),
        )

        # 第五层(/16) 稀疏卷积下采样 后接两个残差块
        self.conv5 = SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            make_sparse_convmodule(encoder_channels[3][-1], encoder_channels[4][0], 3,norm_cfg=norm_cfg, stride=2, padding=1, indice_key=f'spconv5', conv_type='SparseConv3d'),
            SparseBasicBlock(encoder_channels[4][1], encoder_channels[4][1], norm_cfg=norm_cfg, conv_cfg=dict(type='SubMConv3d', indice_key='res5')),
            SparseBasicBlock(encoder_channels[4][1], encoder_channels[4][1], norm_cfg=norm_cfg, conv_cfg=dict(type='SubMConv3d', indice_key='res5')),
        )
        # 第六层(/32) 稀疏卷积下采样 后接两个残差块
        self.conv6 = SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            make_sparse_convmodule(encoder_channels[4][-1], encoder_channels[5][0], 3,norm_cfg=norm_cfg, stride=2, padding=1, indice_key=f'spconv6', conv_type='SparseConv3d'),
            SparseBasicBlock(encoder_channels[5][1], encoder_channels[5][1], norm_cfg=norm_cfg, conv_cfg=dict(type='SubMConv3d', indice_key='res6')),
            SparseBasicBlock(encoder_channels[5][1], encoder_channels[5][1], norm_cfg=norm_cfg, conv_cfg=dict(type='SubMConv3d', indice_key='res6')),
        )
        
        conv_out_in_channels = encoder_channels[-1][-1]
        if self.use_seperate_radar_branch: conv_out_in_channels += self.radar_branch.encoder_channels[-1][-1]
        if self.use_seg_score_map: conv_out_in_channels += 1
        self.conv_out = make_sparse_convmodule(
            conv_out_in_channels,
            self.output_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            norm_cfg=norm_cfg,
            padding=1,
            indice_key='spconv_down2',
            conv_type='SparseConv2d')

        self.shared_conv = make_sparse_convmodule(
            self.output_channels,
            self.output_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            #norm_cfg=norm_cfg,
            norm_cfg=dict(
                type='BN1d', eps=1e-5, momentum=0.1),
            padding=1,
            conv_type='SubMConv2d',
            use_bias_before_norm=self.optional_cfg.get('shared_conv_use_bias', True)
            )
        
        self.dense_out = dense_out
        self.coord_type = 'LIDAR'
        self.loc_to_index = dict()
        for i, _ in enumerate(self.encoder_channels):
            self.loc_to_index[f'x_conv{i+1}'] = i

        self.meta_dict = dict(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
        )
        multi_scale_3d_strides = dict()
        for i, k in enumerate(self.loc_to_index.keys()):
            multi_scale_3d_strides[k] = 2**i
        self.meta_dict.update(multi_scale_3d_strides=multi_scale_3d_strides)
        self.use_img = self.optional_cfg.get('use_img', True)
        if not self.use_img:
            self.fusion_locations = []
            self.use_seg_score_map = False
            self.use_seperate_radar_branch = False
            self.fuse_after_residual = False
            return
        self.fusion_locations = self.optional_cfg.get('fusion_locations', ['x_conv1', 'x_conv2', 'x_conv3'])
        self.seg_cfg = self.optional_cfg.get('seg_cfg', dict(
            seg_method='final_seg',
            feature_src=['x_conv3'],
            seg_locations=['x_conv3'],
            seg_sup_locations=['x_conv3'],
            seg_apply_location='x_conv3'
        ))
        if not isinstance(self.seg_cfg['seg_apply_location'], list):
            self.seg_cfg['seg_apply_location'] = [self.seg_cfg['seg_apply_location']]
        self.apply_seg = not (self.seg_cfg['seg_apply_location'] == [None] or self.seg_cfg['seg_apply_location'].__len__==0)
        assert (not self.apply_seg) or self.seg_cfg['seg_apply_location'][-1] == self.seg_cfg['seg_locations'][-1] or self.seg_cfg['seg_apply_location']==['combine']
        self.seg_locations = self.seg_cfg.get('seg_locations', ['x_conv3'])
        self.seg_sup_locations = self.seg_cfg.get('seg_sup_locations', ['x_conv3'])
        self.use_seg = len(self.seg_locations)>0
        self.use_query_to_seg = self.optional_cfg.get('use_query_to_seg', False)
        self.query_voxel_fuse_method = self.optional_cfg.get('query_voxel_fuse_method', 'add')
        self.apply_seg_scores_method = self.optional_cfg.get('apply_seg_scores_method', 'mul')
        self.deform_weight_act_fun = self.optional_cfg.get('deform_weight_act_fun', 'softmax')
        self.query_init_feat_src = self.optional_cfg.get('query_init_feat_src', ['img','voxel','pos'])
        self.fusion_block_type = self.optional_cfg.get('fusion_block_type', 'deform')
        self.fuse_after_residual = self.optional_cfg.get('fuse_after_residual', False)
        self.init_query_order = self.optional_cfg.get('init_query_order', ('img','voxel','pos'))
        self.deform_block_residual = self.optional_cfg.get('deform_block_residual', True)
        if len(self.fusion_locations) > 0:
            self.img_levels_to_init_query = self.optional_cfg.get('img_levels_to_init_query', [0])
            self.img_levels_to_fuse = self.optional_cfg.get('img_levels_to_fuse', [0, 1, 2, 3, 4])
            self.img_dim = self.optional_cfg.get('img_dim', 256)
            self.init_query_img_dim = self.img_dim*len(self.img_levels_to_init_query)
            self.init_query_dim = 0
            if 'img' in self.query_init_feat_src:
                self.init_query_dim += self.init_query_img_dim
            if 'voxel' in self.query_init_feat_src:
                self.init_query_dim += self.encoder_channels[self.loc_to_index[self.fusion_locations[0]]][0]
            if 'pos' in self.query_init_feat_src:
                self.init_query_dim += 3
            #self.init_query_dim = 3+self.encoder_channels[self.loc_to_index[self.fusion_locations[0]]][0]+self.init_query_img_dim
            if self.optional_cfg.get('norm_init_query', True):
                self.query_proj = nn.Sequential(
                   nn.Linear(self.init_query_dim, self.base_channels),
                   nn.BatchNorm1d(self.base_channels),
                   #nn.ReLU(),
                )
            else:
                self.query_proj = nn.Linear(self.init_query_dim, self.base_channels)

            voxel_fuse = dict(type='VoxelFusion', 
                            img_channels=self.img_dim,
                            out_channels=self.img_dim,
                            img_levels=self.img_levels_to_init_query,
                            lateral_conv=False, 
                            fuse_out=False)
            self.voxel_fuse = MODELS.build(voxel_fuse)
        
        self.fusion_blocks = nn.ModuleDict()
        for i, loc in enumerate(self.fusion_locations):
            if self.fusion_block_type == 'deform':
                fusion_block = dict(type='VoxelFusionBlock',
                                    img_cross_att = 
                                            dict(type='RadarImageCrossAttention',
                                                query_embed_dims=self.encoder_channels[i][0],#256
                                                value_embed_dims=self.img_dim,
                                                output_embed_dims=self.encoder_channels[i][0],#256
                                                deformable_attention=dict(
                                                    type='MSDeformableAttention',
                                                    num_levels=len(self.img_levels_to_fuse),
                                                    weight_act_func=self.deform_weight_act_fun,
                                                    residual=self.deform_block_residual
                                                    ),
                                            )
                                    )
                self.fusion_blocks[loc] = MODELS.build(fusion_block)
            elif self.fusion_block_type == 'simple':
                fusion_block = dict(type='SimpleVoxelFusionBlock', 
                            img_channels=self.img_dim,
                            out_channels=self.encoder_channels[i][0],#self.img_dim,
                            img_levels=self.img_levels_to_fuse,
                            lateral_conv=False, 
                            fuse_out=False)
                self.fusion_blocks[loc] = MODELS.build(fusion_block)
            else:
                raise NotImplementedError
        
        self.seg_heads = nn.ModuleDict()
        self.score_downsample = nn.ModuleDict()
        self.skip_conv = nn.ModuleDict()
        for i, loc in enumerate(self.seg_locations):
            selector = 0 if self.query_voxel_fuse_method != 'pure_cat' else 1
            if self.seg_cfg['seg_method'] == 'final_seg':
                assert len(self.seg_locations) == 1
                seg_in_channels = 0
                for l in self.seg_cfg['feature_src']:
                    seg_in_channels += self.encoder_channels[self.loc_to_index[l]][selector]
            else:
                seg_in_channels = self.encoder_channels[i][selector]
            seg_head = dict(type='ForegroundSegmentationHead',
                        in_channels=seg_in_channels)
            self.seg_heads[loc] = MODELS.build(seg_head)

            if self.seg_cfg['seg_method'] == 'seperate_seg':
                src_loc = loc
                target_loc = self.seg_locations[-1]
                stride = int(self.meta_dict['multi_scale_3d_strides'][target_loc]/self.meta_dict['multi_scale_3d_strides'][src_loc])
                downsample_time = int(math.log2(stride))
                score_downsample_method = self.seg_cfg.get('score_downsample_method', 'avg')
                if score_downsample_method == 'avg':
                    self.score_downsample[loc] = SparseSequential(*[SparseAvgPool3d(3, 2, 1, indice_key=f'spconv{self.loc_to_index[loc]+i+2}') for i in range(downsample_time)])
                elif score_downsample_method == 'maxpool':
                    self.score_downsample[loc] = SparseSequential(*[SparseMaxPool3d(3, 2, 1, indice_key=f'spconv{self.loc_to_index[loc]+i+2}') for i in range(downsample_time)])
                elif score_downsample_method == 'conv':
                    self.score_downsample[loc] = SparseSequential(*[SparseConv3d(1, 1, 3, stride=2, padding=1, bias=True, indice_key=f'spconv{self.loc_to_index[loc]+i+2}') for i in range(downsample_time)])
                else:
                    raise NotImplementedError
                
        if self.seg_cfg['seg_method'] == 'final_seg' and len(self.seg_cfg['feature_src']) > 1:
            for loc in self.seg_cfg['feature_src']:
                src_loc = loc
                target_loc = self.seg_locations[-1]
                idx = self.loc_to_index[loc]
                stride = int(self.meta_dict['multi_scale_3d_strides'][target_loc]/self.meta_dict['multi_scale_3d_strides'][src_loc])
                downsample_time = int(math.log2(stride))
                skip_conv_method = self.seg_cfg.get('skip_conv_method', 'conv')
                if skip_conv_method == 'avg':
                    self.skip_conv[loc] = SparseSequential(*[SparseAvgPool3d(3, stride=2, padding=1, indice_key=f'spconv{idx+i+2}') for i in range(downsample_time)])
                elif skip_conv_method == 'maxpool':
                    self.skip_conv[loc] = SparseSequential(*[SparseMaxPool3d(3, stride=2, padding=1, indice_key=f'spconv{idx+i+2}') for i in range(downsample_time)])
                elif skip_conv_method == 'conv':
                    self.skip_conv[loc] = SparseSequential(*[SparseConv3d(self.encoder_channels[idx][0], self.encoder_channels[idx][0], 3, stride=2, padding=1, bias=True, indice_key=f'spconv{idx+i+2}') for i in range(downsample_time)])
                else:
                    raise NotImplementedError

        if self.optional_cfg.get('use_depth_match', False):
            self.loc_depth_ffn = nn.Linear(7, 64)
            self.sem_depth_ffn = nn.Linear(self.init_query_dim, 64)

        if self.query_voxel_fuse_method == 'dynamic_add':
            self.query_voxel_fuse_weight = nn.ParameterDict([(loc, nn.parameter.Parameter(torch.tensor(0.0), requires_grad=True)) for loc in self.fusion_locations])
        elif self.query_voxel_fuse_method == 'channel_dynamic_add':
            self.query_voxel_fuse_weight = nn.ParameterDict([(loc, nn.parameter.Parameter(torch.zeros([1,self.encoder_channels[self.loc_to_index[loc]][0]]), requires_grad=True)) for loc in self.fusion_locations])
        elif self.query_voxel_fuse_method == 'cat_and_map':
            self.query_voxel_fuse_map = nn.ModuleDict([(loc, nn.Sequential(
                nn.Linear(self.encoder_channels[self.loc_to_index[loc]][0]*2, self.encoder_channels[self.loc_to_index[loc]][0]),
                nn.BatchNorm1d(self.encoder_channels[self.loc_to_index[loc]][0]),
                #nn.ReLU(inplace=True)
                )) for loc in self.fusion_locations])
        
        if self.use_seg_score_map:
            src_loc = self.seg_locations[-1]
            idx = self.loc_to_index[src_loc]
            target_loc = 'x_conv6'
            stride = int(self.meta_dict['multi_scale_3d_strides'][target_loc]/self.meta_dict['multi_scale_3d_strides'][src_loc])
            downsample_time = int(math.log2(stride))
            if self.score_map_downsample_method == 'maxpool':
                self.score_map_downsample = SparseSequential(
                    *[SparseMaxPool3d(3, 2, 1, indice_key=f'spconv{idx+i+2}') for i in range(downsample_time)]
                    )
            elif self.score_map_downsample_method == 'avg':
                self.score_map_downsample = SparseSequential(
                    *[SparseAvgPool3d(3, 2, 1, indice_key=f'spconv{idx+i+2}') for i in range(downsample_time)]
                    )
            else:
                raise NotImplementedError
            

    def init_weights(self):
        if hasattr(self, 'score_conv'):
            torch.nn.init.constant_(self.score_conv.bias, 0.0)
            torch.nn.init.constant_(self.score_conv.weight, 1/3)
            for param in self.score_conv.parameters():
                param.requires_grad_(False)

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
    
    def score_map_bev_out(self, x_conv):
        features_cat = x_conv.features
        indices_cat = x_conv.indices[:, [0, 2, 3]]
        spatial_shape = x_conv.spatial_shape[1:]

        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
        #features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        #features_unique.index_reduce_(0, _inv, features_cat, 'mean', include_self=False) # since pytorch v1.12
        if self.score_map_bev_reduce_method == 'avg':
            features_unique = scatter_mean(features_cat, _inv, 0)
        elif self.score_map_bev_reduce_method == 'maxpool':
            features_unique = scatter_max(features_cat, _inv, 0)[0]
        else:
            raise NotImplementedError

        x_out = SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=spatial_shape,
            batch_size=x_conv.batch_size
        )
        return x_out
    
    def proj_to_img(self, reference_points, meta):
        '''
        Args:
            reference_points (torch.Tensor): shape [N, 3].
            meta (dict): meta info.
        '''
        lidar2img = np.asarray(meta['lidar2img'])
        lidar2img = reference_points.new_tensor(lidar2img)  # (4, 4)
        reference_points = reference_points.clone()

        # denormalize
        # reference_points[..., 0:1] = reference_points[..., 0:1] * \
        #     (pc_range[3] - pc_range[0]) + pc_range[0]
        # reference_points[..., 1:2] = reference_points[..., 1:2] * \
        #     (pc_range[4] - pc_range[1]) + pc_range[1]
        # reference_points[..., 2:3] = reference_points[..., 2:3] * \
        #     (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1) # (N, 4)

        reference_points_cam = torch.matmul(
            reference_points.to(torch.float32),
            lidar2img.to(torch.float32).T)
        eps = 1e-5

        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3],
            torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        reference_points_cam[..., 0] /= meta['batch_input_shape'][1]
        reference_points_cam[..., 1] /= meta['batch_input_shape'][0]

        return reference_points_cam
    
    def flatten_img_feats(self, img_feats):
        bs = img_feats[0].shape[0]
        dtype = img_feats[0].dtype
        device = img_feats[0].device
        # flatten image features of different scales
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(img_feats):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            # (B, C, H, W) -> (B, C, HW) -> (B, HW, C)
            feat = feat.flatten(2).permute(0, 2, 1)
            feat = feat# + self.level_embeds[None, lvl:lvl + 1, :].to(dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 1)  # bs, hw++, c
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        image_data_dict = {
            'img_feats': img_feats,
            'feat_flatten': feat_flatten,
            'spatial_shapes': spatial_shapes,
            'level_start_index': level_start_index
        }
        return image_data_dict
    
    def get_voxel_centorids(self, points, locations, batch_input_metas):
        bs = len(points)
        batched_points = []
        for i in range(bs):
            batched_points.append(torch.cat((i*points[i].new_ones([points[i].shape[0],1]),points[i]), dim=-1))
        batched_points = torch.cat(batched_points, dim=0)
        
        # 每层体素的质心以及质心对应的voxel坐标
        voxel_data_dict = {}
        centroids_all, centroid_voxel_coors_all = voxel_aggregation_utils.get_centroids_per_voxel_layer(batched_points,
        locations,
        self.meta_dict['multi_scale_3d_strides'],
        self.meta_dict['voxel_size'],
        self.meta_dict['point_cloud_range'])

        for k in centroids_all.keys():
            if self.use_img:
                centroids_img = []
                for i in range(bs):
                    batch_mask = (centroid_voxel_coors_all[k][:,0].int()==i)
                    voxel_centroids = centroids_all[k][batch_mask]
                    proj_mat = get_proj_mat_by_coord_type(batch_input_metas[i], self.coord_type)
                    centroids_img.append(proj_to_img(voxel_centroids[:,1:4], batch_input_metas[i], voxel_centroids.new_tensor(proj_mat), self.coord_type, normalize=True))
                centroids_img = torch.cat(centroids_img)
                centroids_img = torch.cat([centroids_all[k][:, 0].unsqueeze(1) ,centroids_img], dim=-1)
            else:
                centroids_img = None
            voxel_data_dict[k] = {
                'centroids': centroids_all[k],
                'centroids_img': centroids_img,
                'centroids_coor': centroid_voxel_coors_all[k],
            }
        return voxel_data_dict
    
    def init_query(self, img_feats, x_conv, voxel_data_dict, batch_input_meta):
        init_query = []
        bs = len(batch_input_meta)
        
        centroids_coor = voxel_data_dict['centroids_coor']
        centroids = voxel_data_dict['centroids']
        centroids_img = voxel_data_dict['centroids_img']
        non_empty_inds = voxel_data_dict['non_empty_inds']
        non_empty_mask = voxel_data_dict['non_empty_mask']
        non_empty_centroids = centroids[non_empty_mask]
        non_empty_centroids_img = centroids_img[non_empty_mask]
        non_empty_centroids_coor = centroids_coor[non_empty_mask]

        if 'img' in self.query_init_feat_src:
            centroids_list = []
            centroids_img_list = []
            batch_mask_list = []

            sampled_img_feats = torch.zeros([x_conv.features[non_empty_inds].shape[0],self.init_query_img_dim], dtype=x_conv.features.dtype, device=x_conv.features.device)
            for i in range(bs):
                batch_mask = (non_empty_centroids_coor[:, 0].int() == i)
                centroids_list.append(non_empty_centroids[batch_mask][...,1:])
                centroids_img_list.append(non_empty_centroids_img[batch_mask][...,1:])
                batch_mask_list.append(batch_mask)
            
            #sampled_img_feats_list = self.voxel_fuse(img_feats, centroids_list, batch_input_meta)
            sampled_img_feats_list = self.voxel_fuse(img_feats, centroids_img_list, batch_input_meta)
            # fix bug: batch index is not continuous because we use empty voxel
            for i in range(bs):
                sampled_img_feats[batch_mask_list[i]] = sampled_img_feats_list[i]
            #sampled_img_feats = torch.cat(sampled_img_feats_list)
            init_query.append(sampled_img_feats)

        if 'voxel' in self.query_init_feat_src:
            voxel_feats = x_conv.features[non_empty_inds]
            init_query.append(voxel_feats)

        if 'pos' in self.query_init_feat_src:
            pos_info = non_empty_centroids[...,1:4].clone()
            pos_info[..., 0] = (pos_info[..., 0]-self.point_cloud_range[0]) / (self.point_cloud_range[3]-self.point_cloud_range[0])
            pos_info[..., 1] = (pos_info[..., 1]-self.point_cloud_range[1]) / (self.point_cloud_range[4]-self.point_cloud_range[1])
            pos_info[..., 2] = (pos_info[..., 2]-self.point_cloud_range[2]) / (self.point_cloud_range[5]-self.point_cloud_range[2])
            init_query.append(pos_info)

        # # TODO:temp code
        # if len(self.init_query_order)==3 and self.init_query_order != ('img', 'voxel', 'pos'):
        #     temp_list=['img','voxel','pos']
        #     temp = (temp_list.index(self.init_query_order[0]), temp_list.index(self.init_query_order[1]), temp_list.index(self.init_query_order[2]))
        #     init_query = [init_query[i] for i in temp]

        #init_query = torch.cat([sampled_img_feats, voxel_feats, non_empty_centroids[...,1:]], dim=-1)
        #init_query = torch.cat([sampled_img_feats, voxel_feats, pos_info], dim=-1)
        init_query = torch.cat(init_query, dim=-1)
        init_query = self.query_proj(init_query)
        # x_conv.features[non_empty_inds] = init_query

        voxel_dict = {'centroids': non_empty_centroids}
        return init_query, voxel_dict
    
    def get_non_empty_voxel_inds(self, x_conv, voxel_data_dict, use_empty_voxel=False, batch_input_metas=None, loc=None):
        centroids_coor = voxel_data_dict['centroids_coor']
        non_empty_inds, non_empty_mask = \
                voxel_aggregation_utils.get_nonempty_voxel_feature_indices(centroids_coor, x_conv)
        voxel_data_dict.update({
            'non_empty_inds': non_empty_inds,
            'non_empty_mask': non_empty_mask
        })
        
        if use_empty_voxel:
            if len(non_empty_inds) == x_conv.features.shape[0]:
                return
            empty_mask = torch.ones((x_conv.features.shape[0]), dtype=torch.bool, device=centroids_coor.device)
            empty_mask[non_empty_inds] = False
            cur_coords = x_conv.indices[empty_mask]
            xyz = voxel_aggregation_utils.get_voxel_centers(cur_coords[:, 1:4], self.meta_dict['multi_scale_3d_strides'][loc], self.meta_dict['voxel_size'],
                    self.meta_dict['point_cloud_range'])
            bxyz = torch.cat([cur_coords[:, 0].float().unsqueeze(1), xyz], dim=-1)
            voxel_data_dict['centroids'] = torch.cat([voxel_data_dict['centroids'][:,:4], bxyz])
            voxel_data_dict['centroids_coor'] = torch.cat([voxel_data_dict['centroids_coor'], cur_coords])
            bs = len(batch_input_metas)
            # TODO: centroids_img
            centroids_img = []
            for i in range(bs):
                batch_mask = (cur_coords[:,0].int()==i)
                xyz_batch = xyz[batch_mask]
                proj_mat = get_proj_mat_by_coord_type(batch_input_metas[i], self.coord_type)
                centroids_img.append(proj_to_img(xyz_batch, batch_input_metas[i], xyz_batch.new_tensor(proj_mat), self.coord_type, normalize=True))
            centroids_img = torch.cat(centroids_img)
            centroids_img = torch.cat([cur_coords[:,0].unsqueeze(1) ,centroids_img], dim=-1)
            voxel_data_dict['centroids_img'] = torch.cat([voxel_data_dict['centroids_img'], centroids_img])

            assert (voxel_data_dict['centroids'].shape[0] == x_conv.features.shape[0]) \
               and (voxel_data_dict['centroids_img'].shape[0] == x_conv.features.shape[0]) \
               and (voxel_data_dict['centroids_coor'].shape[0] == x_conv.features.shape[0])
            # TODO: need to recompute inds?
            centroids_coor = voxel_data_dict['centroids_coor']
            non_empty_inds, non_empty_mask = \
                voxel_aggregation_utils.get_nonempty_voxel_feature_indices(centroids_coor, x_conv)
            voxel_data_dict.update({
            'non_empty_inds': non_empty_inds,
            'non_empty_mask': non_empty_mask
            })

    def query_voxel_fuse(self, x, updated_query, inds, loc, fuse_method=None):
        method = self.query_voxel_fuse_method if fuse_method is None else fuse_method
        if method == 'add':
            new_features = torch.clone(x.features)
            new_features[inds] = updated_query + new_features[inds]
        elif method == 'nothing':
            new_features = torch.clone(x.features)
            new_features[inds] = updated_query
        elif method == 'dynamic_add' or method == 'channel_dynamic_add':
            new_features = torch.clone(x.features)
            weight = self.query_voxel_fuse_weight[loc].sigmoid()
            new_features[inds] = weight * updated_query + (1 - weight) * new_features[inds]
        elif method == 'pure_cat' or method == 'cat_and_map':
            assert len(inds) == x.features.shape[0]
            new_features = x.features.new_zeros([x.features.shape[0], x.features.shape[1]+updated_query.shape[1]])
            new_features[:, :x.features.shape[1]] = torch.clone(x.features)
            new_features[inds, :updated_query.shape[1]] = updated_query
            if method == 'cat_and_map':
                new_features = self.query_voxel_fuse_map[loc](new_features)
        else:
            raise NotImplementedError(f'We do not implement {method} in query_voxel_fuse')
        return new_features
    
    def apply_seg_scores(self, x, seg_preds, inds, apply_method=None):
        method = self.apply_seg_scores_method if apply_method is None else apply_method
        if method == 'mul':
            new_features = torch.clone(x.features)
            new_features[inds] = seg_preds.sigmoid() * new_features[inds]
        elif method == 'cat':
            new_features = x.features.new_zeros([x.features.shape[0],x.features.shape[1]+1])
            new_features[:, :x.features.shape[1]] = torch.clone(x.features)
            #new_features[inds, -1] = seg_preds.sigmoid().squeeze()
            new_features[inds] = torch.cat([new_features[inds, :x.features.shape[1]], seg_preds.sigmoid()], dim=-1)
        elif method == 'nothing':
            return x.features
        else:
            raise NotImplementedError(f'We do not implement {method} in apply_seg_scores')
        return new_features
    
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
        points = kwargs['points']
        imgs = kwargs.get('imgs', None)
        batch_input_metas = kwargs['batch_input_metas']
        bs = len(batch_input_metas)
        img_feats = kwargs.get('img_features', None)
        locations = list(self.meta_dict['multi_scale_3d_strides'].keys())
        if imgs is not None:
            img_data_dict = self.flatten_img_feats(img_feats)
            img_data_dict['imgs'] = imgs
            self.use_img = True
        else:
            self.use_img = False
        #voxel_data_dict = self.get_voxel_centorids(points, locations, batch_input_metas)
        if len(self.fusion_locations) > 0:
            voxel_data_dict = self.get_voxel_centorids(points, self.fusion_locations, batch_input_metas)
        coors = coors.int()
        input_sp_tensor = SparseConvTensor(voxel_features, coors,
                                           self.sparse_shape, batch_size)
        x = self.conv_input(input_sp_tensor)

        seg_results_dict = dict()
        middle_feats = dict()
        middle_feats_for_seg = dict()
        middle_seg_scores = dict()
        score_maps = dict()
        for i in range(self.stage_num):
            block = self.__getattr__(f'conv{i+1}')
            loc = f'x_conv{i+1}'
            if i > 0:
                x = block[0](x) # spconv or conv_input
            
            if self.fuse_after_residual:
                if i > 0:
                    x = block[1](x)
                    x = block[2](x)
                else:
                    x = block[0](x)
                    x = block[1](x)

            if loc in self.fusion_locations:
                # insert fusion block
                self.get_non_empty_voxel_inds(x, voxel_data_dict[loc], use_empty_voxel=True ,batch_input_metas=batch_input_metas, loc=loc)
                #assert (voxel_data_dict[loc]['non_empty_inds'].shape[0] == x.features.shape[0]) and torch.all(voxel_data_dict[loc]['non_empty_mask']) 
                debug = False
                if debug:
                    inds = voxel_data_dict[loc]['non_empty_inds']
                    inds = inds[voxel_data_dict[loc]['centroids_coor'][:,0]==0]
                    coor = x.indices
                    batch_mask = (coor[:, 0] == 0)
                    coor = coor[batch_mask]
                    voxel_centers_all = voxel_aggregation_utils.get_voxel_centers(coor[:, 1:4], self.meta_dict['multi_scale_3d_strides'][loc], self.meta_dict['voxel_size'], self.meta_dict['point_cloud_range'])
                    pts = 255 * voxel_centers_all.new_ones([voxel_centers_all.shape[0], 6])
                    pts[:, :3] = voxel_centers_all
                    pts[inds, 4:] = 0
                    pts = pts.cpu().numpy().astype(np.float32)
                    pts.tofile(f'{loc}_centroids.bin')
                if loc == self.fusion_locations[0]:
                    init_query, voxel_dict = self.init_query(img_feats, x, voxel_data_dict[loc], batch_input_metas)

                    if self.optional_cfg.get('use_depth_match', False):
                        loc_feats = voxel_dict['centroids'][:,1:]
                        semantic_feats = init_query
                        loc_depth_feats = self.loc_depth_ffn(loc_feats)
                        semantic_depth_feats = self.sem_depth_ffn(semantic_feats)
                        match_score = torch.mul(loc_depth_feats, semantic_depth_feats).sum(dim=1).sigmoid().unsqueeze(1)
                        init_query = match_score * init_query

                    #if loc in self.seg_locations:
                    if 'init_query' in self.seg_locations:
                        seg_results = self.seg_heads[loc](init_query)
                        seg_results_dict[loc] = dict(seg_results=seg_results, voxel_dict=voxel_dict)
                        #return x, {'seg_results':seg_results, 'voxel_dict':voxel_dict}

                if loc == self.fusion_locations[0]:
                    updated_query, sample_locs, sample_weights = self.fusion_blocks[loc](x, voxel_data_dict[loc], img_data_dict, external_query=init_query)
                else:
                    updated_query, sample_locs, sample_weights = self.fusion_blocks[loc](x, voxel_data_dict[loc], img_data_dict)

                debug=False
                show_deform=False
                if show_deform:# and loc == self.fusion_locations[1]:
                    for i in range(batch_size):
                        batch_mask = (voxel_data_dict[loc]['centroids_coor'][:,0].int()==0)
                        sample_locs_ = sample_locs[batch_mask].cpu().numpy()
                        sample_weights_ = sample_weights[batch_mask].cpu().numpy()
                        img = img_data_dict['imgs'][i]
                        img = img * batch_input_metas[i]['img_std'] + batch_input_metas[i]['img_mean']
                        import matplotlib.pyplot as plt
                        for k in range(sample_locs_.shape[0]): # num_query
                            plt.figure()
                            plt.imshow(img.permute(1,2,0).int().cpu().numpy())
                            plt.scatter(sample_locs_[k][:, 0]*batch_input_metas[i]['batch_input_shape'][1], sample_locs_[k][:, 1]*batch_input_metas[i]['batch_input_shape'][0], c=sample_weights_[k][:, 0], cmap='autumn', s=1)
                            frame_id = batch_input_metas[i]['img_path'].split('/')[-1].split('.')[0]
                            plt.savefig(f'vis_results/deform_vis/{frame_id}_{k}.png')
                            plt.clf()
                
                if debug and loc == self.fusion_locations[1]:
                    for i in range(batch_size):
                        batch_mask = (voxel_data_dict[loc]['centroids_coor'][:,0].int()==i)
                        coor_img = voxel_data_dict[loc]['centroids_img'][batch_mask][:,1:].cpu().numpy()
                        img = img_data_dict['imgs'][i]
                        img = img * batch_input_metas[i]['img_std'] + batch_input_metas[i]['img_mean']
                        import matplotlib.pyplot as plt
                        plt.figure()
                        plt.imshow(img.permute(1,2,0).int().cpu().numpy())
                        plt.scatter(coor_img[:,0]*batch_input_metas[i]['batch_input_shape'][1], coor_img[:,1]*batch_input_metas[i]['batch_input_shape'][0])
                        frame_id = batch_input_metas[i]['img_path'].split('/')[-1].split('.')[0]
                        plt.savefig(f'vis_results/centroids_img/{frame_id}.png')
                        plt.clf()
                        
                centroids = voxel_data_dict[loc]['centroids']
                centroids_img = voxel_data_dict[loc]['centroids_img']
                non_empty_mask = voxel_data_dict[loc]['non_empty_mask']
                non_empty_centroids = centroids[non_empty_mask]
                non_empty_centroids_img = centroids_img[non_empty_mask]
                voxel_dict = {'centroids': non_empty_centroids, 'inds': voxel_data_dict[loc]['non_empty_inds'], 'xconv_indices':x.indices, 'centroids_img': non_empty_centroids_img}

                if self.optional_cfg.get('mask_fake_forepts', False):
                    batch_gt_instance_3d = kwargs['batch_gt_instance_3d']
                    batch_gt_instance_2d = kwargs['batch_gt_instance_2d']
                    for bs in range(batch_size):
                        mask = (non_empty_centroids[:,0] == bs)
                        point_xyz = non_empty_centroids[mask][:, 1:4]
                        point_2d = voxel_data_dict[loc]['centroids_img'][mask][:, 1:]
                        point_2d[:, 0] *= batch_input_metas[bs]['input_shape'][1]
                        point_2d[:, 1] *= batch_input_metas[bs]['input_shape'][0]
                        if hasattr(batch_gt_instance_3d[bs], 'bboxes_3d'):
                            gt_bboxes_3d = batch_gt_instance_3d[bs].bboxes_3d
                        else:
                            gt_bboxes_3d = batch_gt_instance_3d[bs]
                        if hasattr(batch_gt_instance_2d[bs], 'bboxes'):
                            gt_bboxes_2d = batch_gt_instance_2d[bs].bboxes
                        else:
                            gt_bboxes_2d = torch.tensor(batch_gt_instance_2d[bs]).cuda()
                        point_2d = point_2d.unsqueeze(1)
                        gt_bboxes_2d = gt_bboxes_2d.unsqueeze(0)
                        box2d_fg_flag = (point_2d[...,0]>=gt_bboxes_2d[...,0]) \
                                    & (point_2d[...,0]<=gt_bboxes_2d[...,2]) \
                                    & (point_2d[...,1]>=gt_bboxes_2d[...,1]) \
                                    & (point_2d[...,1]<=gt_bboxes_2d[...,3])
                        box2d_fg_flag = torch.any(box2d_fg_flag, dim=1)
                        box2d_bg_flag = ~box2d_fg_flag
                        box_idxs_of_pts = gt_bboxes_3d.points_in_boxes_part(point_xyz).long()
                        box_fg_flag = box_idxs_of_pts >= 0
                        box_bg_flag = ~box_fg_flag
                        fg_flag = (box2d_bg_flag) | box_fg_flag
                        bg_flag = ~fg_flag
                        # bg_flag = box_bg_flag
                        # img = imgs[bs]*batch_input_metas[bs]['img_std']+batch_input_metas[bs]['img_mean']
                        # img = img.permute(1,2,0).cpu().numpy().astype(np.uint8)
                        # import matplotlib.pyplot as plt
                        # plt.figure()
                        # plt.imshow(img)
                        # if bg_flag.sum() > 0:
                        #     a=point_2d.squeeze().cpu().numpy()
                        #     plt.scatter(a[:,0],a[:,1])
                        # plt.savefig('z.png')
                        # plt.clf()

                        temp = updated_query[mask].detach()
                        temp[bg_flag] = 0
                        updated_query[mask] = temp

                inds = voxel_data_dict[loc]['non_empty_inds']
                new_features = self.query_voxel_fuse(x, updated_query, inds, loc)
                x = x.replace_feature(new_features)

                if loc in self.seg_locations:
                    if self.seg_cfg['seg_method'] == 'seperate_seg':
                        if loc not in seg_results_dict:
                            if self.use_query_to_seg:
                                seg_results = self.seg_heads[loc](updated_query)
                            else:
                                seg_results = self.seg_heads[loc](x.features[inds])
                            seg_results_dict[loc] = dict(seg_results=seg_results, voxel_dict=voxel_dict)

                        reordered_score = seg_results['seg_preds'].new_zeros(seg_results['seg_preds'].shape)
                        reordered_score[inds] = seg_results['seg_preds']
                        score_tensor = SparseConvTensor(reordered_score, x.indices, x.spatial_shape, bs)
                        middle_seg_scores[loc] = score_tensor
                
                if self.seg_cfg['seg_method'] == 'final_seg' and loc in self.seg_cfg['feature_src']:
                    if self.use_query_to_seg:
                        reordered_query = updated_query.new_zeros(updated_query.shape)
                        reordered_query[inds] = updated_query
                        query_tensor = SparseConvTensor(reordered_query, x.indices, x.spatial_shape, bs)
                        middle_feats_for_seg[loc] = query_tensor
                    else:
                        middle_feats_for_seg[loc] = x

                if self.use_seg and loc == self.seg_locations[-1]:
                    if self.seg_cfg['seg_method'] == 'final_seg':
                        all_feats = []
                        for l, x_middle in middle_feats_for_seg.items():
                            if l == loc: continue
                            downsampled_x = self.skip_conv[l](x_middle)
                            #assert (downsampled_x.indices-x.indices).abs().sum() < .5
                            all_feats.append(downsampled_x.features[inds])
                        if self.use_query_to_seg:
                            all_feats.append(updated_query)
                        else:
                            all_feats.append(x.features[inds])
                        all_feats = torch.cat(all_feats, dim=-1)
                        seg_results = self.seg_heads[loc](all_feats)
                        seg_results_dict[loc] = dict(seg_results=seg_results, voxel_dict=voxel_dict)
                    elif self.seg_cfg['seg_method'] == 'seperate_seg':
                        if self.seg_cfg['combine_method'] == 'avg':
                            all_preds = []
                            for l, score_middle in middle_seg_scores.items():
                                downsampled_score = self.score_downsample[l](score_middle)
                                #assert (x.indices-downsampled_score.indices).abs().sum() < .5
                                all_preds.append(downsampled_score.features[inds])
                            all_preds = torch.cat(all_preds, dim=-1)
                            final_preds = torch.mean(all_preds, dim=-1, keepdim=True)
                            seg_results_combine = dict()
                            seg_results_combine['seg_preds'] = final_preds
                            seg_results_dict['combine'] = dict(seg_results=seg_results_combine, voxel_dict=voxel_dict)
                        elif self.seg_cfg['combine_method'] == 'nothing':
                            pass
                        else:
                            raise NotImplementedError
                    else:
                        raise NotImplementedError

                    debug=False
                    if debug:
                        for key, v in seg_results_dict.items():
                            for b in range(batch_size):
                                pred_scores = v['seg_results']['seg_preds'].sigmoid()
                                pred_mask = v['seg_results']['seg_preds'].sigmoid()>0.3
                                centroids = v['voxel_dict']['centroids']
                                batch_mask = centroids[:, 0].int()==b
                                centroids = centroids[batch_mask]
                                pred_mask = pred_mask.squeeze()[batch_mask]
                                pred_scores = pred_scores.squeeze()[batch_mask]
                                a = centroids.new_zeros([centroids.shape[0], 6])
                                a[:, :3] = centroids[:, 1:4]
                                #a[pred_mask, 4] = 255
                                a[:, 4] = pred_scores * 255
                                save_path = f"./vis_results/seg_results_mm_tj4d_{key}"
                                if not os.path.exists(save_path):
                                    os.mkdir(save_path)
                                a.cpu().numpy().astype(np.float32).tofile(f"{save_path}/{batch_input_metas[b]['lidar_path'].split('/')[-1]}")

                if self.use_seg and (loc in self.seg_cfg['seg_apply_location'] or loc == self.seg_locations[-1]) and self.apply_seg:
                    if 'combine' in self.seg_cfg['seg_apply_location'] and loc == self.seg_locations[-1]:
                        seg_preds = seg_results_dict['combine']['seg_results']['seg_preds']
                    else:
                        seg_preds = seg_results_dict[loc]['seg_results']['seg_preds']
                    #seg_preds = seg_results_dict[self.seg_cfg['seg_apply_location']]['seg_results']['seg_preds']
                    new_features = self.apply_seg_scores(x, seg_preds, inds)
                    x = x.replace_feature(new_features)

                    if self.use_seg_score_map:
                        reordered_score = seg_results['seg_preds'].new_zeros(seg_results['seg_preds'].shape)
                        reordered_score[inds] = seg_results['seg_preds'].sigmoid().detach() if self.detach_score_map else seg_results['seg_preds'].sigmoid()
                        score_tensor = SparseConvTensor(reordered_score, x.indices, x.spatial_shape, bs)
                        score_maps[loc] = score_tensor
                    #return x, seg_results_dict
                # return x, {'seg_results':seg_results, 'voxel_dict':voxel_dict}

                #x = self.fusion_blocks[loc](x, voxel_data_dict[loc], img_data_dict)
            
            if not self.fuse_after_residual:
                if i > 0:
                    x = block[1](x)
                    x = block[2](x)
                else:
                    x = block[0](x)
                    x = block[1](x)
            middle_feats[loc] = x

        if self.use_seg_score_map:
            key = self.seg_locations[-1]
            for module in self.score_map_downsample:
                new_key = key[:-1] + f'{int(key[-1])+1}'
                score_maps[new_key] = module(score_maps[key])
                key = new_key
            score_map4 = score_maps['x_conv4']
            score_map5 = score_maps['x_conv5']
            score_map6 = score_maps['x_conv6']
            score_map5.indices[:, 1:] *= 2
            score_map6.indices[:, 1:] *= 4
            score_map4 = score_map4.replace_feature(torch.cat([score_map4.features, score_map5.features, score_map6.features]))
            score_map4.indices = torch.cat([score_map4.indices, score_map5.indices, score_map6.indices])
            out_score_map = self.score_map_bev_out(score_map4)

        x_conv4 = middle_feats['x_conv4']
        x_conv5 = middle_feats['x_conv5']
        x_conv6 = middle_feats['x_conv6']
        x_conv5.indices[:, 1:] *= 2
        x_conv6.indices[:, 1:] *= 4

        x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features, x_conv6.features]))
        x_conv4.indices = torch.cat([x_conv4.indices, x_conv5.indices, x_conv6.indices])

        out = self.bev_out(x_conv4)

        if self.use_seperate_radar_branch:
            out_radar_only = self.radar_branch(voxel_features, coors, batch_size)
            assert (out_radar_only.indices-out.indices).abs().sum() < .5
            out = out.replace_feature(torch.cat([out.features, out_radar_only.features], dim=-1))

        if self.use_seg_score_map:
            assert (out_score_map.indices-out.indices).abs().sum() < .5
            out = out.replace_feature(torch.cat([out.features, out_score_map.features], dim=-1))

        out = self.conv_out(out)
        if self.dense_out:
            spatial_features = self.shared_conv(out).dense()
        else:
            spatial_features = self.shared_conv(out)

        if self.return_middle_feats:
            return spatial_features, middle_feats
        else:
            return spatial_features, seg_results_dict

    def aux_loss(self, points_misc, batch_gt_instances_3d):
        all_seg_loss = dict()
        for loc, v in points_misc.items():
            if loc not in self.seg_cfg['seg_sup_locations']:
                continue
            seg_results = v['seg_results']
            voxel_dict = v['voxel_dict']
            centroids = voxel_dict['centroids']
            #batch_mask_list = voxel_dict['batch_mask_list']
            bs = len(batch_gt_instances_3d)
            total_loss = 0
            seg_targets = self.seg_heads[loc].get_targets(centroids[:,:4], batch_gt_instances_3d)
            seg_loss = self.seg_heads[loc].loss(seg_results, seg_targets)
            all_seg_loss[loc] = seg_loss
        ret = dict()
        for loc, v in all_seg_loss.items():
            for k, loss in v.items():
                ret[k+'_'+loc] = loss
        #return seg_loss
        return ret