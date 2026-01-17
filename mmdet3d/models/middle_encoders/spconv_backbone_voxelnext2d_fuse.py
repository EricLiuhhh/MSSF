from typing import List, Dict
from functools import partial
import math
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import zeros_
try:
    from torch_scatter import scatter_mean, scatter_max
except:
    import warnings
    warnings.warn("No torch_scatter found!")
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
from mmdet3d.registry import MODELS

if IS_SPCONV2_AVAILABLE:
    import spconv.pytorch as spconv
    from spconv.pytorch import SparseConvTensor, SparseSequential, SparseConv3d, SparseMaxPool3d, SparseAvgPool3d
else:
    raise NotImplementedError

from mmdet3d.models.utils import voxel_aggregation_utils
from mmdet3d.models.data_preprocessors.voxelize import VoxelizationByGridShape
from mmdet3d.structures.bbox_3d import get_proj_mat_by_coord_type
from mmdet3d.models.layers.fusion_layers.voxel_fusion import proj_to_img
from mmdet3d.models.layers import make_sparse_convmodule

def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv2d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv2d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out

@MODELS.register_module()
class SparseEncoderVoxelNeXt2DFuse(nn.Module):
    def __init__(self, 
                 sparse_shape, 
                 voxel_size,
                 point_cloud_range,
                 voxel_layer,
                 use_ext_voxel_conv_input = True,
                 keep_z = False,
                 fuse_bev = True,
                 num_point_features = 7, 
                 encoder_channels = ((32, 32, 32), (64, 64, 64, 64, 64), (128, 128, 128, 128, 128, 128, 128), (256, 256, 256, 256), (256, 256, 256, 256), (256, 256, 256, 256)),
                 optional_cfg = dict(),
                 spconv_kernel_sizes=[3, 3, 3, 3], 
                 **kwargs):
        super().__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.encoder_channels = encoder_channels
        self.sparse_shape = sparse_shape
        self.optional_cfg = optional_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.use_seperate_radar_branch = self.optional_cfg.get('use_seperate_radar_branch', False)
        self.use_seg_score_map = self.optional_cfg.get('use_seg_score_map', False)
        self.score_map_downsample_method = self.optional_cfg.get('score_map_downsample_method', 'maxpool')
        self.score_map_bev_reduce_method = self.optional_cfg.get('score_map_bev_reduce_method', 'avg')
        self.detach_score_map = self.optional_cfg.get('detach_score_map', True)
        
        block = post_act_block

        #spconv_kernel_sizes = model_cfg.get('SPCONV_KERNEL_SIZES', [3, 3, 3, 3])

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(encoder_channels[0][0], encoder_channels[0][0], norm_fn=norm_fn, indice_key='res1'),
            *[SparseBasicBlock(encoder_channels[0][i], encoder_channels[0][i+1], norm_fn=norm_fn, indice_key='res1') for i in range(len(encoder_channels[0])-1)]
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408] <- [800, 704]
            block(encoder_channels[0][-1], encoder_channels[1][0], spconv_kernel_sizes[0], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[0]//2), indice_key='spconv2', conv_type='spconv'),
            *[SparseBasicBlock(encoder_channels[1][i], encoder_channels[1][i+1], norm_fn=norm_fn, indice_key='res2') for i in range(len(encoder_channels[1])-1)],
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704] <- [400, 352]
            block(encoder_channels[1][-1], encoder_channels[2][0], spconv_kernel_sizes[1], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[1]//2), indice_key='spconv3', conv_type='spconv'),
            *[SparseBasicBlock(encoder_channels[2][i], encoder_channels[2][i+1], norm_fn=norm_fn, indice_key='res3') for i in range(len(encoder_channels[2])-1)],
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(encoder_channels[2][-1], encoder_channels[3][0], spconv_kernel_sizes[2], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[2]//2), indice_key='spconv4', conv_type='spconv'),
            *[SparseBasicBlock(encoder_channels[3][i], encoder_channels[3][i+1], norm_fn=norm_fn, indice_key='res4') for i in range(len(encoder_channels[3])-1)],
        )

        self.conv5 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(encoder_channels[3][-1], encoder_channels[4][0], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), indice_key='spconv5', conv_type='spconv'),
            *[SparseBasicBlock(encoder_channels[4][i], encoder_channels[4][i+1], norm_fn=norm_fn, indice_key='res5') for i in range(len(encoder_channels[4])-1)],
        )

        self.conv6 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(encoder_channels[4][-1], encoder_channels[5][0], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), indice_key='spconv6', conv_type='spconv'),
            *[SparseBasicBlock(encoder_channels[5][i], encoder_channels[5][i+1], norm_fn=norm_fn, indice_key='res6') for i in range(len(encoder_channels[5])-1)],
        )

        out_channels = encoder_channels[5][-1]
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, indice_key='spconv_down2'),
            norm_fn(out_channels),
            nn.ReLU(),
        )

        self.shared_conv = spconv.SparseSequential(
            spconv.SubMConv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(True),
        )

        self.num_point_features = out_channels
        self.backbone_channels = {
            'x_conv1': encoder_channels[0][-1],
            'x_conv2': encoder_channels[1][-1],
            'x_conv3': encoder_channels[2][-1],
            'x_conv4': encoder_channels[3][-1],
            'x_conv5': encoder_channels[4][-1]
        }
        self.forward_ret_dict = {}

        self.fusion_locations = self.optional_cfg.get('fusion_locations', ['x_conv1', 'x_conv2'])
        self.voxel_layer = nn.ModuleDict()
        for loc in self.fusion_locations:
            self.voxel_layer[loc] = VoxelizationByGridShape(**voxel_layer[loc])
        self.stage_num = len(self.encoder_channels)
        self.encoder_channels[0][0] = self.encoder_channels[0][0] // 2
        self.base_channels = self.encoder_channels[0][0]
        self.voxel_conv_input = make_sparse_convmodule(
                num_point_features,
                self.base_channels,
                3,
                norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                padding=1,
                indice_key='voxel_in',
                conv_type='SubMConv3d',
                order=('conv', ))
        self.voxel_encoder = dict(type='HardSimpleVFE', num_features=num_point_features)
        self.voxel_encoder = MODELS.build(self.voxel_encoder)
        self.coord_type = 'LIDAR'
        self.loc_to_index = dict()
        for i, _ in enumerate(self.encoder_channels):
            self.loc_to_index[f'x_conv{i+1}'] = i

        self.meta_dict = dict(
            #voxel_size=voxel_size,
            voxel_size=voxel_layer[self.fusion_locations[0]]['voxel_size'],
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
        self.seg_cfg = self.optional_cfg.get('seg_cfg', dict(
            seg_method='final_seg',
            feature_src=['x_conv2'],
            seg_locations=['x_conv2'],
            seg_sup_locations=['x_conv2'],
            seg_apply_location='x_conv2'
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
                #self.init_query_dim += self.encoder_channels[self.loc_to_index[self.fusion_locations[0]]][0]
                self.init_query_dim += self.base_channels
            if 'pos' in self.query_init_feat_src:
                self.init_query_dim += 3
            if fuse_bev:
                self.init_query_dim += self.base_channels
            #self.init_query_dim = 3+self.encoder_channels[self.loc_to_index[self.fusion_locations[0]]][0]+self.init_query_img_dim
            query_proj_out_channels = self.base_channels
            # if fuse_bev:
            #     query_proj_out_channels += self.base_channels
            if self.optional_cfg.get('norm_init_query', True):
                self.query_proj = nn.Sequential(
                   nn.Linear(self.init_query_dim, query_proj_out_channels),
                   nn.BatchNorm1d(query_proj_out_channels),
                   #nn.ReLU(),
                )
            else:
                self.query_proj = nn.Linear(self.init_query_dim, query_proj_out_channels)

            voxel_fuse = dict(type='VoxelFusion', 
                            img_channels=self.img_dim,
                            out_channels=self.img_dim,
                            img_levels=self.img_levels_to_init_query,
                            lateral_conv=False, 
                            fuse_out=False)
            self.voxel_fuse = MODELS.build(voxel_fuse)
        
        self.fusion_blocks = nn.ModuleDict()
        for i, loc in enumerate(self.fusion_locations):
            fusion_block_out_channels = query_proj_out_channels if loc == self.fusion_locations[0] else self.encoder_channels[i][1]
            if self.fusion_block_type == 'deform':
                fusion_block = dict(type='VoxelFusionBlock',
                                    img_cross_att = 
                                            dict(type='RadarImageCrossAttention',
                                                query_embed_dims=fusion_block_out_channels,#256
                                                value_embed_dims=self.img_dim,
                                                output_embed_dims=fusion_block_out_channels,#256
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
                            out_channels=fusion_block_out_channels,#self.img_dim,
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

        self.fuse_bev = fuse_bev
        self.keep_z = keep_z
        self.use_ext_voxel_conv_input = use_ext_voxel_conv_input
        if self.use_ext_voxel_conv_input:
            self.ext_voxel_conv_input = nn.ModuleDict()
            for loc in self.fusion_locations:
                if loc == self.fusion_locations[0]:
                    continue
                self.ext_voxel_conv_input[loc] = make_sparse_convmodule(
                                                        num_point_features,
                                                        self.base_channels,
                                                        3,
                                                        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                                                        padding=1,
                                                        indice_key=f'voxel_in_{loc}',
                                                        conv_type='SubMConv3d',
                                                        order=('conv', ))
        if fuse_bev:
            self.height_embeddings = nn.ModuleDict()
            for loc in self.fusion_locations:
                # if loc == self.fusion_locations[0]:
                #     continue
                if self.keep_z:
                    n_height = int((self.point_cloud_range[5]-self.point_cloud_range[2])/self.meta_dict['voxel_size'][2]/self.meta_dict['multi_scale_3d_strides'][0])
                else:
                    n_height = int((self.point_cloud_range[5]-self.point_cloud_range[2])/self.meta_dict['voxel_size'][2]/self.meta_dict['multi_scale_3d_strides'][loc])
                self.height_embeddings[loc] = nn.Embedding(n_height, self.encoder_channels[self.loc_to_index[loc]][0])
                zeros_(self.height_embeddings[loc].weight)

            self.voxel_bev_align_proj = nn.ModuleDict()
            for loc in self.fusion_locations:
                if loc == self.fusion_locations[0]:
                    continue
                align_out_channels = self.encoder_channels[self.loc_to_index[loc]][0]
                self.voxel_bev_align_proj[loc] = nn.Sequential(
                                                nn.Linear(self.base_channels+align_out_channels, align_out_channels),
                                                nn.BatchNorm1d(align_out_channels),
                                                #nn.ReLU(),
                                                )

    @torch.no_grad()
    def voxelize(self, points: List[Tensor], loc) -> Dict[str, Tensor]:
        """Apply voxelization to point cloud.

        Args:
            points (List[Tensor]): Point cloud in one data batch.
            data_samples: (list[:obj:`Det3DDataSample`]): The annotation data
                of every samples. Add voxel-wise annotation for segmentation.

        Returns:
            Dict[str, Tensor]: Voxelization information.

            - voxels (Tensor): Features of voxels, shape is MxNxC for hard
              voxelization, NxC for dynamic voxelization.
            - coors (Tensor): Coordinates of voxels, shape is Nx(1+NDim),
              where 1 represents the batch index.
            - num_points (Tensor, optional): Number of points in each voxel.
            - voxel_centers (Tensor, optional): Centers of voxels.
        """

        voxel_dict = dict()
        voxels, coors, num_points, voxel_centers = [], [], [], []
        for i, res in enumerate(points):
            res_voxels, res_coors, res_num_points = self.voxel_layer[loc](res)
            res_voxel_centers = (
                res_coors[:, [2, 1, 0]] + 0.5) * res_voxels.new_tensor(
                    self.voxel_layer[loc].voxel_size) + res_voxels.new_tensor(
                        self.voxel_layer[loc].point_cloud_range[0:3])
            res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
            voxel_centers.append(res_voxel_centers)

        voxels = torch.cat(voxels, dim=0)
        coors = torch.cat(coors, dim=0)
        num_points = torch.cat(num_points, dim=0)
        voxel_centers = torch.cat(voxel_centers, dim=0)

        voxel_dict['num_points'] = num_points
        voxel_dict['voxel_centers'] = voxel_centers
        voxel_dict['voxels'] = voxels
        voxel_dict['coors'] = coors

        return voxel_dict

    def bev_out(self, x_conv):
        features_cat = x_conv.features
        indices_cat = x_conv.indices

        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, _inv, features_cat)

        x_out = spconv.SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=x_conv.spatial_shape,
            batch_size=x_conv.batch_size
        )
        return x_out
    
    def bev_out_3d(self, x_conv):
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
        self.meta_dict['point_cloud_range']
        )

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
    
    def init_query(self, img_feats, voxel_features, voxel_data_dict, batch_input_meta):
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

            sampled_img_feats = torch.zeros([voxel_features[non_empty_inds].shape[0],self.init_query_img_dim], dtype=voxel_features.dtype, device=voxel_features.device)
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
            voxel_feats = voxel_features[non_empty_inds]
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
            assert len(non_empty_inds) == x_conv.features.shape[0]
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

    def forward(self, voxel_features: Tensor, coors: Tensor,
                batch_size: int, **kwargs):
        
        points = kwargs['points']
        imgs = kwargs.get('imgs', None)
        batch_input_metas = kwargs['batch_input_metas']
        bs = len(batch_input_metas)
        img_feats = kwargs.get('img_features', None)
        locations = list(self.meta_dict['multi_scale_3d_strides'].keys())
        if imgs is not None:
            #pts_voxel_dict = self.voxelize(points)
            img_data_dict = self.flatten_img_feats(img_feats)
            img_data_dict['imgs'] = imgs
            self.use_img = True
        else:
            self.use_img = False

        if len(self.fusion_locations) > 0:
            voxel_data_dict = self.get_voxel_centorids(points, self.fusion_locations, batch_input_metas)

        pillar_features, pillar_coords = voxel_features, coors
        if pillar_coords.shape[1] == 4: # bzyx
            pillar_coords = pillar_coords[:, [0, 2, 3]].contiguous()
        batch_size = batch_size.item() if isinstance(batch_size, Tensor) else batch_size
        input_sp_tensor = spconv.SparseConvTensor(
            features=pillar_features,
            indices=pillar_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = input_sp_tensor

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
                    for subblock_idx in range(1, len(block)):
                        x = block[subblock_idx](x)
                else:
                    x = block(x)

            if loc in self.fusion_locations:
                pts_voxel_dict = self.voxelize(points, loc)
                sparse_shape = [
                    int((self.point_cloud_range[3]-self.point_cloud_range[0])/self.meta_dict['voxel_size'][0]/self.meta_dict['multi_scale_3d_strides'][loc]),
                    int((self.point_cloud_range[4]-self.point_cloud_range[1])/self.meta_dict['voxel_size'][1]/self.meta_dict['multi_scale_3d_strides'][loc]),
                    int((self.point_cloud_range[5]-self.point_cloud_range[2])/self.meta_dict['voxel_size'][2]/self.meta_dict['multi_scale_3d_strides'][loc])]
                sparse_shape = sparse_shape[::-1]
                sparse_shape[0] += 1
                pts_voxel_features = self.voxel_encoder(pts_voxel_dict['voxels'],
                                                    pts_voxel_dict['num_points'],
                                                    pts_voxel_dict['coors'].int())
                input_sp_tensor = SparseConvTensor(pts_voxel_features, pts_voxel_dict['coors'].int(), sparse_shape, batch_size)
                    
                if loc == self.fusion_locations[0]:
                    x_3d = self.voxel_conv_input(input_sp_tensor)
                else:
                    if self.use_ext_voxel_conv_input:
                        x_3d = self.ext_voxel_conv_input[loc](input_sp_tensor)
                    else:
                        x_3d = input_sp_tensor

                if self.fuse_bev and loc != self.fusion_locations[0]:
                    feat = x.dense()
                    bev_feats = feat[x_3d.indices[:, 0].long(), :, x_3d.indices[:, 2].long(), x_3d.indices[:, 3].long()]
                    height_embeddings = self.height_embeddings[loc](x_3d.indices[:, 1].long())
                    bev_feats += height_embeddings
                    #x_3d = x_3d.replace_feature(x_3d.features + bev_feats)
                    x_3d = x_3d.replace_feature(self.voxel_bev_align_proj[loc](torch.cat([x_3d.features, bev_feats], dim=-1)))
                    

                #x_3d_coors = voxel_data_dict[loc]['centroids_coor'].int()
                #dummy_features = x_3d_coors.new_zeros((x_3d_coors.shape[0], 1)).float()
                #x_3d = SparseConvTensor(dummy_features, x_3d_coors, sparse_shape, batch_size)
                # insert fusion block
                self.get_non_empty_voxel_inds(x_3d, voxel_data_dict[loc], use_empty_voxel=True ,batch_input_metas=batch_input_metas, loc=loc)
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
                    if self.fuse_bev:
                        feat = x.dense()
                        bev_feats = feat[x_3d.indices[:, 0].long(), :, x_3d.indices[:, 2].long(), x_3d.indices[:, 3].long()]
                        height_embeddings = self.height_embeddings[loc](x_3d.indices[:, 1].long())
                        bev_feats += height_embeddings
                        init_voxel_features = torch.cat([x_3d.features, bev_feats], dim=-1)
                    else:
                        init_voxel_features = x_3d.features
                    init_query, voxel_dict = self.init_query(img_feats, init_voxel_features, voxel_data_dict[loc], batch_input_metas)
                    #init_query, voxel_dict = self.init_query(img_feats, x_3d, voxel_data_dict[loc], batch_input_metas)

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
                    updated_query, sample_locs, sample_weights = self.fusion_blocks[loc](x_3d, voxel_data_dict[loc], img_data_dict, external_query=init_query)
                else:
                    updated_query, sample_locs, sample_weights = self.fusion_blocks[loc](x_3d, voxel_data_dict[loc], img_data_dict)

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
                voxel_dict = {'centroids': non_empty_centroids, 'inds': voxel_data_dict[loc]['non_empty_inds'], 'xconv_indices':x_3d.indices, 'centroids_img': non_empty_centroids_img}

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
                new_features = self.query_voxel_fuse(x_3d, updated_query, inds, loc)
                x_3d = x_3d.replace_feature(new_features)

                if loc in self.seg_locations:
                    if self.seg_cfg['seg_method'] == 'seperate_seg':
                        if loc not in seg_results_dict:
                            if self.use_query_to_seg:
                                seg_results = self.seg_heads[loc](updated_query)
                            else:
                                seg_results = self.seg_heads[loc](x_3d.features[inds])
                            seg_results_dict[loc] = dict(seg_results=seg_results, voxel_dict=voxel_dict)

                        reordered_score = seg_results['seg_preds'].new_zeros(seg_results['seg_preds'].shape)
                        reordered_score[inds] = seg_results['seg_preds']
                        score_tensor = SparseConvTensor(reordered_score, x_3d.indices, x_3d.spatial_shape, bs)
                        middle_seg_scores[loc] = score_tensor
                
                if self.seg_cfg['seg_method'] == 'final_seg' and loc in self.seg_cfg['feature_src']:
                    if self.use_query_to_seg:
                        reordered_query = updated_query.new_zeros(updated_query.shape)
                        reordered_query[inds] = updated_query
                        query_tensor = SparseConvTensor(reordered_query, x_3d.indices, x_3d.spatial_shape, bs)
                        middle_feats_for_seg[loc] = query_tensor
                    else:
                        middle_feats_for_seg[loc] = x_3d

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
                            all_feats.append(x_3d.features[inds])
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
                    new_features = self.apply_seg_scores(x_3d, seg_preds, inds)
                    x_3d = x_3d.replace_feature(new_features)

                    if self.use_seg_score_map:
                        reordered_score = seg_results['seg_preds'].new_zeros(seg_results['seg_preds'].shape)
                        reordered_score[inds] = seg_results['seg_preds'].sigmoid().detach() if self.detach_score_map else seg_results['seg_preds'].sigmoid()
                        score_tensor = SparseConvTensor(reordered_score, x_3d.indices, x_3d.spatial_shape, bs)
                        score_maps[loc] = score_tensor
                    #return x, seg_results_dict
                # return x, {'seg_results':seg_results, 'voxel_dict':voxel_dict}

                #x = self.fusion_blocks[loc](x, voxel_data_dict[loc], img_data_dict)
            
                x_3d_bev = self.bev_out_3d(x_3d)

                # align idx
                if loc == self.fusion_locations[0]:
                    x_3d_indices_flat = x_3d_bev.indices[:, 0] * x_3d_bev.spatial_shape[0] * x_3d_bev.spatial_shape[1] + x_3d_bev.indices[:, 1] * x_3d_bev.spatial_shape[1] + x_3d_bev.indices[:, 2]
                    x_indices_flat = x.indices[:, 0] * x.spatial_shape[0] * x.spatial_shape[1] + x.indices[:, 1] * x.spatial_shape[1] + x.indices[:, 2]
                    x_3d_indices_sorted, x_3d_sorted_inds = torch.sort(x_3d_indices_flat)
                    x_indices_sorted, x_sorted_inds = torch.sort(x_indices_flat)
                    assert (x_3d_indices_sorted-x_indices_sorted).abs().sum()==0
                    x = x.replace_feature(torch.cat([x.features[x_sorted_inds], x_3d_bev.features[x_3d_sorted_inds]], dim=-1))
                    x.indices = x.indices[x_sorted_inds]

                else:
                    x = x.replace_feature(torch.cat([x.features, x_3d_bev.features]))
                    x.indices = torch.cat([x.indices, x_3d_bev.indices])
                    x = self.bev_out(x)

            if not self.fuse_after_residual:
                if i > 0:
                    for subblock_idx in range(1, len(block)):
                        x = block[subblock_idx](x)
                else:
                    x = block(x)
            middle_feats[loc] = x
        
        # x_conv1 = self.conv1(input_sp_tensor)
        # x_conv2 = self.conv2(x_conv1)
        # x_conv3 = self.conv3(x_conv2)
        # x_conv4 = self.conv4(x_conv3)
        # x_conv5 = self.conv5(x_conv4)
        # x_conv6 = self.conv6(x_conv5)

        x_conv4 = middle_feats['x_conv4']
        x_conv5 = middle_feats['x_conv5']
        x_conv6 = middle_feats['x_conv6']

        x_conv5.indices[:, 1:] *= 2
        x_conv6.indices[:, 1:] *= 4
        x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features, x_conv6.features]))
        x_conv4.indices = torch.cat([x_conv4.indices, x_conv5.indices, x_conv6.indices])

        out = self.bev_out(x_conv4)

        out = self.conv_out(out)
        out = self.shared_conv(out)
        
        return out, seg_results_dict

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