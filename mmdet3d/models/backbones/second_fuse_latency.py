# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional, Sequence, Tuple, List, Dict, Union
import time
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmengine.model import BaseModule
import torch
from torch import Tensor
from torch import nn as nn
from torch.nn.init import zeros_
import torch.nn.functional as F
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptMultiConfig
from mmdet3d.models.utils import voxel_aggregation_utils
from mmdet3d.models.data_preprocessors.voxelize import VoxelizationByGridShape
from mmdet3d.models.layers import make_sparse_convmodule
from mmdet3d.structures.bbox_3d import get_proj_mat_by_coord_type
from mmdet3d.models.layers.fusion_layers.voxel_fusion import proj_to_img
from mmdet3d.utils.avg_meter import AverageMeter

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor, SparseSequential
else:
    from mmcv.ops import SparseConvTensor, SparseSequential

@MODELS.register_module()
class SECONDFuseLatency(BaseModule):
    """Backbone network for SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
        norm_cfg (dict): Config dict of normalization layers.
        conv_cfg (dict): Config dict of convolutional layers.
    """

    def __init__(self,
                 voxel_layer, sparse_shape, num_features, 
                 in_channels: Union[int, Sequence[int]] = 128,
                 out_channels: Sequence[int] = [128, 128, 256],
                 layer_nums: Sequence[int] = [3, 5, 5],
                 layer_strides: Sequence[int] = [2, 2, 2],
                 norm_cfg: ConfigType = dict(
                     type='BN', eps=1e-3, momentum=0.01),
                 conv_cfg: ConfigType = dict(type='Conv2d', bias=False),
                 init_cfg: OptMultiConfig = None,
                 pretrained: Optional[str] = None,
                 fuse_bev: bool = False,
                 loss_seg_weight=1.0, optional_cfg=dict()) -> None:
        super(SECONDFuseLatency, self).__init__(init_cfg=init_cfg)
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        if isinstance(in_channels, int):
            in_filters = [in_channels, *out_channels[:-1]]
        elif isinstance(in_channels, (list, tuple)):
            assert len(in_channels) == len(out_channels)
            in_filters = in_channels
        else:
            raise ValueError
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []
        for i, layer_num in enumerate(layer_nums):
            block = [
                build_conv_layer(
                    conv_cfg,
                    in_filters[i],
                    out_channels[i],
                    3,
                    stride=layer_strides[i],
                    padding=1),
                build_norm_layer(norm_cfg, out_channels[i])[1],
                nn.ReLU(inplace=True),
            ]
            for j in range(layer_num):
                block.append(
                    build_conv_layer(
                        conv_cfg,
                        out_channels[i],
                        out_channels[i],
                        3,
                        padding=1))
                block.append(build_norm_layer(norm_cfg, out_channels[i])[1])
                block.append(nn.ReLU(inplace=True))

            block = nn.Sequential(*block)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        else:
            self.init_cfg = dict(type='Kaiming', layer='Conv2d')


        # fuse
        self.base_channels = 64
        self.img_dim = 256
        self.loss_seg_weight = loss_seg_weight
        self.optional_cfg = optional_cfg
        norm_cfg = dict(
                type='BN1d', eps=1e-3, momentum=0.01)
        self.voxel_size = voxel_layer['voxel_size']
        self.point_cloud_range = voxel_layer['point_cloud_range']
        self.meta_dict = dict(
            multi_scale_3d_strides={
                'x_conv1': 1
            },
            voxel_size = voxel_layer['voxel_size'],
            point_cloud_range = voxel_layer['point_cloud_range']
        )
        self.query_voxel_fuse_method = 'add'
        self.apply_seg_scores_method = 'mul'
        self.seg_cfg = {
            'seg_sup_locations': ['x_conv1']
        }
        self.coord_type = 'LIDAR'
        self.sparse_shape = sparse_shape
        self.conv_input = make_sparse_convmodule(
                num_features,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d',
                order=('conv', ))

        self.use_img = self.optional_cfg.get('use_img', True)
        self.no_pillar = self.optional_cfg.get('no_pillar', False)
        self.fusion_block_type = self.optional_cfg.get('fusion_block_type', 'deform')
        if not self.use_img:
            return
        self.voxel_layer = VoxelizationByGridShape(**voxel_layer)
        self.voxel_encoder = dict(type='HardSimpleVFE', num_features=num_features)
        self.voxel_encoder = MODELS.build(self.voxel_encoder)
        self.img_levels_to_fuse = list(range(5))
        self.fusion_blocks = nn.ModuleDict()
        if self.fusion_block_type == 'deform':
            fusion_block = dict(type='VoxelFusionBlock',
                        img_cross_att = 
                                dict(type='RadarImageCrossAttention',
                                    query_embed_dims=self.base_channels,
                                    value_embed_dims=self.img_dim,
                                    output_embed_dims=self.base_channels,
                                    deformable_attention=dict(
                                        type='MSDeformableAttention',
                                        num_levels=len(self.img_levels_to_fuse),
                                        weight_act_func='softmax'
                                        ),
                                )
                        )
            self.fusion_blocks['x_conv1'] = MODELS.build(fusion_block)
        elif self.fusion_block_type == 'simple':
            fusion_block = dict(type='SimpleVoxelFusionBlock', 
                        img_channels=self.img_dim,
                        out_channels=self.base_channels,#self.img_dim,
                        img_levels=self.img_levels_to_fuse,
                        lateral_conv=False, 
                        fuse_out=False)
            self.fusion_blocks['x_conv1'] = MODELS.build(fusion_block)
        else:
            raise NotImplementedError

        self.seg_heads = nn.ModuleDict()
        seg_head = dict(type='ForegroundSegmentationHead',
            in_channels=self.base_channels)
        self.seg_heads['x_conv1'] = MODELS.build(seg_head)
        # self.conv_out = SparseSequential(
        #     SparseConv3d(self.base_channels, self.base_channels, (3, 1, 1), stride=(2, 1, 1), padding=0,
        #                         bias=False, indice_key='spconv_down2'),
        #     build_norm_layer(norm_cfg, 32)[1],
        #     nn.ReLU(),
        # )
        # self.bev_conv = nn.Sequential(
        #     nn.Conv2d((self.sparse_shape[0]-1)*self.base_channels, self.out_channels, 3, stride=1, padding=1, bias=True),
        #     nn.BatchNorm2d(self.out_channels),
        #     nn.ReLU(True),
        # )

        self.query_init_feat_src = ['img', 'voxel', 'pos']
        self.img_levels_to_init_query = [0]
        self.init_query_img_dim = self.img_dim*len(self.img_levels_to_init_query)
        self.init_query_dim = self.init_query_img_dim + self.base_channels + 3
        self.query_proj = nn.Sequential(
                   nn.Linear(self.init_query_dim, self.base_channels),
                   nn.BatchNorm1d(self.base_channels),
                   #nn.ReLU(),
                )
    
        voxel_fuse = dict(type='VoxelFusion', 
                        img_channels=self.img_dim,
                        out_channels=self.img_dim,
                        img_levels=self.img_levels_to_init_query,
                        lateral_conv=False, 
                        fuse_out=False)
        self.voxel_fuse = MODELS.build(voxel_fuse)

        self.fuse_bev = fuse_bev
        if fuse_bev:
            self.height_embeddings = nn.Embedding(sparse_shape[0], out_channels[0])
            zeros_(self.height_embeddings.weight)

        self.fusion_meter = AverageMeter()
        self.pts_feats_meter = AverageMeter()
    
    @torch.no_grad()
    def voxelize(self, points: List[Tensor]) -> Dict[str, Tensor]:
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
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            res_voxel_centers = (
                res_coors[:, [2, 1, 0]] + 0.5) * res_voxels.new_tensor(
                    self.voxel_layer.voxel_size) + res_voxels.new_tensor(
                        self.voxel_layer.point_cloud_range[0:3])
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
    
    def forward(self,
                feat: Tensor,
                **kwargs) -> Tensor:
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        batch_size = feat.shape[0]
        points = kwargs['points']
        imgs = kwargs.get('imgs', None)
        batch_input_metas = kwargs['batch_input_metas']
        bs = len(batch_input_metas)
        img_feats = kwargs.get('img_features', None)
        locations = list(self.meta_dict['multi_scale_3d_strides'].keys())
        if imgs is not None:
            pts_voxel_dict = self.voxelize(points)
            img_data_dict = self.flatten_img_feats(img_feats)
            img_data_dict['imgs'] = imgs
            self.use_img = True
        else:
            self.use_img = False

        seg_results_dict = {}

        outs = []
        fusion_total = 0.0
        pts_feats_total = 0.0
        for i in range(len(self.blocks)):
            start = time.perf_counter()
            feat = self.blocks[i](feat)
            end = time.perf_counter()
            pts_feats_total += end-start

            
            if self.use_img and i==0:
                voxel_data_dict = self.get_voxel_centorids(points, locations, batch_input_metas)
                pts_voxel_features = self.voxel_encoder(pts_voxel_dict['voxels'],
                                                    pts_voxel_dict['num_points'],
                                                    pts_voxel_dict['coors'].int())
                input_sp_tensor = SparseConvTensor(pts_voxel_features, pts_voxel_dict['coors'].int(),
                                                self.sparse_shape, batch_size)
                x = self.conv_input(input_sp_tensor)

                start = time.perf_counter()
                # fuse with pillar BEV feature
                if self.fuse_bev:
                    bev_feats = feat[x.indices[:, 0].long(), :, x.indices[:, 2].long(), x.indices[:, 3].long()]
                    height_embeddings = self.height_embeddings(x.indices[:, 1].long())
                    bev_feats += height_embeddings
                    x = x.replace_feature(x.features + bev_feats)

                loc = 'x_conv1'
                self.get_non_empty_voxel_inds(x, voxel_data_dict[loc], use_empty_voxel=True ,batch_input_metas=batch_input_metas, loc=loc)
                init_query, voxel_dict = self.init_query(img_feats, x, voxel_data_dict[loc], batch_input_metas)
                
                non_empty_coors = voxel_data_dict[loc]['centroids_coor'][voxel_data_dict[loc]['non_empty_mask']]
                bev_query = feat[non_empty_coors[:, 0], :, non_empty_coors[:, 2], non_empty_coors[:, 3]]
                init_query = init_query + bev_query
                start = time.perf_counter()
                updated_query, sample_locs, _ = self.fusion_blocks[loc](x, voxel_data_dict[loc], img_data_dict, external_query=init_query)
                end = time.perf_counter()
                print(f'fusion block in backbone time: {end-start}')
                inds = voxel_data_dict[loc]['non_empty_inds']
                new_features = self.query_voxel_fuse(x, updated_query, inds, loc)
                x = x.replace_feature(new_features)
                seg_results = self.seg_heads[loc](x.features[inds])
                seg_results_dict[loc] = dict(seg_results=seg_results, voxel_dict=voxel_dict)
                seg_preds = seg_results_dict[loc]['seg_results']['seg_preds']
                new_features = self.apply_seg_scores(x, seg_preds, inds)
                x = x.replace_feature(new_features)
                x = self.bev_out(x)
                x = x.dense()
                end = time.perf_counter()
                fusion_total += end-start

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
                            save_path = f"./vis_results/seg_result_ppmm_tj4d_{key}"
                            if not os.path.exists(save_path):
                                os.mkdir(save_path)
                            a.cpu().numpy().astype(np.float32).tofile(f"{save_path}/{batch_input_metas[b]['lidar_path'].split('/')[-1]}")
                #x = self.bev_conv(x)

                feat = torch.cat([feat, x], dim=1)

            outs.append(feat)

        self.fusion_meter.update(fusion_total)
        self.pts_feats_meter.update(pts_feats_total)
        return tuple(outs), seg_results_dict


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
                ret[k+'_'+loc] = loss# * self.loss_seg_weight
        #return seg_loss
        return ret
