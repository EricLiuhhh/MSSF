from typing import List, Union, Optional
import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmdet3d.utils import OptConfigType, OptMultiConfig
from mmdet3d.registry import MODELS
from .voxel_fusion import VoxelFusion

@MODELS.register_module()
class SimpleVoxelFusionBlock(VoxelFusion):
    def __init__(self, img_channels: Union[List[int], int], out_channels: int, pts_channels: int = 128, mid_channels: int = 128, img_levels: Union[List[int], int] = 3, coord_type: str = 'LIDAR', conv_cfg: OptConfigType = None, norm_cfg: OptConfigType = None, act_cfg: OptConfigType = None, init_cfg: OptMultiConfig = None, activate_out: bool = True, fuse_out: bool = False, dropout_ratio: Union[int, float] = 0, aligned: bool = True, align_corners: bool = True, padding_mode: str = 'zeros', lateral_conv: bool = True, out_method: str = 'proj') -> None:
        super().__init__(img_channels, out_channels, pts_channels, mid_channels, img_levels, coord_type, conv_cfg, norm_cfg, act_cfg, init_cfg, activate_out, fuse_out, dropout_ratio, aligned, align_corners, padding_mode, lateral_conv)
        self.out_method = out_method
        if out_method == 'proj':
            pass
        elif out_method == 'mean':
            self.img_transform = nn.Sequential(
                    nn.Linear(img_channels, out_channels),
                    nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                )
        else:
            raise NotImplementedError

    def forward(self, x, voxel_data_dict, img_data_dict, **kwargs):
        img_feats = img_data_dict['img_feats']
        bs = img_feats[0].shape[0]
        
        centroids_coor = voxel_data_dict['centroids_coor']
        centroids = voxel_data_dict['centroids']
        centroids_img = voxel_data_dict['centroids_img']
        non_empty_inds = voxel_data_dict['non_empty_inds']
        non_empty_mask = voxel_data_dict['non_empty_mask']
        non_empty_centroids = centroids[non_empty_mask]
        non_empty_centroids_img = centroids_img[non_empty_mask]
        non_empty_centroids_coor = centroids_coor[non_empty_mask]

        centroids_list = []
        centroids_img_list = []
        batch_mask_list = []

        sampled_img_feats = torch.zeros([x.features[non_empty_inds].shape[0],sum(self.img_channels)], dtype=x.features.dtype, device=x.features.device)
        sample_locs_all = torch.zeros([x.features[non_empty_inds].shape[0],len(self.img_levels),2], dtype=x.features.dtype, device=x.features.device)
        sample_weights_all = torch.ones([x.features[non_empty_inds].shape[0],len(self.img_levels),1], dtype=x.features.dtype, device=x.features.device)
        for i in range(bs):
            batch_mask = (non_empty_centroids_coor[:, 0].int() == i)
            centroids_list.append(non_empty_centroids[batch_mask][...,1:])
            centroids_img_list.append(non_empty_centroids_img[batch_mask][...,1:])
            batch_mask_list.append(batch_mask)
        sampled_img_feats_list = super().forward(img_feats, centroids_img_list)
        for i in range(bs):
            sampled_img_feats[batch_mask_list[i]] = sampled_img_feats_list[i]
            sample_locs_all[batch_mask_list[i]] = centroids_img_list[i][:, None, :].repeat(1, len(self.img_levels),1)
        
        if self.out_method == 'proj':
            sampled_img_feats = self.img_transform(sampled_img_feats)
        elif self.out_method == 'mean':
            sampled_img_feats = sampled_img_feats.view([sampled_img_feats.shape[0], len(self.img_levels), -1])
            sampled_img_feats = sampled_img_feats.mean(1)
            sampled_img_feats = self.img_transform(sampled_img_feats)
        else:
            raise NotImplementedError
        return sampled_img_feats, sample_locs_all, sample_weights_all