from typing import List, Union, Optional
import torch
from mmengine.model import BaseModule
from mmdet3d.utils import OptConfigType, OptMultiConfig
from mmdet3d.registry import MODELS

@MODELS.register_module()
class VoxelFusionBlock(BaseModule):
    def __init__(self,
                 img_cross_att: Optional[dict] = 
                    dict(type='RadarImageCrossAttention',
                        query_embed_dims=256,
                        value_embed_dims=256,
                        output_embed_dims=256,
                        deformable_attention=dict(
                            type='MSDeformableAttention',
                            num_levels=4
                            ),
                    ),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)
        self.img_cross_att = MODELS.build(img_cross_att)

    def forward(self, x, voxel_data_dict, img_data_dict, external_query=None):
        # x = voxel_data_dict['sparse_tensor']
        centroids = voxel_data_dict['centroids']
        centroids_img = voxel_data_dict['centroids_img']
        centroids_coor = voxel_data_dict['centroids_coor']
        non_empty_inds = voxel_data_dict['non_empty_inds']
        non_empty_mask = voxel_data_dict['non_empty_mask']
        
        # img_feats = img_data_dict['img_feats']
        feat_flatten = img_data_dict['feat_flatten']
        spatial_shapes = img_data_dict['spatial_shapes']
        level_start_index = img_data_dict['level_start_index']

        # extract feature from image
        non_empty_coor = centroids_coor[non_empty_mask]
        bs = feat_flatten.shape[0]
        # fix bug: batch index is not continuous because we use empty voxel
        #updated_query = []
        #sample_locs_list = []
        updated_query = torch.zeros(x.features[non_empty_inds].shape, dtype=x.features.dtype, device=x.features.device)
        sample_locs_all = torch.zeros([x.features[non_empty_inds].shape[0],4*8*len(level_start_index),2], dtype=x.features.dtype, device=x.features.device)
        sample_weights_all = torch.zeros([x.features[non_empty_inds].shape[0],4*8*len(level_start_index),1], dtype=x.features.dtype, device=x.features.device)
        for i in range(bs):
            batch_mask = (non_empty_coor[:, 0] == i)
            inds = non_empty_inds[batch_mask]
            if external_query is not None:
                non_empty_voxel_feats = external_query[batch_mask]
            else:
                non_empty_voxel_feats = x.features[inds]
            ref_pts = centroids_img[batch_mask][:, 1:][None, :, None, :].repeat(1, 1, len(spatial_shapes), 1)
            
            ms_feats, sample_locs, sample_weights = self.img_cross_att(
                            query=non_empty_voxel_feats.unsqueeze(0),
                            key=feat_flatten[i].unsqueeze(0),
                            value=feat_flatten[i].unsqueeze(0),
                            reference_points_cams=ref_pts,
                            spatial_shapes=spatial_shapes,
                            level_start_index=level_start_index
                        )
            #updated_query[inds] = ms_feats[0]
            #sample_locs_all[inds] = sample_locs[0]
            updated_query[batch_mask] = ms_feats[0]
            sample_locs_all[batch_mask] = sample_locs[0]
            sample_weights_all[batch_mask] = sample_weights[0]

            #updated_query.append(ms_feats[0])
            #sample_locs_list.append(sample_locs)
        return updated_query, sample_locs_all, sample_weights_all
        #return torch.cat(updated_query, dim=0), sample_locs_list