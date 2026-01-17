# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple

import torch
from mmdet.models.task_modules import BaseBBoxCoder
from torch import Tensor

from mmdet3d.registry import TASK_UTILS

from .centerpoint_bbox_coders import CenterPointBBoxCoder

@TASK_UTILS.register_module()
class VoxelNeXtBBoxCoder(CenterPointBBoxCoder):
    """Bbox coder for VoxelNeXt.

    Args:
        pc_range (list[float]): Range of point cloud.
        out_size_factor (int): Downsample factor of the model.
        voxel_size (list[float]): Size of voxel.
        post_center_range (list[float], optional): Limit of the center.
            Default: None.
        max_num (int, optional): Max number to be kept. Default: 100.
        score_threshold (float, optional): Threshold to filter boxes
            based on score. Default: None.
        code_size (int, optional): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range: List[float],
                 out_size_factor: int,
                 voxel_size: List[float],
                 post_center_range: Optional[List[float]] = None,
                 max_num: int = 100,
                 score_threshold: Optional[float] = None,
                 code_size: int = 9) -> None:
        super(VoxelNeXtBBoxCoder, self).__init__(pc_range, 
                                                 out_size_factor, 
                                                 voxel_size,
                                                 post_center_range,
                                                 max_num,
                                                 score_threshold,
                                                 code_size)

    def _gather_feat(self,
                     feats: Tensor,
                     inds: Tensor,
                     batch_size, 
                     batch_idx,
                     feat_masks: Optional[Tensor] = None) -> Tensor:
        """Given feats and indexes, returns the gathered feats.

        Args:
            feats (torch.Tensor): Features to be transposed and gathered
                with the shape of [B, 2, W, H].
            inds (torch.Tensor): Indexes with the shape of [B, N].
            feat_masks (torch.Tensor, optional): Mask of the feats.
                Default: None.

        Returns:
            torch.Tensor: Gathered feats.
        """
        feats_list = []
        dim = feats.size(-1)
        _inds = inds.unsqueeze(-1).expand(inds.size(0), inds.size(1), dim)

        for bs_idx in range(batch_size):
            batch_inds = batch_idx==bs_idx
            feat = feats[batch_inds]
            feats_list.append(feat.gather(0, _inds[bs_idx]))
        feats = torch.stack(feats_list)
        return feats
    

    def _topk(self, scores, batch_size, batch_idx, obj, K=40, nuscenes=False) -> Tuple[Tensor]:
        """Get indexes based on scores.

        Args:
            scores (torch.Tensor): scores with the shape of [B, N, W, H].
            K (int, optional): Number to be kept. Defaults to 80.

        Returns:
            tuple[torch.Tensor]
                torch.Tensor: Selected scores with the shape of [B, K].
                torch.Tensor: Selected indexes with the shape of [B, K].
                torch.Tensor: Selected classes with the shape of [B, K].
                torch.Tensor: Selected y coord with the shape of [B, K].
                torch.Tensor: Selected x coord with the shape of [B, K].
        """
        # scores: (N, num_classes)
        topk_score_list = []
        topk_inds_list = []
        topk_classes_list = []

        for bs_idx in range(batch_size):
            batch_inds = batch_idx==bs_idx
            if obj.shape[-1] == 1 and not nuscenes:
                score = scores[batch_inds].permute(1, 0)
                topk_scores, topk_inds = torch.topk(score, K)
                topk_score, topk_ind = torch.topk(obj[topk_inds.view(-1)].squeeze(-1), K) #torch.topk(topk_scores.view(-1), K)
            else:
                score = obj[batch_inds].permute(1, 0)
                topk_scores, topk_inds = torch.topk(score, min(K, score.shape[-1]))
                topk_score, topk_ind = torch.topk(topk_scores.view(-1), min(K, topk_scores.view(-1).shape[-1]))
                #topk_score, topk_ind = torch.topk(score.reshape(-1), K)

            topk_classes = (topk_ind // K).int()
            topk_inds = topk_inds.view(-1).gather(0, topk_ind)
            #print('topk_inds', topk_inds)

            if not obj is None and obj.shape[-1] == 1:
                topk_score_list.append(obj[batch_inds][topk_inds])
            else:
                topk_score_list.append(topk_score)
            topk_inds_list.append(topk_inds)
            topk_classes_list.append(topk_classes)

        topk_score = torch.stack(topk_score_list)
        topk_inds = torch.stack(topk_inds_list)
        topk_classes = torch.stack(topk_classes_list)

        return topk_score, topk_inds, topk_classes

    def _transpose_and_gather_feat(self, feat: Tensor, ind: Tensor) -> Tensor:
        """Given feats and indexes, returns the transposed and gathered feats.

        Args:
            feat (torch.Tensor): Features to be transposed and gathered
                with the shape of [B, 2, W, H].
            ind (torch.Tensor): Indexes with the shape of [B, N].

        Returns:
            torch.Tensor: Transposed and gathered feats.
        """
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def encode(self):
        pass

    def decode(self,
               heat: Tensor,
               rot_sine: Tensor,
               rot_cosine: Tensor,
               hei: Tensor,
               dim: Tensor,
               vel: Tensor,
               voxel_infos: dict,
               reg: Optional[Tensor] = None,
               task_id: int = -1) -> List[Dict[str, Tensor]]:
        """Decode bboxes.

        Args:
            heat (torch.Tensor): Heatmap with the shape of [B, N].
            rot_sine (torch.Tensor): Sine of rotation with the shape of
                [B, 1].
            rot_cosine (torch.Tensor): Cosine of rotation with the shape of
                [B, 1].
            hei (torch.Tensor): Height of the boxes with the shape
                of [B, 1].
            dim (torch.Tensor): Dim of the boxes with the shape of
                [B, 3].
            vel (torch.Tensor): Velocity with the shape of [B, 1].
            reg (torch.Tensor, optional): Regression value of the boxes in
                2D with the shape of [B, 2]. Default: None.
            task_id (int, optional): Index of task. Default: -1.

        Returns:
            list[dict]: Decoded boxes.
        """
        #batch, cat, _, _ = heat.size()
        voxel_indices = voxel_infos['voxel_indices']
        batch_idx = voxel_indices[:, 0]
        spatial_indices = voxel_indices[:, 1:]
        batch = voxel_infos['batch_size']

        scores, inds, clses = self._topk(None, batch, batch_idx, heat, K=self.max_num, nuscenes=True)
        #scores, inds, clses, ys, xs = self._topk(heat, K=self.max_num)
        spatial_indices = self._gather_feat(spatial_indices, inds, batch, batch_idx)
        if reg is not None:
            reg = self._gather_feat(reg, inds, batch, batch_idx)
            xs = (spatial_indices[:, :, -1:] + reg[:, :, 0:1]) * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
            ys = (spatial_indices[:, :, -2:-1] + reg[:, :, 1:2]) * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]
        else:
            xs = (spatial_indices[:, :, -1:] + 0.5) * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
            ys = (spatial_indices[:, :, -2:-1] + 0.5) * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]
        
        # rotation value and direction label
        rot_sine = self._gather_feat(rot_sine, inds, batch, batch_idx)
        rot_sine = rot_sine.view(batch, self.max_num, 1)

        rot_cosine = self._gather_feat(rot_cosine, inds, batch, batch_idx)
        rot_cosine = rot_cosine.view(batch, self.max_num, 1)
        rot = torch.atan2(rot_sine, rot_cosine)

        # height in the bev
        hei = self._gather_feat(hei, inds, batch, batch_idx)
        hei = hei.view(batch, self.max_num, 1)

        # dim of the box
        dim = self._gather_feat(dim, inds, batch, batch_idx)
        dim = dim.view(batch, self.max_num, 3)

        # class label
        clses = clses.view(batch, self.max_num).float()
        scores = scores.view(batch, self.max_num)

        if vel is None:  # KITTI FORMAT
            final_box_preds = torch.cat([xs, ys, hei, dim, rot], dim=2)
        else:  # exist velocity, nuscene format
            vel = self._transpose_and_gather_feat(vel, inds)
            vel = vel.view(batch, self.max_num, 2)
            final_box_preds = torch.cat([xs, ys, hei, dim, rot, vel], dim=2)

        final_scores = scores
        final_preds = clses

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=heat.device)
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(2)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(2)

            predictions_dicts = []
            for i in range(batch):
                cmask = mask[i, :]
                if self.score_threshold:
                    cmask &= thresh_mask[i]

                boxes3d = final_box_preds[i, cmask]
                scores = final_scores[i, cmask]
                labels = final_preds[i, cmask]

                # mao_c
                corr_inds = inds[i, cmask]
                predictions_dict = {
                    'bboxes': boxes3d,
                    'scores': scores,
                    'labels': labels,
                    'inds': corr_inds,
                }

                predictions_dicts.append(predictions_dict)
        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')

        return predictions_dicts
