# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, List, Tuple

import numpy as np
import torch
from mmdet.models.utils import multi_apply
from mmdet.utils.memory import cast_tensor_type
from mmengine import ConfigDict
from mmengine.runner import amp
from mmengine.structures import InstanceData
from torch import Tensor
from torch import nn as nn

from mmdet3d.models.task_modules import PseudoSampler
from mmdet3d.models.test_time_augs import merge_aug_bboxes_3d
from mmdet3d.models.layers import nms_bev, nms_normal_bev
from mmdet3d.structures import limit_period, xywhr2xyxyr
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.utils.typing_utils import (ConfigType, InstanceList,
                                        OptConfigType, OptInstanceList)

from .anchor3d_head import Anchor3DHead


@MODELS.register_module()
class PointPillarHead(Anchor3DHead):
    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 feat_channels: int = 256,
                 use_direction_classifier: bool = True,
                 anchor_generator: ConfigType = dict(
                     type='Anchor3DRangeGenerator',
                     range=[0, -39.68, -1.78, 69.12, 39.68, -1.78],
                     strides=[2],
                     sizes=[[3.9, 1.6, 1.56]],
                     rotations=[0, 1.57],
                     custom_values=[],
                     reshape_out=False),
                 assigner_per_size: bool = False,
                 assign_per_class: bool = False,
                 diff_rad_by_sin: bool = True,
                 dir_offset: float = -np.pi / 2,
                 dir_limit_offset: int = 0,
                 bbox_coder: ConfigType = dict(type='DeltaXYZWLHRBBoxCoder'),
                 loss_cls: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(
                     type='mmdet.SmoothL1Loss',
                     beta=1.0 / 9.0,
                     loss_weight=2.0),
                 loss_dir: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss', loss_weight=0.2),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptConfigType = None) -> None:
        super().__init__(num_classes, in_channels, feat_channels,
                         use_direction_classifier, anchor_generator,
                         assigner_per_size, assign_per_class, diff_rad_by_sin,
                         dir_offset, dir_limit_offset, bbox_coder, loss_cls,
                         loss_bbox, loss_dir, train_cfg, test_cfg, init_cfg)
        
    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                dir_cls_pred_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                input_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                **kwargs) -> InstanceData:
        """Transform a single points sample's features extracted from the head
        into bbox results.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single point cloud sample, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single point cloud sample, each item
                has shape (num_priors * C, H, W).
            dir_cls_pred_list (list[Tensor]): Predictions of direction class
                from all scale levels of a single point cloud sample, each
                item has shape (num_priors * 2, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            input_meta (dict): Contain point clouds and image meta info.
            cfg (:obj:`ConfigDict`): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_priors)
        mlvl_bboxes = []
        mlvl_max_scores = []
        mlvl_scores = []
        mlvl_label_pred = []
        mlvl_dir_scores = []
        mlvl_cls_score = []
        for cls_score, bbox_pred, dir_cls_pred, priors in zip(
                cls_score_list, bbox_pred_list, dir_cls_pred_list,
                mlvl_priors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            assert cls_score.size()[-2:] == dir_cls_pred.size()[-2:]
            dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]

            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.num_classes)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2,
                                          0).reshape(-1, self.box_code_size)

            nms_pre = cfg.get('nms_pre', -1)
            if self.use_sigmoid_cls:
                max_scores, pred_labels = scores.max(dim=1)
            else:
                max_scores, pred_labels = scores[:, :-1].max(dim=1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                topk_scores, topk_inds = max_scores.topk(nms_pre)
                priors = priors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                max_scores = topk_scores
                scores = scores[topk_inds, :]
                dir_cls_score = dir_cls_score[topk_inds]
                pred_labels = pred_labels[topk_inds]

            bboxes = self.bbox_coder.decode(priors, bbox_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_max_scores.append(max_scores)
            mlvl_scores.append(scores)
            mlvl_label_pred.append(pred_labels)
            mlvl_dir_scores.append(dir_cls_score)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_bboxes_for_nms = xywhr2xyxyr(input_meta['box_type_3d'](
            mlvl_bboxes, box_dim=self.box_code_size).bev)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_max_scores = torch.cat(mlvl_max_scores)
        mlvl_label_pred = torch.cat(mlvl_label_pred)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)

        if self.use_sigmoid_cls:
            # Add a dummy background class to the front when using sigmoid
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        score_thr = cfg.get('score_thr', 0)
        # results = box3d_multiclass_nms(mlvl_bboxes, mlvl_bboxes_for_nms,
        #                                mlvl_scores, score_thr, cfg.max_num,
        #                                cfg, mlvl_dir_scores)
        results = self.class_agnostic_nms(mlvl_bboxes, mlvl_bboxes_for_nms, mlvl_max_scores, mlvl_label_pred, mlvl_scores, mlvl_dir_scores, score_thr, cfg, input_meta)
        # bboxes, scores, labels, dir_scores = results
        # if bboxes.shape[0] > 0:
        #     dir_rot = limit_period(bboxes[..., 6] - self.dir_offset,
        #                            self.dir_limit_offset, np.pi)
        #     bboxes[..., 6] = (
        #         dir_rot + self.dir_offset +
        #         np.pi * dir_scores.to(bboxes.dtype))
        # bboxes = input_meta['box_type_3d'](bboxes, box_dim=self.box_code_size)
        # results = InstanceData()
        # results.bboxes_3d = bboxes
        # results.scores_3d = scores
        # results.labels_3d = labels

        return results
    
    def class_agnostic_nms(self, mlvl_bboxes: Tensor,
                           mlvl_bboxes_for_nms: Tensor,
                           mlvl_max_scores: Tensor, mlvl_label_pred: Tensor,
                           mlvl_cls_score: Tensor, mlvl_dir_scores: Tensor,
                           score_thr: int, cfg: ConfigDict,
                           input_meta: dict) -> Dict:
        """Class agnostic nms for single batch.

        Args:
            mlvl_bboxes (torch.Tensor): Bboxes from Multi-level.
            mlvl_bboxes_for_nms (torch.Tensor): Bboxes for nms
                (bev or minmax boxes) from Multi-level.
            mlvl_max_scores (torch.Tensor): Max scores of Multi-level bbox.
            mlvl_label_pred (torch.Tensor): Class predictions
                of Multi-level bbox.
            mlvl_cls_score (torch.Tensor): Class scores of
                Multi-level bbox.
            mlvl_dir_scores (torch.Tensor): Direction scores of
                Multi-level bbox.
            score_thr (int): Score threshold.
            cfg (:obj:`ConfigDict`): Training or testing config.
            input_meta (dict): Contain pcd and img's meta info.

        Returns:
            dict: Predictions of single batch. Contain the keys:

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Predicted 3d bboxes.
            - scores_3d (torch.Tensor): Score of each bbox.
            - labels_3d (torch.Tensor): Label of each bbox.
            - cls_preds (torch.Tensor): Class score of each bbox.
        """
        bboxes = []
        scores = []
        labels = []
        dir_scores = []
        cls_scores = []
        score_thr_inds = mlvl_max_scores > score_thr
        _scores = mlvl_max_scores[score_thr_inds]
        _bboxes_for_nms = mlvl_bboxes_for_nms[score_thr_inds, :]
        if cfg.use_rotate_nms:
            nms_func = nms_bev
        else:
            nms_func = nms_normal_bev
        selected = nms_func(_bboxes_for_nms, _scores, cfg.nms_thr)

        _mlvl_bboxes = mlvl_bboxes[score_thr_inds, :]
        _mlvl_dir_scores = mlvl_dir_scores[score_thr_inds]
        _mlvl_label_pred = mlvl_label_pred[score_thr_inds]
        _mlvl_cls_score = mlvl_cls_score[score_thr_inds]

        if len(selected) > 0:
            bboxes.append(_mlvl_bboxes[selected])
            scores.append(_scores[selected])
            labels.append(_mlvl_label_pred[selected])
            cls_scores.append(_mlvl_cls_score[selected])
            dir_scores.append(_mlvl_dir_scores[selected])
            dir_rot = limit_period(bboxes[-1][..., 6] - self.dir_offset,
                                   self.dir_limit_offset, np.pi)
            bboxes[-1][..., 6] = (
                dir_rot + self.dir_offset +
                np.pi * dir_scores[-1].to(bboxes[-1].dtype))

        if bboxes:
            bboxes = torch.cat(bboxes, dim=0)
            scores = torch.cat(scores, dim=0)
            cls_scores = torch.cat(cls_scores, dim=0)
            labels = torch.cat(labels, dim=0)
            if bboxes.shape[0] > cfg.max_num:
                _, inds = scores.sort(descending=True)
                inds = inds[:cfg.max_num]
                bboxes = bboxes[inds, :]
                labels = labels[inds]
                scores = scores[inds]
                cls_scores = cls_scores[inds]
            bboxes = input_meta['box_type_3d'](
                bboxes, box_dim=self.box_code_size)
            result = InstanceData()
            result.bboxes_3d = bboxes
            result.scores_3d = scores
            result.labels_3d = labels
            result.cls_preds = cls_scores
            return result
        else:
            result = InstanceData()
            result.bboxes_3d = input_meta['box_type_3d'](
                mlvl_bboxes.new_zeros([0, self.box_code_size]),
                box_dim=self.box_code_size)
            result.scores_3d = mlvl_bboxes.new_zeros([0])
            result.labels_3d = mlvl_bboxes.new_zeros([0])
            result.cls_preds = mlvl_bboxes.new_zeros(
                [0, mlvl_cls_score.shape[-1]])
            return result