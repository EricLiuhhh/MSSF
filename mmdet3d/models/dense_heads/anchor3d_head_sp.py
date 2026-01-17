# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch import Tensor
from torch import nn as nn
from mmcv.ops import points_in_boxes_part
from mmdet.models.utils import select_single_mlvl
from mmengine.config import ConfigDict
from mmdet.models.utils import multi_apply
from mmdet.utils.memory import cast_tensor_type
from mmengine.runner import amp


from mmdet3d.models.task_modules import PseudoSampler
from mmdet3d.models.test_time_augs import merge_aug_bboxes_3d
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.utils.typing_utils import (ConfigType, InstanceList,
                                        OptConfigType, OptInstanceList)
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor, SparseSequential, SubMConv2d
else:
    from mmcv.ops import SparseConvTensor, SparseSequential
from .base_3d_dense_head import Base3DDenseHead
from .train_mixins import AnchorTrainMixin


@MODELS.register_module()
class Anchor3DHeadSP(Base3DDenseHead, AnchorTrainMixin):
    """Anchor-based head for SECOND/PointPillars/MVXNet/PartA2.

    Args:
        num_classes (int): Number of classes.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of channels of the feature map.
        use_direction_classifier (bool): Whether to add a direction classifier.
        anchor_generator(dict): Config dict of anchor generator.
        assigner_per_size (bool): Whether to do assignment for each separate
            anchor size.
        assign_per_class (bool): Whether to do assignment for each class.
        diff_rad_by_sin (bool): Whether to change the difference into sin
            difference for box regression loss.
        dir_offset (float | int): The offset of BEV rotation angles.
            (TODO: may be moved into box coder)
        dir_limit_offset (float | int): The limited range of BEV
            rotation angles. (TODO: may be moved into box coder)
        bbox_coder (dict): Config dict of box coders.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_dir (dict): Config of direction classifier loss.
        train_cfg (dict): Train configs.
        test_cfg (dict): Test configs.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

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
                 init_cfg: OptConfigType = None,
                 use_sparse=True) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.diff_rad_by_sin = diff_rad_by_sin
        self.use_direction_classifier = use_direction_classifier
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.assigner_per_size = assigner_per_size
        self.assign_per_class = assign_per_class
        self.dir_offset = dir_offset
        self.dir_limit_offset = dir_limit_offset
        self.use_sparse = use_sparse
        warnings.warn(
            'dir_offset and dir_limit_offset will be depressed and be '
            'incorporated into box coder in the future')

        # build anchor generator
        self.prior_generator = TASK_UTILS.build(anchor_generator)
        # In 3D detection, the anchor stride is connected with anchor size
        self.num_anchors = self.prior_generator.num_base_anchors
        # build box coder
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.box_code_size = self.bbox_coder.code_size

        # build loss function
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in [
            'mmdet.FocalLoss', 'mmdet.GHMC'
        ]
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_dir = MODELS.build(loss_dir)

        self._init_layers()
        self._init_assigner_sampler()

        if init_cfg is None:
            self.init_cfg = dict(
                type='Normal',
                layer='Conv2d',
                std=0.01,
                override=dict(
                    type='Normal', name='conv_cls', std=0.01, bias_prob=0.01))

    def _init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        if self.train_cfg is None:
            return

        if self.sampling:
            self.bbox_sampler = TASK_UTILS.build(self.train_cfg.sampler)
        else:
            self.bbox_sampler = PseudoSampler()
        if isinstance(self.train_cfg.assigner, dict):
            self.bbox_assigner = TASK_UTILS.build(self.train_cfg.assigner)
        elif isinstance(self.train_cfg.assigner, list):
            self.bbox_assigner = [
                TASK_UTILS.build(res) for res in self.train_cfg.assigner
            ]

    def _init_layers(self):
        """Initialize neural network layers of the head."""
        self.cls_out_channels = self.num_anchors * self.num_classes
        if not self.use_sparse:
            self.conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
            self.conv_reg = nn.Conv2d(self.feat_channels,
                                    self.num_anchors * self.box_code_size, 1)
            if self.use_direction_classifier:
                self.conv_dir_cls = nn.Conv2d(self.feat_channels,
                                            self.num_anchors * 2, 1)
        else:
            self.conv_cls = SubMConv2d(self.feat_channels, self.cls_out_channels, 1, bias=True)
            self.conv_reg = SubMConv2d(self.feat_channels, self.num_anchors * self.box_code_size, 1, bias=True)
            if self.use_direction_classifier:
                self.conv_dir_cls = SubMConv2d(self.feat_channels, self.num_anchors * 2, 1, bias=True)

    def forward_single(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward function on a single-scale feature map.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_base_priors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_base_priors * C.
                dir_cls_pred (Tensor | None): Direction classification
                    prediction for a single scale level, the channels
                    number is num_base_priors * 2.
        """
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        dir_cls_pred = None
        if self.use_direction_classifier:
            dir_cls_pred = self.conv_dir_cls(x)
        return cls_score, bbox_pred, dir_cls_pred

    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor]]:
        """Forward pass.

        Args:
            x (tuple[Tensor]): Features from the upstream network,
                each is a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores, bbox and direction
                classification prediction.

                - cls_scores (list[Tensor]): Classification scores for all
                    scale levels, each is a 4D-tensor, the channels number
                    is num_base_priors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all
                    scale levels, each is a 4D-tensor, the channels number
                    is num_base_priors * C.
                - dir_cls_preds (list[Tensor|None]): Direction classification
                    predictions for all scale levels, each is a 4D-tensor,
                    the channels number is num_base_priors * 2.
        """
        return multi_apply(self.forward_single, x)

    # TODO: Support augmentation test
    def aug_test(self,
                 aug_batch_feats,
                 aug_batch_input_metas,
                 rescale=False,
                 **kwargs):
        aug_bboxes = []
        # only support aug_test for one sample
        for x, input_meta in zip(aug_batch_feats, aug_batch_input_metas):
            outs = self.forward(x)
            bbox_list = self.get_results(*outs, [input_meta], rescale=rescale)
            bbox_dict = dict(
                bboxes_3d=bbox_list[0].bboxes_3d,
                scores_3d=bbox_list[0].scores_3d,
                labels_3d=bbox_list[0].labels_3d)
            aug_bboxes.append(bbox_dict)
        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, aug_batch_input_metas,
                                            self.test_cfg)
        return [merged_bboxes]

    def get_anchors(self,
                    featmap_sizes: List[tuple],
                    input_metas: List[dict],
                    device: str = 'cuda') -> list:
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            input_metas (list[dict]): contain pcd and img's meta info.
            device (str): device of current module.

        Returns:
            list[list[torch.Tensor]]: Anchors of each image, valid flags
                of each image.
        """
        num_imgs = len(input_metas)
        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.prior_generator.grid_anchors(
            featmap_sizes, device=device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]
        return anchor_list

    def _loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
                             dir_cls_pred: Tensor, labels: Tensor,
                             label_weights: Tensor, bbox_targets: Tensor,
                             bbox_weights: Tensor, dir_targets: Tensor,
                             dir_weights: Tensor, num_total_samples: int):
        """Calculate loss of Single-level results.

        Args:
            cls_score (Tensor): Class score in single-level.
            bbox_pred (Tensor): Bbox prediction in single-level.
            dir_cls_pred (Tensor): Predictions of direction class
                in single-level.
            labels (Tensor): Labels of class.
            label_weights (Tensor): Weights of class loss.
            bbox_targets (Tensor): Targets of bbox predictions.
            bbox_weights (Tensor): Weights of bbox loss.
            dir_targets (Tensor): Targets of direction predictions.
            dir_weights (Tensor): Weights of direction loss.
            num_total_samples (int): The number of valid samples.

        Returns:
            tuple[torch.Tensor]: Losses of class, bbox
                and direction, respectively.
        """
        # classification loss
        if num_total_samples is None:
            num_total_samples = int(cls_score.shape[0])
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        assert labels.max().item() <= self.num_classes
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)

        # regression loss
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1, self.box_code_size)
        bbox_targets = bbox_targets.reshape(-1, self.box_code_size)
        bbox_weights = bbox_weights.reshape(-1, self.box_code_size)

        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero(
                        as_tuple=False).reshape(-1)
        num_pos = len(pos_inds)

        pos_bbox_pred = bbox_pred[pos_inds]
        pos_bbox_targets = bbox_targets[pos_inds]
        pos_bbox_weights = bbox_weights[pos_inds]

        # dir loss
        if self.use_direction_classifier:
            dir_cls_pred = dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)
            dir_targets = dir_targets.reshape(-1)
            dir_weights = dir_weights.reshape(-1)
            pos_dir_cls_pred = dir_cls_pred[pos_inds]
            pos_dir_targets = dir_targets[pos_inds]
            pos_dir_weights = dir_weights[pos_inds]

        if num_pos > 0:
            code_weight = self.train_cfg.get('code_weight', None)
            if code_weight:
                pos_bbox_weights = pos_bbox_weights * bbox_weights.new_tensor(
                    code_weight)
            if self.diff_rad_by_sin:
                pos_bbox_pred, pos_bbox_targets = self.add_sin_difference(
                    pos_bbox_pred, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_bbox_pred,
                pos_bbox_targets,
                pos_bbox_weights,
                avg_factor=num_total_samples)

            # direction classification loss
            loss_dir = None
            if self.use_direction_classifier:
                loss_dir = self.loss_dir(
                    pos_dir_cls_pred,
                    pos_dir_targets,
                    pos_dir_weights,
                    avg_factor=num_total_samples)
        else:
            loss_bbox = pos_bbox_pred.sum()
            if self.use_direction_classifier:
                loss_dir = pos_dir_cls_pred.sum()

        return loss_cls, loss_bbox, loss_dir

    @staticmethod
    def add_sin_difference(boxes1: Tensor, boxes2: Tensor) -> tuple:
        """Convert the rotation difference to difference in sine function.

        Args:
            boxes1 (torch.Tensor): Original Boxes in shape (NxC), where C>=7
                and the 7th dimension is rotation dimension.
            boxes2 (torch.Tensor): Target boxes in shape (NxC), where C>=7 and
                the 7th dimension is rotation dimension.

        Returns:
            tuple[torch.Tensor]: ``boxes1`` and ``boxes2`` whose 7th
                dimensions are changed.
        """
        rad_pred_encoding = torch.sin(boxes1[..., 6:7]) * torch.cos(
            boxes2[..., 6:7])
        rad_tg_encoding = torch.cos(boxes1[..., 6:7]) * torch.sin(boxes2[...,
                                                                         6:7])
        boxes1 = torch.cat(
            [boxes1[..., :6], rad_pred_encoding, boxes1[..., 7:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :6], rad_tg_encoding, boxes2[..., 7:]],
                           dim=-1)
        return boxes1, boxes2

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            dir_cls_preds: List[Tensor],
            batch_gt_instances_3d: InstanceList,
            batch_input_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None,
            **kwargs) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[torch.Tensor]): Multi-level class scores.
            bbox_preds (list[torch.Tensor]): Multi-level bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Multi-level direction
                class predictions.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances. It usually includes ``bboxes_3d``
                and ``labels_3d`` attributes.
            batch_input_metas (list[dict]): Contain pcd and img's meta info.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, list[torch.Tensor]]: Classification, bbox, and
                direction losses of each level.

                - loss_cls (list[torch.Tensor]): Classification losses.
                - loss_bbox (list[torch.Tensor]): Box regression losses.
                - loss_dir (list[torch.Tensor]): Direction classification
                    losses.
        """
        if self.use_sparse and not isinstance(cls_scores, torch.Tensor):
            cls_scores = [item.dense() for item in cls_scores]
            bbox_preds = [item.dense() for item in bbox_preds]
            if dir_cls_preds is not None:
                dir_cls_preds = [item.dense() for item in dir_cls_preds]
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels
        device = cls_scores[0].device
        anchor_list = self.get_anchors(
            featmap_sizes, batch_input_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.anchor_target_3d(
            anchor_list,
            batch_gt_instances_3d,
            batch_input_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
            num_classes=self.num_classes,
            label_channels=label_channels,
            sampling=self.sampling)

        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         dir_targets_list, dir_weights_list, num_total_pos,
         num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        if ('bev_non_empty_mask' in kwargs) and self.use_sparse:
            batch_size = cls_scores[0].shape[0]
            m = kwargs['bev_non_empty_mask'].view((batch_size, -1)).repeat_interleave(self.num_classes*2, dim=1)
            label_weights_list[0][~m] = 0.0
            bbox_weights_list[0][~m] = 0.0
            dir_weights_list[0][~m] = 0.0
            #num_total_samples = bbox_weights_list[0].sum()/7

        # num_total_samples = None
        with amp.autocast(enabled=False):
            losses_cls, losses_bbox, losses_dir = multi_apply(
                self._loss_by_feat_single,
                cast_tensor_type(cls_scores, dst_type=torch.float32),
                cast_tensor_type(bbox_preds, dst_type=torch.float32),
                cast_tensor_type(dir_cls_preds, dst_type=torch.float32),
                labels_list,
                label_weights_list,
                bbox_targets_list,
                bbox_weights_list,
                dir_targets_list,
                dir_weights_list,
                num_total_samples=num_total_samples)
        return dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dir=losses_dir)

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        dir_cls_preds: List[Tensor],
                        batch_input_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        **kwargs) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            score_factors (list[Tensor], optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Defaults to None.
            batch_input_metas (list[dict], Optional): Batch inputs meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each sample
            after the post process.
            Each item usually contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (BaseInstance3DBoxes): Prediction of bboxes,
              contains a tensor with shape (num_instances, C), where
              C >= 7.
        """
        if self.use_sparse and not isinstance(cls_scores, torch.Tensor):
            cls_scores = [item.dense() for item in cls_scores]
            bbox_preds = [item.dense() for item in bbox_preds]
            if dir_cls_preds is not None:
                dir_cls_preds = [item.dense() for item in dir_cls_preds]

            assert len(cls_scores) == 1
            m = kwargs['bev_non_empty_mask']
            cls_scores = cls_scores[0].permute(0, 2, 3, 1)
            cls_scores[~m] = -10
            cls_scores = [cls_scores.permute(0, 3, 1, 2)]

        assert len(cls_scores) == len(bbox_preds)
        assert len(cls_scores) == len(dir_cls_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_anchors(
            featmap_sizes, device=cls_scores[0].device)
        mlvl_priors = [
            prior.reshape(-1, self.box_code_size) for prior in mlvl_priors
        ]

        result_list = []

        for input_id in range(len(batch_input_metas)):

            input_meta = batch_input_metas[input_id]
            cls_score_list = select_single_mlvl(cls_scores, input_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, input_id)
            dir_cls_pred_list = select_single_mlvl(dir_cls_preds, input_id)

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                dir_cls_pred_list=dir_cls_pred_list,
                mlvl_priors=mlvl_priors,
                input_meta=input_meta,
                cfg=cfg,
                rescale=rescale,
                **kwargs)
            result_list.append(results)

        if 'points' in kwargs:
            batch_points = kwargs['points']
            for k, (dets, points) in enumerate(zip(result_list, batch_points)):
                num_boxes = len(dets.bboxes_3d)
                box_idxs_of_pts = points_in_boxes_part(points[None,:,:3], dets.bboxes_3d.tensor[None, ...]).squeeze()
                selected = []
                for i in range(num_boxes):
                    points_num_in_box = (box_idxs_of_pts == i).sum()
                    if points_num_in_box > 0:
                        selected.append(i)
                if len(selected) == 0:
                    result_list[k] = result_list[k][:0]
                else:
                    result_list[k] = result_list[k][selected]
        return result_list