# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, List, Dict, Optional, Sequence
import torch
from torch import Tensor
from mmengine.logging import print_log
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
from mmdet3d.structures import Det3DDataSample
if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor
else:
    from mmcv.ops import SparseConvTensor
from .single_stage import SingleStage3DDetector
from ..backbones.second_fuse import SECONDFuse

@MODELS.register_module()
class VoxelNetFuse(SingleStage3DDetector):
    r"""`VoxelNet <https://arxiv.org/abs/1711.06396>`_ for 3D detection."""

    def __init__(self,
                 voxel_encoder: ConfigType,
                 middle_encoder: ConfigType,
                 backbone: OptConfigType = None,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 img_backbone: OptConfigType = None,
                 img_neck: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.voxel_encoder = MODELS.build(voxel_encoder)
        self.middle_encoder = MODELS.build(middle_encoder)

        if img_backbone:
            self.img_backbone = MODELS.build(img_backbone)
        if img_neck is not None:
            self.img_neck = MODELS.build(img_neck)
            
        self.freeze_img = True

    def init_weights(self):
        """Initialize model weights."""
        super(VoxelNetFuse, self).init_weights()

        if self.freeze_img:
            if self.with_img_backbone:
                print_log('freezed image backbone', 'current')
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.with_img_neck:
                print_log('freezed image neck', 'current')
                for param in self.img_neck.parameters():
                    param.requires_grad = False
    @property
    def with_img_backbone(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, 'img_backbone') and self.img_backbone is not None
    
    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_neck') and self.img_neck is not None
    
    def extract_img_feat(self, img: Tensor, input_metas: List[dict]) -> dict:
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in input_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            img_feats = self.img_backbone(img)
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats
    
    def extract_pts_feat(
            self,
            voxel_dict: Dict[str, Tensor],
            points: Optional[List[Tensor]] = None,
            img_feats: Optional[Sequence[Tensor]] = None,
            batch_input_metas: Optional[List[dict]] = None,
            imgs: Optional[Tensor] = None,
    ) -> Sequence[Tensor]:
        """Extract features of points.

        Args:
            voxel_dict(Dict[str, Tensor]): Dict of voxelization infos.
            points (List[tensor], optional):  Point cloud of multiple inputs.
            img_feats (list[Tensor], tuple[tensor], optional): Features from
                image backbone.
            batch_input_metas (list[dict], optional): The meta information
                of multiple samples. Defaults to True.

        Returns:
            Sequence[tensor]: points features of multiple inputs
            from backbone or neck.
        """
        voxel_features = self.voxel_encoder(voxel_dict['voxels'],
                                                voxel_dict['num_points'],
                                                voxel_dict['coors'])
        batch_size = voxel_dict['coors'][-1, 0] + 1
        x, points_misc = self.middle_encoder(voxel_features, voxel_dict['coors'],
                                    batch_size, points=points, img_features=img_feats,
                                    batch_input_metas=batch_input_metas,
                                    imgs=imgs)
        if self.with_backbone:
            if isinstance(self.backbone, SECONDFuse):
                x, points_misc = self.backbone(x, points=points, img_features=img_feats,
                                    batch_input_metas=batch_input_metas,
                                    imgs=imgs)
            else:
                x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        if isinstance(x, (torch.Tensor, SparseConvTensor)):
            x = [x]
        return x, points_misc

    def extract_feat(self, batch_inputs_dict: dict,
                     batch_input_metas: List[dict]) -> tuple:
        """Extract features from images and points.

        Args:
            batch_inputs_dict (dict): Dict of batch inputs. It
                contains

                - points (List[tensor]):  Point cloud of multiple inputs.
                - imgs (tensor): Image tensor with shape (B, C, H, W).
            batch_input_metas (list[dict]): Meta information of multiple inputs
                in a batch.

        Returns:
             tuple: Two elements in tuple arrange as
             image features and point cloud features.
        """
        voxel_dict = batch_inputs_dict.get('voxels', None)
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        img_feats = self.extract_img_feat(imgs, batch_input_metas)

        pts_feats, points_misc = self.extract_pts_feat(
            voxel_dict,
            points=points,
            img_feats=img_feats,
            batch_input_metas=batch_input_metas,
            imgs=imgs)
        
        return (img_feats, pts_feats, points_misc)

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        img_feats, pts_feats, points_misc = self.extract_feat(batch_inputs_dict,
                                                 batch_input_metas)
        if pts_feats:
            #kwargs.update(points=batch_inputs_dict['points'])
            #kwargs.update(points_misc=points_misc)
            results_list_3d = self.bbox_head.predict(
                pts_feats, batch_data_samples, **kwargs)
        else:
            results_list_3d = None

        # if img_feats and self.with_img_bbox:
        #     # TODO check this for camera modality
        #     results_list_2d = self.predict_imgs(img_feats, batch_data_samples,
        #                                         **kwargs)
        # else:
        #     results_list_2d = None
        
        # results_list_3d = [InstanceData()]
        results_list_2d = None

        detsamples = self.add_pred_to_datasample(batch_data_samples,
                                                 results_list_3d,
                                                 results_list_2d)

        #detsamples = batch_data_samples
        #assert len(detsamples) == 1
        # for i in range(len(detsamples)):
        #     detsamples[i].seg_preds = dict()
        #     detsamples[i].centroids = dict()
        #     for loc, res in points_misc.items():
        #         batch_mask = (points_misc[loc]['voxel_dict']['centroids'][:,0].int()==i)
        #         pred_pts_seg = points_misc[loc]['seg_results']['seg_preds'][batch_mask]
        #         voxel_centroids = points_misc[loc]['voxel_dict']['centroids'][batch_mask][:,1:4]
        #         detsamples[i].seg_preds[loc] = pred_pts_seg.squeeze()
        #         detsamples[i].centroids[loc] = voxel_centroids
        return detsamples

    
    def loss(self, batch_inputs_dict: Dict[List, torch.Tensor],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        """
        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' and `imgs` keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor): Tensor of batch images, has shape
                  (B, C, H ,W)
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, .

        Returns:
            dict[str, Tensor]: A dictionary of loss components.

        """

        batch_input_metas = [item.metainfo for item in batch_data_samples]
        img_feats, pts_feats, points_misc = self.extract_feat(batch_inputs_dict,
                                                 batch_input_metas)
        losses = dict()
        batch_gt_instances_3d = [
            data_sample.gt_instances_3d
            for data_sample in batch_data_samples
        ]
        if isinstance(self.backbone, SECONDFuse):
            aux_loss = self.backbone.aux_loss(points_misc, batch_gt_instances_3d)
        else:
            aux_loss = self.middle_encoder.aux_loss(points_misc,
                                                batch_gt_instances_3d)
        losses.update(aux_loss)
        
        if pts_feats:
            losses_pts = self.bbox_head.loss(pts_feats, batch_data_samples,
                                                 **kwargs)
            losses.update(losses_pts)

        return losses