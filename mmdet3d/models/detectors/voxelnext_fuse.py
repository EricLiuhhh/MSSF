# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence
import os
import pickle
import torch
from torch import Tensor
from mmengine.structures import InstanceData
from mmengine.logging import print_log
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor
else:
    from mmcv.ops import SparseConvTensor
from .mvx_two_stage import MVXTwoStageDetector

@MODELS.register_module()
class VoxelNeXtFuse(MVXTwoStageDetector):
    """Base class of Multi-modality VoxelNet.

    Args:
        pts_voxel_encoder (dict, optional): Point voxelization
            encoder layer. Defaults to None.
        pts_middle_encoder (dict, optional): Middle encoder layer
            of points cloud modality. Defaults to None.
        pts_fusion_layer (dict, optional): Fusion layer.
            Defaults to None.
        img_backbone (dict, optional): Backbone of extracting
            images feature. Defaults to None.
        pts_backbone (dict, optional): Backbone of extracting
            points features. Defaults to None.
        img_neck (dict, optional): Neck of extracting
            image features. Defaults to None.
        pts_neck (dict, optional): Neck of extracting
            points features. Defaults to None.
        pts_bbox_head (dict, optional): Bboxes head of
            point cloud modality. Defaults to None.
        img_roi_head (dict, optional): RoI head of image
            modality. Defaults to None.
        img_rpn_head (dict, optional): RPN head of image
            modality. Defaults to None.
        train_cfg (dict, optional): Train config of model.
            Defaults to None.
        test_cfg (dict, optional): Train config of model.
            Defaults to None.
        init_cfg (dict, optional): Initialize config of
            model. Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`Det3DDataPreprocessor`. Defaults to None.
    """

    def __init__(self,
                 pts_voxel_encoder: Optional[dict] = None,
                 pts_middle_encoder: Optional[dict] = None,
                 pts_fusion_layer: Optional[dict] = None,
                 img_backbone: Optional[dict] = None,
                 pts_backbone: Optional[dict] = None,
                 img_neck: Optional[dict] = None,
                 pts_neck: Optional[dict] = None,
                 pts_bbox_head: Optional[dict] = None,
                 img_roi_head: Optional[dict] = None,
                 img_rpn_head: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 **kwargs):

        super(VoxelNeXtFuse,
              self).__init__(pts_voxel_encoder, pts_middle_encoder,
                             pts_fusion_layer, img_backbone, pts_backbone,
                             img_neck, pts_neck, pts_bbox_head, img_roi_head,
                             img_rpn_head, train_cfg, test_cfg, init_cfg,
                             data_preprocessor, **kwargs)
        self.freeze_img = kwargs.get('freeze_img', True)

    def init_weights(self):
        """Initialize model weights."""
        super(VoxelNeXtFuse, self).init_weights()

        if self.freeze_img:
            if self.with_img_backbone:
                print_log('freezed image backbone', 'current')
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.with_img_neck:
                print_log('freezed image neck', 'current')
                for param in self.img_neck.parameters():
                    param.requires_grad = False

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
            batch_gt_instance_3d = None, # only used for ablation exps
            batch_gt_instance_2d = None,
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
        if not self.with_pts_bbox:
            return None
        voxel_features = self.pts_voxel_encoder(voxel_dict['voxels'],
                                                voxel_dict['num_points'],
                                                voxel_dict['coors'], img_feats,
                                                batch_input_metas)
        batch_size = voxel_dict['coors'][-1, 0] + 1
        x, points_misc = self.pts_middle_encoder(voxel_features, voxel_dict['coors'],
                                    batch_size, points=points, img_features=img_feats,
                                    batch_input_metas=batch_input_metas,
                                    imgs=imgs, batch_gt_instance_3d=batch_gt_instance_3d, batch_gt_instance_2d=batch_gt_instance_2d)
        if self.with_pts_backbone:
            x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        if isinstance(x, (torch.Tensor, SparseConvTensor)):
            x = [x]
        return x, points_misc

    def extract_feat(self, batch_inputs_dict: dict,
                     batch_input_metas: List[dict],
                     batch_gt_instance_3d=None,
                     batch_gt_instance_2d=None) -> tuple:
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

        # save_dir = 'data/precompute_img_feats'
        # if not os.path.exists(save_dir):
        #     os.mkdir(save_dir)
        # for i, meta in enumerate(batch_input_metas):
        #     frame_id = meta['lidar_path'].split('/')[-1].split('.')[0]
        #     ms_feats = []
        #     for feat in img_feats:
        #         ms_feats.append(feat[i].unsqueeze(0).to(torch.float16).cpu())
        #     torch.save(ms_feats, os.path.join(save_dir, f'{frame_id}.pt'))
        # return None, None, None
        # self.pts_fusion_layer()
        pts_feats, points_misc = self.extract_pts_feat(
            voxel_dict,
            points=points,
            img_feats=img_feats,
            batch_input_metas=batch_input_metas,
            imgs=imgs,
            batch_gt_instance_3d=batch_gt_instance_3d,
            batch_gt_instance_2d=batch_gt_instance_2d)
        
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
        batch_gt_instances_3d = [
            data_sample.eval_ann_info['gt_bboxes_3d']
            for data_sample in batch_data_samples
        ]
        batch_gt_instances_2d = [
            data_sample.eval_ann_info['gt_bboxes']
            for data_sample in batch_data_samples
        ]
        img_feats, pts_feats, points_misc = self.extract_feat(batch_inputs_dict,
                                                 batch_input_metas, batch_gt_instances_3d, batch_gt_instances_2d)
        if pts_feats and self.with_pts_bbox:
            kwargs.update(points=batch_inputs_dict['points'])
            #kwargs.update(points_misc=points_misc)
            results_list_3d = self.pts_bbox_head.predict(
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

        if results_list_3d is not None or results_list_2d is not None:
            detsamples = self.add_pred_to_datasample(batch_data_samples,
                                                    results_list_3d,
                                                    results_list_2d)
        else:
            detsamples = batch_data_samples
            
        #detsamples = batch_data_samples
        #assert len(detsamples) == 1
        for i in range(len(detsamples)):
            detsamples[i].seg_preds = dict()
            detsamples[i].centroids = dict()
            detsamples[i].centroids_img = dict()
            for loc, res in points_misc.items():
                batch_mask = (points_misc[loc]['voxel_dict']['centroids'][:,0].int()==i)
                pred_pts_seg = points_misc[loc]['seg_results']['seg_preds'][batch_mask]
                voxel_centroids = points_misc[loc]['voxel_dict']['centroids'][batch_mask][:,1:4]
                voxel_centroids_img = points_misc[loc]['voxel_dict']['centroids_img'][batch_mask][:,1:]
                detsamples[i].seg_preds[loc] = pred_pts_seg.squeeze()
                detsamples[i].centroids[loc] = voxel_centroids
                detsamples[i].centroids_img[loc] = voxel_centroids_img
        return detsamples


    def seg_predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                    batch_input_metas,
                    **kwargs) -> List[Det3DDataSample]:
        """seg_predict
        """
        img_feats, pts_feats, points_misc = self.extract_feat(batch_inputs_dict,
                                                 batch_input_metas)
        # if pts_feats and self.with_pts_bbox:
        #     kwargs.update(points=batch_inputs_dict['points'])
        #     results_list_3d = self.pts_bbox_head.predict(
        #         pts_feats, batch_data_samples, **kwargs)
        # else:
        #     results_list_3d = None

        # if img_feats and self.with_img_bbox:
        #     # TODO check this for camera modality
        #     results_list_2d = self.predict_imgs(img_feats, batch_data_samples,
        #                                         **kwargs)
        # else:
        #     results_list_2d = None
        
        return points_misc['x_conv3']
    
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
        batch_gt_instances_3d = [
            data_sample.gt_instances_3d
            for data_sample in batch_data_samples
        ]
        batch_gt_instances_2d = [
            data_sample.gt_instances
            for data_sample in batch_data_samples
        ]
        img_feats, pts_feats, points_misc = self.extract_feat(batch_inputs_dict,
                                                 batch_input_metas,
                                                 batch_gt_instances_3d,
                                                 batch_gt_instances_2d) # gt_instance_3d only used for ablation exps
        losses = dict()
        aux_loss = self.pts_middle_encoder.aux_loss(points_misc,
                                                batch_gt_instances_3d)
        losses.update(aux_loss)
        
        if pts_feats:
            losses_pts = self.pts_bbox_head.loss(pts_feats, batch_data_samples,
                                                 **kwargs)
            losses.update(losses_pts)

        # if img_feats:
        #     losses_img = self.loss_imgs(img_feats, batch_data_samples)
        #     losses.update(losses_img)
        return losses