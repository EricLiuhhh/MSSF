# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
from typing import Dict, Optional, Sequence
import time
import torch
import numba
import matplotlib.pyplot as plt
import mmcv
import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmdet3d.evaluation import seg_eval
from mmdet3d.registry import METRICS

@numba.jit(nopython=True)
def _stat_blur(scores:np.ndarray, blur_flag:np.ndarray, fore_flag:np.ndarray):
    sample_num = 100
    ths = np.linspace(0, 1, sample_num)
    blurs = np.zeros((sample_num,), dtype=np.int64)
    blur_dens = np.zeros((sample_num,), dtype=np.int64)

    for k, th in enumerate(ths):
        seg_labels = scores > th
        blurs[k] += seg_labels[blur_flag].sum()
        #blur_dens[k] += seg_labels[fore_flag].sum()
        blur_dens[k] += fore_flag.sum()
    return blurs, blur_dens

@numba.jit(nopython=True)
def _stat_recall(scores:np.ndarray, n_bboxes:int, points_label:np.ndarray, fg_flag:np.ndarray, bg_flag:np.ndarray, ign_flag:np.ndarray):
    sample_num = 100
    ths = np.linspace(0, 1, sample_num)
    cnt = np.zeros((sample_num,), dtype=np.int64)
    fps = np.zeros((sample_num,), dtype=np.int64)
    fp_dens = np.zeros((sample_num,), dtype=np.int64)

    for k, th in enumerate(ths):
        seg_labels = scores > th
        for i in range(n_bboxes):
            if seg_labels[points_label==(i+1)].sum() > 0:
                cnt[k] += 1
        #fps[k] += seg_labels[bg_flag].sum()
            fps[k] += seg_labels[points_label==(i+1)].sum()
        fp_dens[k] += seg_labels[~ign_flag].sum()
    return cnt, fps, fp_dens

@METRICS.register_module()
class FGSegMetric(BaseMetric):
    """3D semantic segmentation evaluation metric.

    Args:
        collect_device (str, optional): Device name used for collecting
            results from different ranks during distributed training.
            Must be 'cpu' or 'gpu'. Defaults to 'cpu'.
        prefix (str): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None.
        pklfile_prefix (str, optional): The prefix of pkl files, including
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Default: None.
        submission_prefix (str, optional): The prefix of submission data.
            If not specified, the submission data will not be generated.
            Default: None.
    """

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 pklfile_prefix: str = None,
                 submission_prefix: str = None,
                 **kwargs):
        self.pklfile_prefix = pklfile_prefix
        self.submission_prefix = submission_prefix
        super(FGSegMetric, self).__init__(
            prefix=prefix, collect_device=collect_device)
        
        self.num_classes = 1
        self.extra_width = 0.1
        #self.save_dir = kwargs.get('save_dir', './')
        self.save_dir = submission_prefix if submission_prefix is not None else './'

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``,
        which will be used to compute the metrics when all batches
        have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        #self.img_input_shape = data_samples[0]['batch_input_shape']
        self.img_input_shape = (1216, 1952)
        for data_sample in data_samples:
            pred_3d = data_sample['seg_preds']
            centroids = data_sample['centroids']
            centroids_img = data_sample['centroids_img']
            eval_ann_info = data_sample['eval_ann_info']
            cpu_pred_3d = dict()
            cpu_centroids = dict()
            cpu_centroids_img = dict()
            for loc in pred_3d.keys():
                if hasattr(pred_3d[loc], 'to'):
                    cpu_pred_3d[loc] = pred_3d[loc].to('cpu')
                else:
                    cpu_pred_3d[loc] = pred_3d[loc]
                if hasattr(centroids[loc], 'to'):
                    cpu_centroids[loc] = centroids[loc].to('cpu')
                else:
                    cpu_centroids[loc] = centroids[loc]
                if hasattr(centroids_img[loc], 'to'):
                    cpu_centroids_img[loc] = centroids_img[loc].to('cpu')
                else:
                    cpu_centroids_img[loc] = centroids_img[loc]
            self.results.append((eval_ann_info, cpu_pred_3d, cpu_centroids, cpu_centroids_img))

    def format_results(self, results):
        r"""Format the results to txt file. Refer to `ScanNet documentation
        <http://kaldir.vc.in.tum.de/scannet_benchmark/documentation>`_.

        Args:
            outputs (list[dict]): Testing results of the dataset.

        Returns:
            tuple: (outputs, tmp_dir), outputs is the detection results,
                tmp_dir is the temporal directory created for saving submission
                files when ``submission_prefix`` is not specified.
        """

        submission_prefix = self.submission_prefix
        if submission_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            submission_prefix = osp.join(tmp_dir.name, 'results')
        mmcv.mkdir_or_exist(submission_prefix)
        ignore_index = self.dataset_meta['ignore_index']
        # need to map network output to original label idx
        cat2label = np.zeros(len(self.dataset_meta['label2cat'])).astype(
            np.int64)
        for original_label, output_idx in self.dataset_meta['label2cat'].items(
        ):
            if output_idx != ignore_index:
                cat2label[output_idx] = original_label

        for i, (eval_ann, result) in enumerate(results):
            sample_idx = eval_ann['point_cloud']['lidar_idx']
            pred_sem_mask = result['semantic_mask'].numpy().astype(np.int64)
            pred_label = cat2label[pred_sem_mask]
            curr_file = f'{submission_prefix}/{sample_idx}.txt'
            np.savetxt(curr_file, pred_label, fmt='%d')


    
    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        #if self.submission_prefix:
        #    self.format_results(results)
        #    return None
        
        # import copy
        # keys = copy.deepcopy(list(results[0][-1].keys()))
        # for result in results:
        #     all_points = []
        #     all_seg_preds = []
        #     for loc in result[-1].keys():
        #         all_points.append(result[-1][loc])
        #         all_seg_preds.append(result[1][loc])
        #     result[-1]['combine'] = torch.cat(all_points)
        #     result[1]['combine'] = torch.cat(all_seg_preds)
        #     for loc in keys:
        #         del result[-1][loc]
        #         del result[1][loc]
        if not osp.exists(self.save_dir):
            os.makedirs(self.save_dir)
        sample_num = 100
        ths = np.linspace(0, 1, sample_num)
        cnt = dict()
        fps = dict()
        fp_dens = dict()
        den = dict()

        blurs = dict()
        blur_dens = dict()
        no_blurs = dict()
        no_blur_dens = dict()

        pos_mean = dict()
        pos_std = dict()
        neg_mean = dict()
        neg_std = dict()
        all_locs = list(results[0][-1].keys())
        for loc in all_locs:
            #cnt[loc] = [0 for _ in range(sample_num)]
            #fps[loc] = [0 for _ in range(sample_num)]
            #fp_dens[loc] = [0 for _ in range(sample_num)]
            cnt[loc] = np.zeros((sample_num,), dtype=np.int64)
            fps[loc] = np.zeros((sample_num,), dtype=np.int64)
            fp_dens[loc] = np.zeros((sample_num,), dtype=np.int64)
            den[loc] = 0

            blurs[loc] = np.zeros((sample_num,), dtype=np.int64)
            blur_dens[loc] = np.zeros((sample_num,), dtype=np.int64)
            no_blurs[loc] = np.zeros((sample_num,), dtype=np.int64)
            no_blur_dens[loc] = np.zeros((sample_num,), dtype=np.int64)

            pos_mean[loc] = []
            pos_std[loc] = 0
            neg_mean[loc] = []
            neg_std[loc] = 0
        
        s = time.time()
        for loc in all_locs:
            for result in results:
                gt_bboxes_3d = result[0]['gt_bboxes_3d'].cuda()
                gt_bboxes_2d = torch.tensor(result[0]['gt_bboxes'])
                gt_labels_3d = result[0]['gt_labels_3d']
        
                point_xyz = result[-2][loc]
                point_xyz = point_xyz.cuda()
                point_cls_labels_single = point_xyz.new_zeros(
                    point_xyz.shape[0]).long()
                enlarged_gt_boxes = gt_bboxes_3d.enlarged_box(self.extra_width)

                box_idxs_of_pts = gt_bboxes_3d.points_in_boxes_part(point_xyz).long()
                extend_box_idxs_of_pts = enlarged_gt_boxes.points_in_boxes_part(
                    point_xyz).long()
                box_fg_flag = box_idxs_of_pts >= 0
                fg_flag = box_fg_flag.clone()
                ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
                point_cls_labels_single[ignore_flag] = -1
                #gt_box_of_fg_points = gt_labels_3d[box_idxs_of_pts[fg_flag].cpu()]
                gt_box_of_fg_points = box_idxs_of_pts[fg_flag]
                # point_cls_labels_single[
                #     fg_flag] = 1 if self.num_classes == 1 else\
                #     gt_box_of_fg_points.long()
                point_cls_labels_single[fg_flag] = gt_box_of_fg_points.long()+1
                point_cls_labels_single = point_cls_labels_single.cpu()
                # th = 0.3
                bg_flag = (~ignore_flag) & (~fg_flag)
                seg_preds = result[1][loc].sigmoid()
                # for k, th in enumerate(ths):
                #     seg_labels = seg_preds > th
                #     for i in range(len(gt_bboxes_3d)):
                #         if seg_labels[point_cls_labels_single==(i+1)].sum() > 0:
                #             cnt[loc][k] += 1
                #     #fps[k] += seg_labels[bg_flag].sum()
                #         fps[loc][k] += seg_labels[point_cls_labels_single==(i+1)].sum()
                #     fp_dens[loc][k] += seg_labels[~ignore_flag].sum()

                point_2d = result[-1][loc]
                point_2d[:, 0] *= self.img_input_shape[1]
                point_2d[:, 1] *= self.img_input_shape[0]
                point_2d = point_2d.unsqueeze(1)
                gt_bboxes_2d = gt_bboxes_2d.unsqueeze(0)
                box2d_fg_flag = (point_2d[...,0]>=gt_bboxes_2d[...,0]) \
                            & (point_2d[...,0]<=gt_bboxes_2d[...,2]) \
                            & (point_2d[...,1]>=gt_bboxes_2d[...,1]) \
                            & (point_2d[...,1]<=gt_bboxes_2d[...,3])
                box2d_fg_flag = torch.any(box2d_fg_flag, dim=1)
                box2d_bg_flag = ~box2d_fg_flag
                blur_flag = box2d_fg_flag & (~fg_flag.cpu()) # 2d前景并且不是3d前景
                _blurs, _blur_dens = _stat_blur(seg_preds.numpy(), blur_flag.numpy(), box2d_fg_flag.numpy())
                blurs[loc] += _blurs
                blur_dens[loc] += _blur_dens

                no_blur_flag = box2d_fg_flag & (fg_flag.cpu()) # 2d前景并且是3d前景
                _no_blurs, _no_blur_dens = _stat_blur(seg_preds.numpy(), no_blur_flag.numpy(), box2d_fg_flag.numpy())
                no_blurs[loc] += _no_blurs
                no_blur_dens[loc] += _no_blur_dens

                _cnt, _fps, _fp_dens = _stat_recall(seg_preds.numpy(), len(gt_bboxes_3d), point_cls_labels_single.numpy(), fg_flag.cpu().numpy(), bg_flag.cpu().numpy(), ignore_flag.cpu().numpy())
                cnt[loc] += _cnt
                fps[loc] += _fps
                fp_dens[loc] += _fp_dens
                
                den[loc] += len(gt_bboxes_3d)

                pos_mean[loc] += [seg_preds[fg_flag]]
                neg_mean[loc] += [seg_preds[bg_flag]]
        print(f'compute time:{time.time()-s}')
        ret_dict = dict()
        for loc in results[0][-1].keys():
            covs = [c/den[loc] for c in cnt[loc]]
            #pres = [fp/fp_den for fp in fps]
            pres = [fp/fp_den if fp_den != 0 else 1 for fp,fp_den in zip(fps[loc],fp_dens[loc])]
            AR = 0
            AP = 0
            for cov in covs:
                AR += cov / sample_num
            for pre in pres:
                AP += pre / sample_num
            ret_dict[f'AR_{loc}'] = AR
            ret_dict[f'AP_{loc}'] = AP
            
            plt.figure()
            plt.plot(ths, covs)
            plt.savefig(os.path.join(self.save_dir, f'AR_{loc}.png'))
            plt.figure()
            plt.plot(ths, pres)
            plt.savefig(os.path.join(self.save_dir, f'AP_{loc}.png'))
            np.save(os.path.join(self.save_dir, f'ar.npy'), np.array(covs))
            np.save(os.path.join(self.save_dir, f'ap.npy'), np.array(pres))

            blur_ratio = [blur/blur_den if blur_den != 0 else 0 for blur,blur_den in zip(blurs[loc],blur_dens[loc])]
            BR = 0
            for b in blur_ratio:
                BR += b / sample_num
            ret_dict[f'BR_{loc}'] = BR
            
            plt.figure()
            plt.plot(ths, blur_ratio)
            plt.ylim([0, 1])
            plt.xlim([0, 1])
            plt.savefig(os.path.join(self.save_dir, f'BR_{loc}.png'))
            np.save(os.path.join(self.save_dir, f'br.npy'), np.array(blur_ratio))

            no_blur_ratio = [no_blur/no_blur_den if no_blur_den != 0 else 0 for no_blur,no_blur_den in zip(no_blurs[loc],no_blur_dens[loc])]
            np.save(os.path.join(self.save_dir, f'nobr.npy'), np.array(no_blur_ratio))

            pos_mean[loc] = torch.cat(pos_mean[loc])
            neg_mean[loc] = torch.cat(neg_mean[loc])
            plt.figure()
            plt.hist(pos_mean[loc], 50)
            plt.savefig(os.path.join(self.save_dir, f'hist_pos_{loc}.png'))
            plt.hist(neg_mean[loc], 50)
            plt.savefig(os.path.join(self.save_dir, f'hist_neg_{loc}.png'))
            print('pos mean:', pos_mean[loc].mean(), ' pos std:', pos_mean[loc].std())
            print('neg mean:', neg_mean[loc].mean(), ' neg std:', neg_mean[loc].std())

            np.save(os.path.join(self.save_dir, f'pos_score_{loc}.npy'), pos_mean[loc].numpy())
            np.save(os.path.join(self.save_dir, f'neg_score_{loc}.npy'), neg_mean[loc].numpy())

        for k, v in ret_dict.items():
            print(f'{k}:{v:4f}')
        return ret_dict
    
    def compute_metrics_slow(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        if self.submission_prefix:
            self.format_results(results)
            return None
        
        # import copy
        # keys = copy.deepcopy(list(results[0][-1].keys()))
        # for result in results:
        #     all_points = []
        #     all_seg_preds = []
        #     for loc in result[-1].keys():
        #         all_points.append(result[-1][loc])
        #         all_seg_preds.append(result[1][loc])
        #     result[-1]['combine'] = torch.cat(all_points)
        #     result[1]['combine'] = torch.cat(all_seg_preds)
        #     for loc in keys:
        #         del result[-1][loc]
        #         del result[1][loc]

        sample_num = 100
        ths = np.linspace(0, 1, sample_num)
        cnt = dict()
        fps = dict()
        fp_dens = dict()
        den = dict()

        pos_mean = dict()
        pos_std = dict()
        neg_mean = dict()
        neg_std = dict()
        for loc in results[0][-1].keys():
            cnt[loc] = [0 for _ in range(sample_num)]
            fps[loc] = [0 for _ in range(sample_num)]
            fp_dens[loc] = [0 for _ in range(sample_num)]
            den[loc] = 0

            pos_mean[loc] = []
            pos_std[loc] = 0
            neg_mean[loc] = []
            neg_std[loc] = 0
        for result in results:
            gt_bboxes_3d = result[0]['gt_bboxes_3d'].cuda()
            gt_labels_3d = result[0]['gt_labels_3d']
            for loc in result[-1].keys():
                point_xyz = result[-1][loc]
                point_xyz = point_xyz.cuda()
                point_cls_labels_single = point_xyz.new_zeros(
                    point_xyz.shape[0]).long()
                enlarged_gt_boxes = gt_bboxes_3d.enlarged_box(self.extra_width)

                box_idxs_of_pts = gt_bboxes_3d.points_in_boxes_part(point_xyz).long()
                extend_box_idxs_of_pts = enlarged_gt_boxes.points_in_boxes_part(
                    point_xyz).long()
                box_fg_flag = box_idxs_of_pts >= 0
                fg_flag = box_fg_flag.clone()
                ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
                point_cls_labels_single[ignore_flag] = -1
                #gt_box_of_fg_points = gt_labels_3d[box_idxs_of_pts[fg_flag].cpu()]
                gt_box_of_fg_points = box_idxs_of_pts[fg_flag]
                # point_cls_labels_single[
                #     fg_flag] = 1 if self.num_classes == 1 else\
                #     gt_box_of_fg_points.long()
                point_cls_labels_single[fg_flag] = gt_box_of_fg_points.long()+1
                point_cls_labels_single = point_cls_labels_single.cpu()
                # th = 0.3
                bg_flag = (~ignore_flag) & (~fg_flag)
                seg_preds = result[1][loc].sigmoid()
                for k, th in enumerate(ths):
                    seg_labels = seg_preds > th
                    for i in range(len(gt_bboxes_3d)):
                        if seg_labels[point_cls_labels_single==(i+1)].sum() > 0:
                            cnt[loc][k] += 1
                    #fps[k] += seg_labels[bg_flag].sum()
                        fps[loc][k] += seg_labels[point_cls_labels_single==(i+1)].sum()
                    fp_dens[loc][k] += seg_labels[~ignore_flag].sum()

                den[loc] += len(gt_bboxes_3d)

                pos_mean[loc] += [seg_preds[fg_flag]]
                neg_mean[loc] += [seg_preds[bg_flag]]
        
        ret_dict = dict()
        for loc in results[0][-1].keys():
            covs = [c/den[loc] for c in cnt[loc]]
            #pres = [fp/fp_den for fp in fps]
            pres = [fp/fp_den if fp_den != 0 else 1 for fp,fp_den in zip(fps[loc],fp_dens[loc])]
            AR = 0
            AP = 0
            for cov in covs:
                AR += cov / sample_num
            for pre in pres:
                AP += pre / sample_num
            ret_dict[f'AR_{loc}'] = AR
            ret_dict[f'AP_{loc}'] = AP

            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(ths, covs)
            plt.savefig(f'AR_r50_16dim_{loc}.png')
            plt.figure()
            plt.plot(ths, pres)
            plt.savefig(f'AP_r50_16dim_{loc}.png')

            pos_mean[loc] = torch.cat(pos_mean[loc])
            neg_mean[loc] = torch.cat(neg_mean[loc])
            plt.figure()
            plt.hist(pos_mean[loc], 50)
            plt.savefig(f'hist_pos_{loc}.png')
            plt.hist(neg_mean[loc], 50)
            plt.savefig(f'hist_neg_{loc}.png')
            print('pos mean:', pos_mean[loc].mean(), ' pos std:', pos_mean[loc].std())
            print('neg mean:', neg_mean[loc].mean(), ' neg std:', neg_mean[loc].std())

            torch.save(pos_mean[loc], f'pos_score_{loc}.pt')
            torch.save(neg_mean[loc], f'neg_score_{loc}.pt')

        for k, v in ret_dict.items():
            print(f'{k}:{v:4f}')
        return ret_dict
