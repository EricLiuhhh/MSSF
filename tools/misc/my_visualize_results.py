# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import mmcv
from mmengine import mkdir_or_exist, load
from mmdet3d.visualization import Det3DLocalVisualizer

# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='My visualize results')
    parser.add_argument('results_dir', help='results_dir')
    parser.add_argument('dataset', help='dataset_type')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir', default='exps/temp')
    parser.add_argument(
        '--score-thr', type=float, default=0.1, help='bbox score threshold')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.dataset == 'vod':
        point_cloud_range = [0, -25.6, -3, 51.2, 25.6, 2]
        bev_shape = 52
    elif args.dataset == 'tj4d':
        point_cloud_range = [0, -39.68, -4, 69.12, 39.68, 2]
        bev_shape = 70
    else:
        raise NotImplementedError
    show_dir_img = osp.join(args.show_dir, 'img')
    show_dir_pcl = osp.join(args.show_dir, 'pcl')
    mkdir_or_exist(show_dir_img)
    mkdir_or_exist(show_dir_pcl)

    vis_backends = [dict(type='LocalVisBackend')]
    visualizer = dict(vis_backends=vis_backends, name='visualizer', save_dir=args.show_dir)
    visualizer = Det3DLocalVisualizer(**visualizer)
    res_dir = args.results_dir
    res_files = os.listdir(res_dir)
    res_files = sorted(res_files, key=lambda x: int(x.split('.')[0]))
    for f in tqdm(res_files):
        detsamples = load(os.path.join(res_dir, f))
        for detsample in detsamples:
            lidar_path = detsample.lidar_path
            num_pts_feats = detsample.num_pts_feats
            frame_id = osp.basename(lidar_path).split('.')[0]
            # if frame_id not in ['00234', '05039', '08253']:
            #    continue
            if frame_id not in ['010150', '100000', '310062']:
                continue
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, num_pts_feats)
            mask = (points[:,0] > point_cloud_range[0]) & (points[:,0] < point_cloud_range[3]) & (points[:,1] > point_cloud_range[1]) & (points[:,1] < point_cloud_range[4]) & (points[:,2] > point_cloud_range[2]) & (points[:,2] < point_cloud_range[5])
            points = points[mask]
            pred_instances_3d = detsample.pred_instances_3d
            pred_instances_3d = pred_instances_3d[
                    pred_instances_3d.scores_3d > args.score_thr]

            # img_path = detsample.img_path
            # metainfo_update = detsample.metainfo
            # if isinstance(img_path, list):
            #     img_path = osp.join(osp.dirname(lidar_path), '..', 'image_2', img_path[0])
            # if isinstance(detsample.metainfo['lidar2img'], list):
            #     metainfo_update['lidar2img'] = np.array(detsample.metainfo['lidar2img'])[2]
            #     metainfo_update['lidar2cam'] = np.array(detsample.metainfo['lidar2cam'])[0]
            #     metainfo_update['cam2img'] = np.array(detsample.metainfo['cam2img'])[2]

            # img = mmcv.imread(img_path, channel_order='rgb')
            # visualizer.set_image(img)
            # visualizer.draw_proj_bboxes_3d(detsample.gt_instances_3d.bboxes_3d, metainfo_update, show_face=False, line_widths=1, edge_colors='orange')
            # visualizer.draw_proj_bboxes_3d(pred_instances_3d, metainfo_update, show_face=False, line_widths=1, edge_colors='cyan')
            # visualizer.fig_save.savefig(osp.join(show_dir_img, f'{frame_id}_img.png'), dpi=visualizer.dpi, bbox_inches='tight', pad_inches=0.0)

            visualizer.set_bev_image(bev_shape=bev_shape, show_fov_line=False)
            visualizer.draw_bev_points(points, sizes=1)
            visualizer.draw_bev_bboxes(detsample.gt_instances_3d.bboxes_3d, scale=1, edge_colors='orange')
            

            visualizer.draw_bev_bboxes(pred_instances_3d.bboxes_3d, scale=1, edge_colors='cyan')
            visualizer.fig_save.savefig(osp.join(show_dir_pcl, f'{frame_id}_pcl.png'), dpi=visualizer.dpi, bbox_inches='tight', pad_inches=0.0)

if __name__ == '__main__':
    main()
