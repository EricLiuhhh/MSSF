_base_ = [
    '../_base_/datasets/tj4d-3d-4class.py',
    '../_base_/models/centerpoint_pillar016_second_secfpn_tj4d_nocbgs.py',
    '../_base_/schedules/cyclic-40e.py', '../_base_/default_runtime.py'
]

point_cloud_range = [0, -39.68, -4, 69.12, 39.68, 2]
class_names = ['Pedestrian', 'Cyclist', 'Car', 'Truck']
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=class_names)
# [X, Y, Z, V_r, Range, Power, Alpha, Beta, V_r_comp]
# only use [X, Y, Z, V_r, Power]
# use_dim = [0, 1, 2, 3, 5]
# only use [X, Y, Z, V_r_comp, Power]
use_dim = [0, 1, 2, 8, 5]

model = dict(
    data_preprocessor=dict(
        voxel_layer=dict(point_cloud_range=point_cloud_range)),
    pts_voxel_encoder=dict(point_cloud_range=point_cloud_range),
    pts_bbox_head=dict(bbox_coder=dict(pc_range=point_cloud_range[:2])),
    # model training and testing settings
    train_cfg=dict(pts=dict(point_cloud_range=point_cloud_range)),
    test_cfg=dict(pts=dict(pc_range=point_cloud_range[:2])))

dataset_type = 'TJ4DDataset'
data_root = 'data/tj4d/'
backend_args = None

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=9,
        use_dim=use_dim,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    #dict(type='ObjectSample', db_sampler=db_sampler, use_ground_plane=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_labels_3d', 'gt_bboxes_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=9,
        use_dim=use_dim,
        backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]

train_dataloader = dict(
    _delete_=True,
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='kitti_infos_train.pkl',
            data_prefix=dict(pts='training/velodyne_reduced'),
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            metainfo=metainfo,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR',
            backend_args=backend_args)))
test_dataloader = dict(
    dataset=dict(pipeline=test_pipeline, metainfo=dict(classes=class_names)))
val_dataloader = dict(
    dataset=dict(pipeline=test_pipeline, metainfo=dict(classes=class_names)))

#train_cfg = dict(val_interval=2)
val_evaluator = [
    # dict(
    #     type='KittiMetric',
    #     ann_file=data_root + 'kitti_infos_val.pkl',
    #     metric='bbox',
    #     backend_args=backend_args),
    dict(
        type='TJ4DMetric',
        ann_file=data_root + 'kitti_infos_val.pkl',
        metric='bbox',
        backend_args=backend_args)
]
test_evaluator = val_evaluator
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1))

train_cfg = dict(val_interval=40)