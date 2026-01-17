_base_ = [
    '../_base_/datasets/vod-3d-3class.py',
    '../_base_/models/voxelnext_voxel005_vod_multimodal_r50pytorch.py',
    '../_base_/schedules/cyclic-40e.py', '../_base_/default_runtime.py'
]
#runner_type = 'FastRunner'
point_cloud_range = [0, -25.6, -3, 51.2, 25.6, 2]
class_names = ['Pedestrian', 'Cyclist', 'Car']
input_modality = dict(use_lidar=True, use_camera=True)
metainfo = dict(classes=class_names)

model = dict(
    data_preprocessor=dict(
        voxel_layer=dict(point_cloud_range=point_cloud_range)),
    pts_bbox_head=dict(bbox_coder=dict(pc_range=point_cloud_range[:2])),
    pts_middle_encoder=dict(
        optional_cfg=dict(
            query_init_feat_src = ['img', 'voxel']
        )),
    # model training and testing settings
    train_cfg=dict(pts=dict(point_cloud_range=point_cloud_range)),
    test_cfg=dict(pts=dict(pc_range=point_cloud_range[:2])))

dataset_type = 'VodDataset'
data_root = 'data/vod/'
backend_args = None

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=7,
        use_dim=7,
        backend_args=backend_args),
    dict(type='LoadImageFromFile', 
         backend_args=backend_args,
         color_type='unchanged',
         to_float32=False),
    # dict(type='LoadPrecomputeImgFeats',
    #      data_path='/home/lhs/mmdetection3d/data/precompute_img_feats'),
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
        keys=['points', 'img', 'gt_labels_3d', 'gt_bboxes_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=7,
        use_dim=7,
        backend_args=backend_args),
    dict(type='LoadImageFromFile', 
         backend_args=backend_args,
         to_float32=False,
         color_type='unchanged',),
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
    dict(type='Pack3DDetInputs', keys=['points', 'img'])
]

train_dataloader = dict(
    _delete_=True,
    batch_size=8,#2,
    num_workers=8,#4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,#2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='kitti_infos_train.pkl',
            data_prefix=dict(pts='training/velodyne_reduced',img='training/image_2'),
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            #demo_load=True,
            metainfo=metainfo,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR',
            backend_args=backend_args)))
test_dataloader = dict(
    batch_size=8,
    dataset=dict(pipeline=test_pipeline, 
                 metainfo=dict(classes=class_names),
                 data_prefix=dict(pts='training/velodyne_reduced',img='training/image_2'),
                 modality=input_modality,
                 #demo_load=True,
                 )
            )
val_dataloader = dict(
    dataset=dict(pipeline=test_pipeline, 
                 metainfo=dict(classes=class_names),
                 data_prefix=dict(pts='training/velodyne_reduced',img='training/image_2'),
                 modality=input_modality,
                 #demo_load=True,
                 )
            )

train_cfg = dict(val_interval=2)

#lr = 0.003
lr = 0.001
#lr = 0.0002
#lr = 0.00005
# The optimizer follows the setting in SECOND.Pytorch, but here we use
# the official AdamW optimizer implemented by PyTorch.
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01),
    clip_grad=dict(max_norm=10, norm_type=2))
# learning rate
param_scheduler = [
    # learning rate scheduler
    # During the first 16 epochs, learning rate increases from 0 to lr * 10
    # during the next 24 epochs, learning rate decreases from lr * 10 to
    # lr * 1e-4
    dict(
        type='CosineAnnealingLR',
        T_max=16,
        eta_min=lr * 10,
        begin=0,
        end=16,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=24,
        eta_min=lr * 1e-4,
        begin=16,
        end=40,
        by_epoch=True,
        convert_to_iter_based=True),
    # momentum scheduler
    # During the first 16 epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next 24 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type='CosineAnnealingMomentum',
        T_max=16,
        eta_min=0.85 / 0.95,
        begin=0,
        end=16,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=24,
        eta_min=1,
        begin=16,
        end=40,
        by_epoch=True,
        convert_to_iter_based=True)
]


default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1))
train_cfg = dict(by_epoch=True, max_epochs=40, val_interval=40)

val_evaluator = [
    # dict(
    #     type='FGSegMetric',
    #     backend_args=backend_args),
    dict(
        type='KittiMetric',
        ann_file=data_root + 'kitti_infos_val.pkl',
        metric='bbox',
        backend_args=backend_args)
    ]
test_evaluator = val_evaluator

#load_from = '/home/lhs/mmdetection3d/exps/vx_mm_seg_r50pt_ms_stage3/epoch_7.pth'
#load_from = '/home/lhs/mmdetection3d/exps/vx_mm_seg_r50pt_ms_det_fixbug/epoch_40.pth'