_base_ = [
    '../_base_/datasets/vod-3d-3class.py',
    #'../_base_/models/voxelnext_voxel005_vod_multimodal_r50pytorch.py',
    '../_base_/schedules/cyclic-40e.py', '../_base_/default_runtime.py'
]

#runner_type = 'FastRunner'
voxel_size = [0.05, 0.05, 0.125]
point_cloud_range = [0, -25.6, -3, 51.2, 25.6, 2]
class_names = ['Pedestrian', 'Cyclist', 'Car']
input_modality = dict(use_lidar=True, use_camera=True)
metainfo = dict(classes=class_names)

model = dict(
    type='VoxelNeXtSegDet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        pad_size_divisor=32,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        voxel=True,
        voxel_layer=dict(
            max_num_points=10,
            voxel_size=voxel_size,
            max_voxels=(16000, 40000),
            point_cloud_range=point_cloud_range)),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=7),
    pts_middle_encoder=dict(
        type='SparseEncoderVoxelNeXtSegDet',
        load_from='/home/lhs/mmdetection3d/exps/vx_mm_seg_r50pt_ms_stage3_5/epoch_40.pth',
        
        in_channels=7,
        sparse_shape=[41, 1024, 1024],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128, 128), (128, 128, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0)),
        block_type='basicblock'),
    pts_bbox_head=dict(
        type='VoxelNeXtHead',
        in_channels=128,
        tasks=[
            dict(num_class=3, class_names=['Pedestrian', 'Cyclist', 'Car'])
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        share_conv_channel=128,
        bbox_coder=dict(
            type='VoxelNeXtBBoxCoder',
            post_center_range=[0, -25.6, -3, 51.2, 25.6, 2],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=7,
            pc_range=point_cloud_range[:2]),
        separate_head=dict(
            type='SeparateHeadVoxelNeXt', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='FocalLossSparse'),
        loss_bbox=dict(
            type='RegLossSparse'),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[1024, 1024, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            point_cloud_range=point_cloud_range)),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=[0, -25.6, -3, 51.2, 25.6, 2],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2,
            pc_range=point_cloud_range[:2])),
    seg_branch=dict(
        type='VoxelNeXtFuse',
        data_preprocessor=dict(
            type='Det3DDataPreprocessor',
            pad_size_divisor=32,
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            voxel=True,
            voxel_layer=dict(
                max_num_points=10,
                voxel_size=voxel_size,
                max_voxels=(16000, 40000),
                point_cloud_range=point_cloud_range)),
        img_backbone=dict(
            type='mmdet.ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=4,#1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(
                type='Pretrained',
                checkpoint='checkpoints/htc_r50_fpn_coco-20e_20e_nuim_20201008_211415-d6c60a2c.pth',
                prefix='backbone.')),
        img_neck=dict(
            type='mmdet.FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5,
            init_cfg=dict(
                type='Pretrained',
                checkpoint='checkpoints/htc_r50_fpn_coco-20e_20e_nuim_20201008_211415-d6c60a2c.pth',
                prefix='neck.')),
        pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=7),
        pts_middle_encoder=dict(
            type='SparseEncoderVoxelNeXtFuse',
            in_channels=7,
            sparse_shape=[41, 1024, 1024],
            output_channels=256,
            base_channels=32,
            order=('conv', 'norm', 'act'),
            encoder_channels=((32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256), (256, 256, 256), (256, 256, 256)),
            encoder_paddings=((0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0)),
            block_type='basicblock'),
        # pts_backbone=dict(
        #     type='SECOND',
        #     in_channels=256,
        #     out_channels=[128, 256],
        #     layer_nums=[5, 5],
        #     layer_strides=[1, 2],
        #     norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        #     conv_cfg=dict(type='Conv2d', bias=False)),
        # pts_neck=dict(
        #     type='SECONDFPN',
        #     in_channels=[128, 256],
        #     out_channels=[256, 256],
        #     upsample_strides=[1, 2],
        #     norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        #     upsample_cfg=dict(type='deconv', bias=False),
        #     use_conv_for_no_stride=True),
        pts_bbox_head=dict(
            type='VoxelNeXtHead',
            in_channels=256,
            # tasks=[
            #     dict(num_class=1, class_names=['Pedestrian']),
            #     dict(num_class=1, class_names=['Cyclist']),
            #     dict(num_class=1, class_names=['Car']),
            # ],
            tasks=[
                dict(num_class=3, class_names=['Pedestrian', 'Cyclist', 'Car'])
            ],
            common_heads=dict(
                reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
            share_conv_channel=256,
            bbox_coder=dict(
                type='VoxelNeXtBBoxCoder',
                post_center_range=[0, -25.6, -3, 51.2, 25.6, 2],
                max_num=500,
                score_threshold=0.1,
                out_size_factor=8,
                voxel_size=voxel_size[:2],
                code_size=7,
                pc_range=point_cloud_range[:2]),
            separate_head=dict(
                type='SeparateHeadVoxelNeXt', init_bias=-2.19, final_kernel=3),
            loss_cls=dict(type='FocalLossSparse'),
            loss_bbox=dict(
                type='RegLossSparse'),
            norm_bbox=True),
        # model training and testing settings
        train_cfg=dict(
            pts=dict(
                grid_size=[1024, 1024, 40],
                voxel_size=voxel_size,
                out_size_factor=8,
                dense_reg=1,
                gaussian_overlap=0.1,
                max_objs=500,
                min_radius=2,
                code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                point_cloud_range=point_cloud_range)),
        test_cfg=dict(
            pts=dict(
                post_center_limit_range=[0, -25.6, -3, 51.2, 25.6, 2],
                max_per_img=500,
                max_pool_nms=False,
                min_radius=[4, 12, 10, 1, 0.85, 0.175],
                score_threshold=0.1,
                out_size_factor=8,
                voxel_size=voxel_size[:2],
                nms_type='rotate',
                pre_max_size=1000,
                post_max_size=83,
                nms_thr=0.2,
                pc_range=point_cloud_range[:2]))))

# model = dict(
#     data_preprocessor=dict(
#         voxel_layer=dict(point_cloud_range=point_cloud_range)),
#     pts_bbox_head=dict(bbox_coder=dict(pc_range=point_cloud_range[:2])),
#     # model training and testing settings
#     train_cfg=dict(pts=dict(point_cloud_range=point_cloud_range)),
#     test_cfg=dict(pts=dict(pc_range=point_cloud_range[:2])))

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
    sampler=dict(type='DefaultSampler', shuffle=True),#shuffle=False),
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


lr = 0.003
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
        T_max=14,
        eta_min=lr * 10,
        begin=0,
        end=14,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=26,
        eta_min=lr * 1e-4,
        begin=14,
        end=40,
        by_epoch=True,
        convert_to_iter_based=True),
    # momentum scheduler
    # During the first 16 epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next 24 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type='CosineAnnealingMomentum',
        T_max=14,
        eta_min=0.85 / 0.95,
        begin=0,
        end=14,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=26,
        eta_min=1,
        begin=14,
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