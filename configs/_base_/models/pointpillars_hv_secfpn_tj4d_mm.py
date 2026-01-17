voxel_size = [0.16, 0.16, 6]

model = dict(
    type='VoxelNetFuse',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        pad_size_divisor=32,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        voxel=True,
        voxel_layer=dict(
            max_num_points=32,  # max_points_per_voxel
            point_cloud_range=[0, -39.68, -4, 69.12, 39.68, 2],
            voxel_size=voxel_size,
            max_voxels=(16000, 40000))),
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
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
    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=5,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=[0, -39.68, -4, 69.12, 39.68, 2]),
    middle_encoder=dict(
        type='PointPillarsScatterFuse', 
        in_channels=64, 
        output_shape=[496, 432],
        voxel_layer=dict(
            max_num_points=10,
            point_cloud_range=[0, -39.68, -4, 69.12, 39.68, 2],
            voxel_size=[0.16, 0.16, 0.15],
            max_voxels=(16000, 40000)),
        sparse_shape=[41, 496, 432],
        out_channels=128,
        num_features=5,
        loss_seg_weight=1.0),
    backbone=dict(
        type='SECOND',
        in_channels=64+64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=4,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        assign_per_class=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                [0, -39.68, -1.163, 69.12, 39.68, -1.163],
                [0, -39.68, -1.353, 69.12, 39.68, -1.353],
                [0, -39.68, -1.363, 69.12, 39.68, -1.363],
                [0, -39.68, -1.403, 69.12, 39.68, -1.403]
            ],
            sizes=[[0.8, 0.6, 1.69], [1.77, 0.78, 1.60], [4.56, 1.84, 1.70], [10.76, 2.66, 3.47]],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False,
            loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        assigner=[
            dict(  # for Pedestrian
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(  # for Cyclist
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(  # for Car
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1),
            dict(  # for Truck
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1),
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=4096,
        max_num=500))
