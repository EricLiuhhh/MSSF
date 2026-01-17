voxel_size = [0.05, 0.05, 0.125]
model = dict(
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
            checkpoint='checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth',
            prefix='backbone.')),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth',
            prefix='neck.')),
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        pad_size_divisor=32,
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        voxel=True,
        voxel_layer=dict(
            point_cloud_range=[0, -25.6, -3, 51.2, 25.6, 2],
            max_num_points=10,
            voxel_size=voxel_size,
            max_voxels=(16000, 40000))),
)

train_pipeline = [
    dict(type='LoadImageFromFile', 
         backend_args=None,
         color_type='unchanged',
         to_float32=False),
    dict(
        type='Pack3DDetInputs',
        keys=['img'])
]