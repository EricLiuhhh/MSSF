# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from functools import partial
import json
import numpy as np
from tqdm import tqdm
import umap
from sklearn.manifold import TSNE
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision.transforms.functional import resize, InterpolationMode
from mmengine.config import Config, ConfigDict, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.dataset import Compose
from mmengine.registry import (TRANSFORMS, EVALUATOR, FUNCTIONS,
                               HOOKS, LOG_PROCESSORS, LOOPS, MODEL_WRAPPERS,
                               MODELS, OPTIM_WRAPPERS, PARAM_SCHEDULERS,
                               RUNNERS, VISUALIZERS, DefaultScope)
from mmdet3d.custom_runner import FastRunner
from mmdet3d.utils import register_all_modules
COCO_PANOPTIC_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other', 'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged', 'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged', 'food-other-merged', 'building-other-merged', 'rock-merged', 'wall-other-merged', 'rug-merged']
# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet3D test (and eval) a model')
    parser.add_argument('--config', default='configs/_base_/models/voxelnext_voxel005_vod_multimodal_vis.py', help='test config file path')
    args = parser.parse_args()
    return args



def main():
    register_all_modules()
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    input_trans = Compose(cfg.train_pipeline)
    preprocessor = MODELS.build(cfg.model.data_preprocessor)
    collate_fn_cfg = dict(type='pseudo_collate')
    if isinstance(collate_fn_cfg, dict):
        collate_fn_type = collate_fn_cfg.pop('type')
        if isinstance(collate_fn_type, str):
            collate_fn = FUNCTIONS.get(collate_fn_type)
        else:
            collate_fn = collate_fn_type
        collate_fn = partial(collate_fn, **collate_fn_cfg)  # type: ignore

    
    backbone = MODELS.build(cfg.model.img_backbone)
    fpn = MODELS.build(cfg.model.img_neck)

    backbone.eval()
    backbone.cuda()
    fpn.eval()
    fpn.cuda()

    backbone.init_weights()
    fpn.init_weights()

    with open('/home/lhs/mmdetection3d/data/vod/ImageSets/val.txt', 'r') as f:
        frames = f.readlines()
    all_frames = sorted([s.strip() for s in frames]) 

    all_feats = []
    all_labels = []
    cnt = 0
    mapper = TSNE(n_components=2, n_iter=5000)
    #mapper = umap.UMAP()
    interval = 5
    for frame_id in tqdm(all_frames):
        if cnt % interval == 0:
            inputs={'img_path':f'data/vod/training/image_2/{frame_id}.jpg'}
            with torch.no_grad():
                batch_data = preprocessor(collate_fn([input_trans(inputs)]), training=False)
                bb_feats = backbone(batch_data['inputs']['imgs'].cuda())
                fpn_feats = fpn(bb_feats)
            
            #feats = bb_feats[-1][0].permute(1,2,0).reshape((-1, 2048))
            feats = fpn_feats[0][0].permute(1,2,0).cpu()#.reshape((-1, 256))
            #feats_2d = mapper.fit_transform(feats.detach().cpu().numpy())
            # plt.scatter(feats_2d[:,0], feats_2d[:,1])#, c=colors[inverse_indices])
            # plt.savefig('vis_feats.png')
        
            with open(f'/home/lhs/SLidR/superpixels/vod/superpixels_seem/{frame_id}.json', 'r') as f:
                meta = json.load(f)
            meta.append({'id':0, 'category_id':200, 'isthing':False})
            
            sp = np.array(Image.open(f'/home/lhs/SLidR/superpixels/vod/superpixels_seem/{frame_id}.png'))
            # sp_upsample = resize(torch.tensor(sp).unsqueeze(0), batch_data['inputs']['imgs'].shape[2:], InterpolationMode.NEAREST)[0]
            # new_sp = np.zeros_like(sp)
        
            # for d in meta:
            #     cat_id = d['category_id']
            #     if cat_id >= 3:
            #         new_sp[sp==d['id']] = cat_id
            #     else:
            #         new_sp[sp==d['id']] = cat_id
            # sp = new_sp

            sp_downsample = resize(torch.tensor(sp).unsqueeze(0), fpn_feats[0].shape[2:], InterpolationMode.NEAREST)[0]
            # unique_ids, inverse_indices = np.unique(sp_downsample.numpy(), return_inverse=True)
            # colors = plt.cm.get_cmap('rainbow', len(unique_ids))(np.arange(len(unique_ids)))

            #feats_cls = torch.zeros([colors.shape[0], 256])
            #labels = torch.zeros([colors.shape[0]])
            feats_cls = []
            labels = []
            for d in meta:
                cat_id = d['category_id']
                isthing = d['isthing']
                if isthing and cat_id in [0, 1, 2, 3]:
                    mask = sp_downsample==d['id']
                    if mask.sum() > 0:
                        # feats_cls[d['id']] = feats[mask].mean(0)
                        # labels[d['id']] = cat_id
                        feats_cls.append(feats[mask].mean(0))
                        labels.append(cat_id)
                        # plt.imshow(mask)
                        # plt.title(f'{COCO_PANOPTIC_CLASSES[cat_id]}')
                        # plt.savefig('z.png')
                        # plt.clf()
            if len(feats_cls) > 0:
                feats_cls = torch.stack(feats_cls)
                labels = torch.tensor(labels)

                all_feats.append(feats_cls)
                all_labels.append(labels)
            torch.cuda.empty_cache()
        cnt += 1
        # if cnt // interval > 200:
        #     break

    feats = torch.concat(all_feats, dim=0).numpy()
    labels = torch.concat(all_labels).numpy()

    unique_ids, inverse_indices = np.unique(labels, return_inverse=True)
    colors = plt.cm.get_cmap('rainbow', len(unique_ids))(np.arange(len(unique_ids)))
    feats_2d = mapper.fit_transform(feats)

    for i in unique_ids:
        mask = (inverse_indices == i)
        label = COCO_PANOPTIC_CLASSES[i]
        plt.scatter(feats_2d[mask][:,0], feats_2d[mask][:,1], c=colors[inverse_indices[mask]], label=label)
    #plt.scatter(feats_2d[:,0], feats_2d[:,1], c=colors[inverse_indices])
    plt.legend()
    plt.savefig('vis_feats.png')
    
    # _, inverse_indices = np.unique(sp, return_inverse=True)
    # colored_sp = colors[inverse_indices].reshape(sp.shape[0], sp.shape[1], 4)
    # colored_sp=(colored_sp[:,:,:3]*255).astype(np.uint8)

    
    # plt.figure()
    # plt.imshow(colored_sp)
    # plt.savefig('test_sp.png')
if __name__ == '__main__':
    main()
