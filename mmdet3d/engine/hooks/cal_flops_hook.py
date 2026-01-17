from typing import Sequence
import prettytable
import torch
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmdet3d.registry import HOOKS
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE
if IS_SPCONV2_AVAILABLE:
    import spconv.pytorch as spconv
else:
    raise NotImplementedError
from mmdet3d.utils.custom_ops_flops import dcn_flops_counter_hook_thop, multihead_attention_counter_hook, count_window_msa
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2dPack
from torch.nn import MultiheadAttention
from mmcv.cnn.bricks.transformer import MultiheadAttention as MultiheadAttention_mmcv
from mmdet.models.backbones.swin import WindowMSA, ShiftWindowMSA
from thop import profile, clever_format, profile_acts

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        result = f"average value: {self.avg:.3f}"
        return result
    

@HOOKS.register_module()
class CalFLOPsHook(Hook):

    def __init__(self):
        self.flops_meter = AverageMeter()
        self.acts_meter = AverageMeter()
        self.params = None

        self.predefined_pts = self.gen_predefined_pts(3000, 7).to(torch.float32).cuda()

    def gen_predefined_pts(self, N, C, pc_range=(0, -25.6, -3, 51.2, 25.6, 2)):
        # 定义范围
        xmin, ymin, zmin, xmax, ymax, zmax = pc_range
        # 随机生成 1000 个 3D 点
        points = torch.empty(N, C)
        points[:, 0] = torch.rand(N) * (xmax - xmin) + xmin  # x 范围
        points[:, 1] = torch.rand(N) * (ymax - ymin) + ymin  # y 范围
        points[:, 2] = torch.rand(N) * (zmax - zmin) + zmin  # z 范围
        points[:, 3:] = torch.rand(N, C-3)

        return points

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch = None, outputs = None) -> None:
        data_batch['inputs']['points'][0] = self.predefined_pts
        data = runner.model.data_preprocessor(data_batch, False)
        macs, params, acts, layer_infos = profile_acts(runner.model, inputs=(data['inputs'], data['data_samples'], 'predict'),
                           custom_ops={spconv.SubMConv3d: spconv.SubMConv3d.count_your_model,
                                       spconv.SubMConv2d: spconv.SubMConv2d.count_your_model,
                                       spconv.SparseConv3d: spconv.SparseConv3d.count_your_model,
                                       spconv.SparseConv2d: spconv.SparseConv2d.count_your_model,
                                       ModulatedDeformConv2dPack: dcn_flops_counter_hook_thop,
                                       MultiheadAttention: multihead_attention_counter_hook,
                                       WindowMSA: count_window_msa
                                       #MultiheadAttention_mmcv: multihead_attention_counter_hook,
                                       }, verbose=False, report_missing=True, ret_layer_info=True
                           )
        
        self.flops_meter.update(macs)
        self.acts_meter.update(acts)
        self.params = params

        macs, params, acts = clever_format([self.flops_meter.avg, self.params, self.acts_meter.avg], "%.3f")
        tab = prettytable.PrettyTable(['Module', 'FLOPs', 'Params', 'ACTs'])
        tab.add_row(['All', f'{macs}', f'{params}', f'{acts}'])
        
        for k, v in layer_infos.items():
            macs, params, acts = clever_format([v[0], v[1], 0.0], "%.3f")
            tab.add_row([getattr(runner.model, k).__class__.__name__, f'{macs}', f'{params}', '-'])

        print('Model Summary:\n'+tab.get_string())

    def after_test(self, runner: Runner) -> None:
        macs, params, acts = clever_format([self.flops_meter.avg, self.params, self.acts_meter.avg], "%.3f")
        tab = prettytable.PrettyTable(['Module', 'FLOPs', 'Params', 'ACTs'])
        tab.add_row(['All', f'{macs}', f'{params}', f'{acts}'])
        runner.logger.info('Model Summary:\n'+tab.get_string())