import numpy as np
import torch
import torch.nn as nn
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2dPack


def dcn_flops_counter_hook_thop(conv_module: ModulatedDeformConv2dPack, input: tuple,
                            output: torch.Tensor) -> None:
    # Can have multiple inputs, getting the first one
    batch_size = input[0].shape[0]
    output_dims = list(output.shape[2:])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(
        np.prod(kernel_dims)) * in_channels * filters_per_channel

    active_elements_count = batch_size * int(np.prod(output_dims))

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0

    if conv_module.bias is not None:

        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops

    conv_module.total_ops += int(overall_flops)


    batch_size = input[0].shape[0]
    output_dims = list(output.shape[2:])

    kernel_dims = list(conv_module.conv_offset.kernel_size)
    in_channels = conv_module.conv_offset.in_channels
    out_channels = conv_module.conv_offset.out_channels
    groups = conv_module.conv_offset.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(
        np.prod(kernel_dims)) * in_channels * filters_per_channel

    active_elements_count = batch_size * int(np.prod(output_dims))

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0

    if conv_module.conv_offset is not None:

        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops

    conv_module.total_ops += int(overall_flops)


def multihead_attention_counter_hook(multihead_attention_module, input, output):
    flops = 0

    q, k, v = input

    batch_first = multihead_attention_module.batch_first \
        if hasattr(multihead_attention_module, 'batch_first') else False
    if batch_first:
        batch_size = q.shape[0]
        len_idx = 1
    else:
        batch_size = q.shape[1]
        len_idx = 0

    dim_idx = 2

    qdim = q.shape[dim_idx]
    kdim = k.shape[dim_idx]
    vdim = v.shape[dim_idx]

    qlen = q.shape[len_idx]
    klen = k.shape[len_idx]
    vlen = v.shape[len_idx]

    num_heads = multihead_attention_module.num_heads
    assert qdim == multihead_attention_module.embed_dim

    if multihead_attention_module.kdim is None:
        assert kdim == qdim
    if multihead_attention_module.vdim is None:
        assert vdim == qdim

    flops = 0

    # Q scaling
    flops += qlen * qdim

    # Initial projections
    flops += (
        (qlen * qdim * qdim)  # QW
        + (klen * kdim * kdim)  # KW
        + (vlen * vdim * vdim)  # VW
    )

    if multihead_attention_module.in_proj_bias is not None:
        flops += (qlen + klen + vlen) * qdim

    # attention heads: scale, matmul, softmax, matmul
    qk_head_dim = qdim // num_heads
    v_head_dim = vdim // num_heads

    head_flops = (
        (qlen * klen * qk_head_dim)  # QK^T
        + (qlen * klen)  # softmax
        + (qlen * klen * v_head_dim)  # AV
    )

    flops += num_heads * head_flops

    # final projection, bias is always enabled
    flops += qlen * vdim * (vdim + 1)

    flops *= batch_size
    multihead_attention_module.total_ops += int(flops)


def count_window_msa(m, x, y):
    embed_dims = m.embed_dims
    num_heads = m.num_heads
    B, N, C = x[0].shape
    # qkv = model.qkv(x)
    m.total_ops += B * N * embed_dims * 3 * embed_dims
    # attn = (q @ k.transpose(-2, -1))
    m.total_ops += B * num_heads * N * (embed_dims // num_heads) * N
    # x = (attn @ v)
    m.total_ops += num_heads * B * N * N * (embed_dims // num_heads)
    # x = m.proj(x)
    m.total_ops += B * N * embed_dims * embed_dims
