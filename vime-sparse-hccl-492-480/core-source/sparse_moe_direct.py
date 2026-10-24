# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guarded sparse writes into Ascend Qwen3-MoE inference-layout weights."""

import logging
import re
from collections import Counter

import torch

logger = logging.getLogger(__name__)


@torch.no_grad()
def apply_direct_moe_patches(model, patches):
    """Apply supported patches; preserve original objects/order for fallback.

    The caller must validate every patch before invoking this function and must
    not enter temporary checkpoint-loader views until this function returns.
    """
    pairs = [(patch, list(patch.shape)) for patch in patches]
    remaining = _apply_direct_moe_patches(model, pairs)
    remaining_ids = {id(patch) for patch, _ in remaining}
    handled = {patch.name for patch in patches if id(patch) not in remaining_ids}
    return [patch for patch, _ in remaining], handled


def _apply_moe_group(param, entries, shape, tp_rank, tp_size, down, *, verify=False):
    """Compact once per runtime weight, preserving HF absolute-copy semantics.

    SGLang avoids per-expert dynamic boolean indexing in its masked loader.
    Here the same TP selection is performed for a whole runtime w13/w2 group.
    Explicit output_size prevents repeat_interleave from reading a device count.
    """
    lengths = [patch.indices.numel() for patch, _, _ in entries]
    total = sum(lengths)
    if not total:
        return
    device = entries[0][0].indices.device
    indices = torch.cat([patch.indices for patch, _, _ in entries]).long()
    values = torch.cat([patch.values for patch, _, _ in entries])
    width = shape[1 if down else 0] // tp_size
    descriptors = torch.tensor(
        [[local_expert, width if projection == "up_proj" else 0] for _, local_expert, projection in entries],
        dtype=torch.int64,
        device=device,
    )
    descriptors = torch.repeat_interleave(
        descriptors,
        torch.tensor(lengths, dtype=torch.int64, device=device),
        dim=0,
        output_size=total,
    )
    row, col = indices // shape[1], indices % shape[1]
    coordinate = col if down else row
    selected = (
        ((coordinate >= tp_rank * width) & (coordinate < (tp_rank + 1) * width) & ~torch.isnan(values))
        .nonzero()
        .view(-1)
    )
    if down:
        target = descriptors[:, 0] * width * shape[0] + (col - tp_rank * width) * shape[0] + row
    else:
        target = (
            descriptors[:, 0] * shape[1] * (2 * width) + col * (2 * width) + descriptors[:, 1] + row - tp_rank * width
        )
    local_indices = target.index_select(0, selected)
    local_values = values.index_select(0, selected)
    flat = param.data.view(-1)
    flat.index_copy_(0, local_indices, local_values)
    if verify:
        if not torch.equal(flat.index_select(0, local_indices), local_values):
            raise RuntimeError("Sparse MoE rank-local verification failed")


def _apply_direct_moe_patches(
    model,
    patches,
    *,
    verify=False,
):
    # Exact guards keep unfamiliar models, quantization and layouts on the
    # native loader. Static EP is supported through its global-to-local map.
    if getattr(getattr(model, "config", None), "model_type", None) != "qwen3_moe":
        return patches
    # Preserve ordered overwrite semantics for split/repeated parameter names.
    names = [patch.name for patch, _ in patches]
    if len(names) != len(set(names)):
        logger.info("Sparse MoE apply: direct_patches=0 fallback_patches=%d reason=repeated_names", len(patches))
        return patches
    from vllm_ascend.ops.fused_moe.routed_experts import (
        AscendRoutedExperts,
        AscendUnquantizedFusedMoEMethod,
    )

    fallback, grouped = [], {}
    reasons = Counter()
    skipped_nonlocal = 0
    for patch, shape in patches:
        match = re.fullmatch(r"(.+\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", patch.name)
        if match is None or len(shape) != 2:
            reasons["non_expert_or_shape"] += 1
            fallback.append((patch, shape))
            continue
        prefix, expert_text, projection = match.groups()
        try:
            runner = model.get_submodule(prefix)
        except AttributeError:
            reasons["missing_module"] += 1
            fallback.append((patch, shape))
            continue
        layer = getattr(runner, "routed_experts", runner)
        if type(layer) is not AscendRoutedExperts or type(layer.quant_method) is not AscendUnquantizedFusedMoEMethod:
            reasons["unsupported_method"] += 1
            fallback.append((patch, shape))
            continue
        config = layer.moe_config
        parallel = config.moe_parallel_config
        expert = int(expert_text)
        tp_size, tp_rank = parallel.tp_size, config.tp_rank
        param = getattr(layer, "w2_weight" if projection == "down_proj" else "w13_weight", None)
        shard_dim = 1 if projection == "down_proj" else 0
        expected = (
            (shape[1] // tp_size, shape[0]) if projection == "down_proj" else (shape[1], 2 * (shape[0] // tp_size))
        )
        expert_map = getattr(layer, "ascend_expert_map", None)
        eplb_enabled = getattr(config, "enable_eplb", False) or getattr(layer.quant_method, "dynamic_eplb", False)
        if (
            eplb_enabled
            or not config.is_act_and_mul
            or param is None
            or getattr(param, "is_transposed", False)
            or shape[shard_dim] % tp_size
            or param.ndim != 3
            or tuple(param.shape[1:]) != expected
            or param.dtype != torch.bfloat16
            or param.dtype != patch.values.dtype
            or not param.is_contiguous()
            or not _supports_logical_flat_index_copy(param)
        ):
            reasons["layout_or_mapping_guard"] += 1
            fallback.append((patch, shape))
            continue
        if parallel.ep_size == 1:
            local_expert = expert
        elif expert_map is not None and not eplb_enabled:
            cached_map = getattr(layer, "_sparse_expert_map_cpu", None)
            if cached_map is None:
                cached_map = expert_map.detach().cpu()
                layer._sparse_expert_map_cpu = cached_map
            if not 0 <= expert < cached_map.numel():
                raise IndexError(f"Global expert {expert} outside expert map")
            local_expert = int(cached_map[expert])
            if local_expert < 0:
                # The expert belongs to another EP rank.  Drop it before any
                # full-shape temporary is allocated by the native loader.
                skipped_nonlocal += 1
                continue
        else:
            local_expert = -1
        if not 0 <= local_expert < param.shape[0]:
            reasons["layout_or_mapping_guard"] += 1
            fallback.append((patch, shape))
            continue
        group = grouped.setdefault(
            id(param),
            (param, [], tuple(shape), tp_rank, tp_size, projection == "down_proj"),
        )
        if group[2:] != (tuple(shape), tp_rank, tp_size, projection == "down_proj"):
            reasons["inconsistent_group_shape"] += 1
            fallback.append((patch, shape))
            continue
        group[1].append((patch, local_expert, projection))
    for param, entries, shape, tp_rank, tp_size, down in grouped.values():
        _apply_moe_group(param, entries, shape, tp_rank, tp_size, down, verify=verify)
    logger.info(
        "Sparse MoE apply: direct_patches=%d runtime_parameters=%d "
        "fallback_patches=%d skipped_nonlocal=%d fallback_reasons=%s",
        sum(len(group[1]) for group in grouped.values()),
        len(grouped),
        len(fallback),
        skipped_nonlocal,
        dict(reasons),
    )
    return fallback


def _supports_logical_flat_index_copy(tensor: torch.Tensor) -> bool:
    """Reject transformed Ascend layouts until physical mapping is defined."""
    if tensor.device.type != "npu":
        return True
    try:
        import torch_npu
    except ImportError:
        return False
    npu_format = str(torch_npu.get_npu_format(tensor)).upper()
    return npu_format in {"ND", "0", "2"}
