# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint-coordinate sparse patches for the HCCL transport."""

import math

import torch
from vllm.distributed.weight_transfer.sparse_nccl_engine import SparseNCCLTrainerWeightTransferEngine
from vllm.distributed.weight_transfer.sparse_nccl_engine import SparseWeightPatch as SparseWeightPatch

SPARSE_HCCL_VALUE_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def validate_sparse_patch(patch: SparseWeightPatch) -> None:
    """Reject invalid payloads before starting a worker lifecycle or collective."""
    _validate_sparse_patches([patch])


def _validate_sparse_patches(patches: list[SparseWeightPatch]) -> None:
    """Read one aggregate flag per device; locate invalid payloads on failure."""
    flags: dict[torch.device, list[torch.Tensor]] = {}
    for patch in patches:
        _validate_sparse_patch_structure(patch)
        for flag in _sparse_patch_error_flags(patch):
            flags.setdefault(flag.device, []).append(flag)
    if any(torch.stack(device_flags).any().item() for device_flags in flags.values()):
        for patch in patches:
            index_error, value_error = _sparse_patch_error_flags(patch)
            if index_error.item():
                raise ValueError(f"Sparse indices out of range or duplicated: {patch.name}")
            if value_error.item():
                raise ValueError(f"Sparse values cannot contain NaN: {patch.name}")


def _validate_sparse_patch_structure(patch: SparseWeightPatch) -> None:
    # Keep this standalone validator complete while sharing the trainer's
    # structural checks for shape presence, int32 indices and flattened payloads.
    SparseNCCLTrainerWeightTransferEngine._validate_patch(patch)
    if not isinstance(patch.name, str) or not patch.name:
        raise ValueError("Sparse checkpoint name must be non-empty")
    if any(type(dim) is not int or dim < 0 for dim in patch.full_shape):
        raise ValueError(f"Invalid checkpoint full_shape for {patch.name}")
    if patch.values.dtype not in SPARSE_HCCL_VALUE_DTYPES:
        raise ValueError(f"Sparse values require an HCCL-supported floating dtype: {patch.name}")
    numel = math.prod(patch.full_shape)
    if numel > torch.iinfo(torch.int64).max:
        raise ValueError(f"Checkpoint shape exceeds int64 indexing: {patch.name}")


def _sparse_patch_error_flags(patch: SparseWeightPatch) -> tuple[torch.Tensor, torch.Tensor]:
    numel = math.prod(patch.full_shape)
    indices = patch.indices
    invalid = torch.zeros((), dtype=torch.bool, device=indices.device)
    if indices.numel():
        ordered = indices.sort().values
        # Comparing an int32 tensor with a larger Python scalar wraps the
        # scalar into int32. Keep the wire dtype, but widen bounds arithmetic.
        wide_indices = indices.to(dtype=torch.int64)
        invalid = torch.stack(
            [
                ((wide_indices < 0) | (wide_indices >= numel)).any(),
                (ordered[1:] == ordered[:-1]).any(),
            ]
        ).any()
    return invalid, torch.isnan(patch.values).any()
