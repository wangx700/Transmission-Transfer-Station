from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.distributed.weight_transfer.sparse_weight_patch import SparseWeightPatch


@pytest.mark.parametrize("projection", ["gate_proj", "up_proj", "down_proj"])
@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("expert", [0, 2, 3])
def test_static_ep_and_tp_mapping(monkeypatch, projection, tp_rank, expert):
    from vllm.model_executor.model_loader.checkpoint_weight_patch import CheckpointWeightPatch

    from vllm_ascend.distributed.weight_transfer import sparse_moe_direct as direct
    from vllm_ascend.ops.fused_moe import routed_experts

    class Method:
        pass

    class Experts:
        pass

    monkeypatch.setattr(routed_experts, "AscendRoutedExperts", Experts)
    monkeypatch.setattr(routed_experts, "AscendUnquantizedFusedMoEMethod", Method)
    layer = Experts()
    layer.quant_method = Method()
    layer.moe_config = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(tp_size=2, ep_size=2),
        tp_rank=tp_rank,
        enable_eplb=False,
        is_act_and_mul=True,
    )
    layer.ascend_expert_map = torch.tensor([-1, -1, 0, 1])
    layer.w13_weight = torch.nn.Parameter(torch.arange(32, dtype=torch.bfloat16).reshape(2, 4, 4))
    layer.w2_weight = torch.nn.Parameter(torch.arange(16, dtype=torch.bfloat16).reshape(2, 2, 4))
    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_moe"), get_submodule=lambda name: layer)
    param = layer.w2_weight if projection == "down_proj" else layer.w13_weight
    expected = param.detach().clone()
    patch = CheckpointWeightPatch(
        f"model.layers.0.mlp.experts.{expert}.{projection}.weight",
        (4, 4),
        torch.bfloat16,
        torch.tensor([0.0], dtype=torch.bfloat16),
        torch.tensor([1], dtype=torch.int32),
    )
    if expert != 0 and tp_rank == 0:
        target = (4 if projection != "up_proj" else 6) + (expert - 2) * param[0].numel()
        expected.view(-1)[target] = 0
    pointer = param.data_ptr()
    remaining, handled = direct.apply_direct_moe_patches(model, [patch])
    assert not remaining and handled == {patch.name}
    assert torch.equal(param, expected) and param.data_ptr() == pointer

    # Unsupported storage must fall back even for a nonlocal expert.
    param.data = param.data.float()
    assert direct.apply_direct_moe_patches(model, [patch])[0] == [patch]


def test_repeated_names_preserve_fallback_order():
    from vllm.model_executor.model_loader.checkpoint_weight_patch import CheckpointWeightPatch

    from vllm_ascend.distributed.weight_transfer.sparse_moe_direct import apply_direct_moe_patches

    patch = CheckpointWeightPatch("same", (2,), torch.bfloat16, torch.zeros(1), torch.zeros(1, dtype=torch.int32))
    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_moe"))
    remaining, handled = apply_direct_moe_patches(model, [patch, patch])
    assert remaining == [patch, patch] and not handled


@pytest.mark.parametrize("down", [False, True])
@pytest.mark.parametrize("tp_rank", [0, 1, 2, 3])
def test_batched_moe_group_matches_nan_masked_tp_loader(down, tp_rank):
    from vllm_ascend.distributed.weight_transfer.sparse_moe_direct import _apply_moe_group

    tp, experts, hidden, intermediate = (4, 3, 8, 12)
    shape = (hidden, intermediate) if down else (intermediate, hidden)
    runtime_shape = (experts, intermediate // tp, hidden) if down else (experts, hidden, 2 * intermediate // tp)
    actual = torch.randn(runtime_shape, dtype=torch.bfloat16)
    expected = actual.clone()
    entries = []
    for expert in range(experts):
        for projection in ["down_proj"] if down else ["gate_proj", "up_proj"]:
            indices = torch.arange(expert, shape[0] * shape[1], 7, dtype=torch.int32)
            values = torch.arange(indices.numel(), dtype=torch.bfloat16) - 5
            values[::5] = float("nan")
            values[1::5] = -0.0
            patch = SparseWeightPatch(
                name=f"{expert}.{projection}", indices=indices, values=values, full_shape=tuple(shape)
            )
            entries.append((patch, expert, projection))
            dense = torch.full(shape, float("nan"), dtype=torch.bfloat16)
            dense.view(-1)[indices.long()] = values
            width = intermediate // tp
            if down:
                source = dense[:, tp_rank * width : (tp_rank + 1) * width].t()
                destination = expected[expert]
            else:
                source = dense[tp_rank * width : (tp_rank + 1) * width].t()
                offset = width if projection == "up_proj" else 0
                destination = expected[expert, :, offset : offset + width]
            destination.copy_(torch.where(torch.isnan(source), destination, source))
    _apply_moe_group(actual, entries, shape, tp_rank, tp, down)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
