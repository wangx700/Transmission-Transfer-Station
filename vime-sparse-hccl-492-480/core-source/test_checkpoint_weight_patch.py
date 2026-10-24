# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager

import pytest
import torch
from vllm.model_executor.model_loader.checkpoint_weight_patch import (
    CheckpointWeightPatch,
    load_checkpoint_weight_patches,
)

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("numel", [2**31 - 1, 2**31, 2**31 + 1])
def test_int32_indices_validate_against_large_checkpoint_shape(numel):
    # The extension consumes the patch after real validation, avoiding a giant
    # staging allocation while exercising the actual bounds comparison.
    patch = CheckpointWeightPatch(
        "large.weight",
        (numel,),
        torch.float32,
        torch.tensor([2.0, 3.0]),
        torch.tensor([0, numel - 1 if numel < 2**31 else 2**31 - 1], dtype=torch.int32),
    )

    def apply(patches):
        assert patches == [patch]
        return [], {"large.weight"}

    assert load_checkpoint_weight_patches(object(), [patch], sparse_apply=apply) == {
        "large.weight"
    }


@pytest.mark.parametrize("index", [-1, 2**31 - 1])
def test_invalid_int32_large_checkpoint_index_rejected_before_apply(index):
    patch = CheckpointWeightPatch(
        "large.weight",
        (2**31 - 1,),
        torch.float32,
        torch.tensor([1.0]),
        torch.tensor([index], dtype=torch.int32),
    )

    def never_apply(_patches):
        pytest.fail("invalid indices must be rejected before applying")

    with pytest.raises(ValueError, match="out-of-range"):
        load_checkpoint_weight_patches(object(), [patch], sparse_apply=never_apply)


def test_phase_observer_preserves_dense_and_sparse_results():
    phases = []

    @contextmanager
    def observe(name):
        phases.append(name)
        yield

    model = _PackedTPModel()
    load_checkpoint_weight_patches(
        model,
        [
            CheckpointWeightPatch(
                "q_proj.weight", (4,), torch.float32, torch.tensor([1.0, 2.0, 3.0, 4.0])
            )
        ],
        phase_observer=observe,
    )
    assert torch.equal(model.packed_weight[:2], torch.tensor([3.0, 4.0]))
    load_checkpoint_weight_patches(
        model,
        [
            CheckpointWeightPatch(
                "q_proj.weight",
                (4,),
                torch.float32,
                torch.tensor([8.0]),
                torch.tensor([3], dtype=torch.int32),
            )
        ],
        phase_observer=observe,
    )
    assert torch.equal(model.packed_weight[:2], torch.tensor([3.0, 8.0]))
    assert phases == ["receiver_validate", "receiver_staging", "receiver_load"] * 2


class _PackedTPModel(torch.nn.Module):
    """Model that packs the TP rank 1 shards of Q and K into one parameter."""

    def __init__(self):
        super().__init__()
        self.packed_weight = torch.nn.Parameter(torch.full((4,), -1.0))
        self.load_calls: list[list[str]] = []

    def load_weights(self, weights):
        weights = list(weights)
        self.load_calls.append([name for name, _ in weights])
        loaded_names = set()
        for name, loaded_weight in weights:
            if name == "q_proj.weight":
                destination = self.packed_weight.data[:2]
            elif name == "k_proj.weight":
                destination = self.packed_weight.data[2:]
            else:
                raise AssertionError(name)
            # Slice each full checkpoint tensor and load its second half for TP rank 1.
            destination.copy_(loaded_weight.narrow(0, 2, 2))
            loaded_names.add(name)
        return loaded_names


def _make_patch(
    name: str,
    *,
    values: list[float],
    indices: list[int] | None = None,
) -> CheckpointWeightPatch:
    return CheckpointWeightPatch(
        name=name,
        shape=(4,),
        dtype=torch.float32,
        values=torch.tensor(values),
        indices=None if indices is None else torch.tensor(indices, dtype=torch.int32),
    )


def test_dense_and_sparse_patches_follow_packed_tp_loader():
    model = _PackedTPModel()

    # Seed the TP-local packed weights through the dense path.
    dense_loaded = load_checkpoint_weight_patches(
        model,
        [
            _make_patch("q_proj.weight", values=[0.0, 1.0, 2.0, 3.0]),
            _make_patch("k_proj.weight", values=[10.0, 11.0, 12.0, 13.0]),
        ],
    )
    assert dense_loaded == {"q_proj.weight", "k_proj.weight"}
    assert torch.equal(model.packed_weight, torch.tensor([2.0, 3.0, 12.0, 13.0]))

    # Sparse indices address the full checkpoint tensor; index 0 is outside this shard.
    model.load_calls.clear()
    original_copy = torch.Tensor.copy_
    sparse_loaded = load_checkpoint_weight_patches(
        model,
        [
            _make_patch(
                "q_proj.weight",
                indices=[0, 3],
                values=[100.0, 30.0],
            ),
            _make_patch("k_proj.weight", indices=[2], values=[20.0]),
            _make_patch("q_proj.weight", indices=[2], values=[22.0]),
        ],
    )

    assert sparse_loaded == {"q_proj.weight", "k_proj.weight"}
    assert model.load_calls == [
        ["q_proj.weight", "k_proj.weight"],
        ["q_proj.weight"],
    ]
    assert torch.equal(model.packed_weight, torch.tensor([22.0, 30.0, 20.0, 13.0]))
    assert torch.Tensor.copy_ is original_copy

    # The empty result tells the helper that the loader intentionally made no write.
    class NoLocalWeightModel(torch.nn.Module):
        """Model that deliberately owns none of the supplied weights."""

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))

        def load_weights(self, _weights):
            return set()

    no_local_weight_model = NoLocalWeightModel()
    no_local_weight_loaded = load_checkpoint_weight_patches(
        no_local_weight_model,
        [_make_patch("q_proj.weight", indices=[3], values=[30.0])],
    )
    assert no_local_weight_loaded == set()
    assert torch.equal(no_local_weight_model.weight, torch.tensor([1.0]))


def test_sparse_extension_is_validated_before_writes_and_outside_loader_context():
    from contextlib import contextmanager

    model = _PackedTPModel()
    events = []

    @contextmanager
    def loader_context():
        events.append("enter")
        yield
        events.append("exit")

    def apply(patches):
        assert not events
        events.append("direct")
        model.packed_weight.data[0] = 99
        return patches[1:], {patches[0].name}

    patches = [
        _make_patch("q_proj.weight", indices=[2], values=[99.0]),
        _make_patch("k_proj.weight", indices=[2], values=[20.0]),
    ]
    loaded = load_checkpoint_weight_patches(
        model, patches, sparse_apply=apply, load_context=loader_context
    )
    assert loaded == {"q_proj.weight", "k_proj.weight"}
    assert model.packed_weight[0] == 99 and model.packed_weight[2] == 20
    assert events == ["direct", "enter", "exit"]

    events.clear()
    patches.append(_make_patch("bad", indices=[4], values=[1.0]))
    with pytest.raises(ValueError):
        load_checkpoint_weight_patches(
            model, patches, sparse_apply=apply, load_context=loader_context
        )
    assert events == []


def test_dense_updates_bypass_sparse_extension_and_restore_loader_context_on_error():
    from contextlib import contextmanager

    events = []

    @contextmanager
    def loader_context():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    class FailedLoader:
        def load_weights(self, weights):
            raise RuntimeError("loader failed")

    def never_apply(_patches):
        pytest.fail("dense bootstrap must not invoke sparse extension")

    with pytest.raises(RuntimeError, match="loader failed"):
        load_checkpoint_weight_patches(
            FailedLoader(),
            [_make_patch("weight", values=[1.0, 2.0, 3.0, 4.0])],
            sparse_apply=never_apply,
            load_context=loader_context,
        )
    assert events == ["enter", "exit"]


def test_sparse_extension_rejects_reordered_fallback():
    patches = [
        _make_patch("q_proj.weight", indices=[2], values=[1.0]),
        _make_patch("k_proj.weight", indices=[2], values=[2.0]),
    ]
    with pytest.raises(ValueError, match="ordered subset"):
        load_checkpoint_weight_patches(
            _PackedTPModel(),
            patches,
            sparse_apply=lambda rows: (list(reversed(rows)), set()),
        )
