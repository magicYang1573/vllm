# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import uuid
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.models.deepseek_v4_1.nvidia import engram_mooncake
from vllm.models.deepseek_v4_1.nvidia.engram_mooncake import (
    MooncakeEngramBackend,
    _consume_previous_lookup,
    _dequant_packed_engram_rows,
    _Layer,
    _LayerBuffers,
    _lookup,
    _LookupSlot,
    engram_head_shard,
)


class _Event:
    def synchronize(self):
        pass


class _Table:
    def __init__(self):
        self.calls = []

    def lookup_many_into_registered(
        self, layer_ids, row_ids, output_addresses, output_sizes
    ):
        self.calls.append(
            (
                list(layer_ids),
                [ids.copy() for ids in row_ids],
                list(output_addresses),
                list(output_sizes),
            )
        )


def _layer(layer_id, hash_index, offsets):
    tokens, heads, row_bytes = 4, len(offsets), 4
    return _Layer(
        model_id=layer_id,
        store_id=layer_id * 2,
        hash_index=hash_index,
        head_start=0,
        head_sizes=np.asarray((10,) * heads),
        offsets=np.asarray(offsets, dtype=np.int64),
        block_size=32,
        buffers=[
            _LayerBuffers(
                host_ids=torch.empty(tokens, heads, dtype=torch.int32),
                local_ids=np.empty((tokens, heads), dtype=np.int64),
                host_dead=torch.empty(tokens, heads, dtype=torch.bool),
                packed=torch.empty(tokens, heads, row_bytes, dtype=torch.uint8),
                dead=torch.empty(0),
                rows=torch.empty(0),
            )
        ],
    )


def test_engram_head_shard_uses_only_rank_owned_heads():
    sizes = tuple(range(11, 21))
    start, local, offsets = engram_head_shard(sizes, num_shards=3, shard_rank=1)
    assert start == 4
    assert local == sizes[4:8]
    assert offsets == tuple(np.cumsum((0, *sizes[:-1]))[4:8])


def test_lookup_batches_layers_and_zeros_dead_and_trailing_rows():
    first = _layer(1, 0, (0, 10))
    second = _layer(14, 1, (0, 10))
    first.buffers[0].host_ids[...] = torch.tensor(
        [[1, 12], [-1, 15], [-1, -1], [-1, -1]], dtype=torch.int32
    )
    second.buffers[0].host_ids[...] = torch.tensor(
        [[3, 19], [4, -1], [-1, -1], [-1, -1]], dtype=torch.int32
    )
    table = _Table()
    assert _lookup(table, {0: first, 1: second}, 0, 4, shard_rank=0) == 2
    assert len(table.calls) == 1
    layer_ids, row_ids, output_addresses, output_sizes = table.calls[0]
    assert layer_ids == [2, 28]
    np.testing.assert_array_equal(row_ids[0], [[[1, 2], [0, 5]]])
    np.testing.assert_array_equal(row_ids[1], [[[3, 9], [4, 0]]])
    assert output_addresses == [
        first.buffers[0].packed.data_ptr(),
        second.buffers[0].packed.data_ptr(),
    ]
    assert output_sizes == [2 * 2 * 4, 2 * 2 * 4]
    torch.testing.assert_close(
        first.buffers[0].host_dead,
        torch.tensor([[False, False], [True, False], [True, True], [True, True]]),
    )
    torch.testing.assert_close(
        second.buffers[0].host_dead,
        torch.tensor([[False, False], [False, True], [True, True], [True, True]]),
    )


def test_failed_lookup_is_consumed_once_and_allows_retry():
    failed: Future[int] = Future()
    failed.set_exception(RuntimeError("lookup failed"))
    slot = _LookupSlot(ids_ready=_Event(), rows_consumed=_Event(), future=failed)

    with pytest.raises(RuntimeError, match="lookup failed"):
        _consume_previous_lookup(slot)
    assert slot.future is None
    _consume_previous_lookup(slot)


@pytest.mark.parametrize(
    "active_rows, needs_flush, expected_flushes",
    [(2, True, 1), (0, True, 0), (2, False, 0)],
)
def test_wait_flushes_gpudirect_writes_once(
    monkeypatch, active_rows, needs_flush, expected_flushes
):
    future: Future[int] = Future()
    future.set_result(active_rows)
    slot = _LookupSlot(
        ids_ready=_Event(),
        rows_consumed=_Event(),
        future=future,
        writes_flushed=False,
    )
    backend = MooncakeEngramBackend.__new__(MooncakeEngramBackend)
    backend._slots = [slot]
    backend._needs_gpudirect_flush = needs_flush
    flushes = []
    monkeypatch.setattr(engram_mooncake, "dbo_current_ubatch_id", lambda: 0)
    monkeypatch.setattr(
        engram_mooncake, "_flush_gpudirect_writes", lambda: flushes.append(True)
    )

    backend.wait()
    backend.wait()

    assert len(flushes) == expected_flushes


def test_packed_row_dequant_matches_torch():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    tokens, actual_heads, part_heads, dim, block = 3, 2, 3, 256, 32
    row_bytes = dim + dim // block
    generator = torch.Generator().manual_seed(7)
    # Include every FP8 encoding, especially the two NaNs the byte decoder lost.
    weight = torch.arange(256).to(torch.uint8).expand(tokens, actual_heads, -1)
    scales = torch.randint(
        120,
        134,
        (tokens, actual_heads, dim // block),
        generator=generator,
        dtype=torch.uint8,
    )
    packed = torch.cat((weight, scales), dim=-1).cuda()
    dead = torch.zeros(tokens, actual_heads, dtype=torch.bool, device="cuda")
    dead[1, 0] = True
    output = torch.zeros(tokens, part_heads, dim, dtype=torch.bfloat16, device="cuda")
    num_rows = tokens * actual_heads
    _dequant_packed_engram_rows[((num_rows + 15) // 16,)](
        packed,
        dead,
        output,
        num_rows,
        ACTUAL_HEADS=actual_heads,
        PART_HEADS=part_heads,
        ROW_BYTES=row_bytes,
        DIM=dim,
        QUANT_BLOCK=block,
        BLOCK_R=16,
    )
    expected = (
        weight.view(torch.float8_e4m3fn).float().unflatten(-1, (-1, block))
        * scales.view(torch.float8_e8m0fnu).float().unsqueeze(-1)
    ).flatten(-2)
    assert not output[1, 0].count_nonzero()
    expected[1, 0].zero_()
    torch.testing.assert_close(
        output[:, :actual_heads].cpu(), expected.bfloat16(), equal_nan=True
    )
    assert not output[:, actual_heads:].count_nonzero()


@pytest.mark.parametrize("model,version", [("another-checkpoint", 2), ("model", 1)])
def test_manifest_rejects_wrong_checkpoint_or_legacy_publication(
    tmp_path, model, version
):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(dict(version=version, model=model, publication="id")))
    with pytest.raises(ValueError, match="checkpoint|Republish"):
        MooncakeEngramBackend(str(path), 1, 0, 1, "model")


@pytest.mark.skipif(
    os.environ.get("VLLM_TEST_MOONCAKE") != "1",
    reason="requires a dedicated Mooncake Store and VLLM_USE_BREAKABLE_CUDAGRAPH=1",
)
def test_store_graph_replay_uses_live_hashes_and_reuses_both_slots(
    tmp_path, monkeypatch
):
    """A captured hash producer must run before host lookup, including slot reuse."""
    from mooncake.store import EngramStore, EngramStoreConfig, ReplicateConfig

    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    owner = engram_mooncake.create_store()
    config = EngramStoreConfig()
    config.table_vocab_sizes, config.row_bytes = [7, 11], 264
    table = EngramStore({1: config}, store_client=owner)
    sources = [
        torch.cat(
            (
                torch.arange(1, size + 1)
                .float()[:, None]
                .expand(-1, 256)
                .to(torch.float8_e4m3fn)
                .view(torch.uint8),
                torch.full((size, 8), 127, dtype=torch.uint8),
            ),
            -1,
        ).numpy()
        for size in (7, 11)
    ]
    replicate = ReplicateConfig()
    replicate.with_hard_pin = True
    backend = None
    try:
        table.populate(1, sources, replicate)
        manifest = dict(
            version=2,
            publication=uuid.uuid4().hex,
            model=str(tmp_path.resolve()),
            num_shards=1,
            layers={"1": dict(table_vocab_sizes=[7, 11], head_dim=256, row_bytes=264)},
        )
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest))
        assert owner.put("engram:manifest", path.read_bytes(), replicate) == 0
        backend = MooncakeEngramBackend(str(path), 1, 0, 2, str(tmp_path))
        embedding = SimpleNamespace(
            dim=256, block_size=32, head_start=0, part_n_hash_cols=3
        )
        backend.attach(embedding, 1, 0, (7, 11), 4)
        hashes = torch.zeros(4, 1, 2, dtype=torch.int32, device="cuda")
        hashes[:, :, 1] = 7
        slot = [0]
        monkeypatch.setattr(engram_mooncake, "dbo_current_ubatch_id", lambda: slot[0])
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for slot[0] in (0, 1):
                backend.prefetch(hashes)
                backend.rows(0)
                capture = BreakableCUDAGraphCapture()
                with capture:
                    backend.prefetch(hashes + 0)
                    output = backend.rows(0)
                for ids in (
                    [[1, 9], [3, -1], [-1, -1], [-1, -1]],
                    [[6, 17], [0, 7], [4, 10], [2, 11]],
                    [[-1, -1]] * 4,
                ):
                    hashes.copy_(torch.tensor(ids, dtype=torch.int32)[:, None])
                    capture.replay()
                    stream.synchronize()
                    expected = torch.zeros(4, 3, 256, dtype=torch.bfloat16)
                    for token, row in enumerate(ids):
                        for head, row_id in enumerate(row):
                            if row_id != -1:
                                expected[token, head] = row_id - (7 if head else 0) + 1
                    torch.testing.assert_close(output.cpu(), expected)
                assert capture.num_graphs and capture.num_eager_breaks
            # A second model owns its own backend; stale publications fail closed.
            manifest["publication"] = uuid.uuid4().hex
            path.write_text(json.dumps(manifest))
            other = MooncakeEngramBackend(str(path), 1, 0, 1, str(tmp_path))
            try:
                other.attach(embedding, 1, 0, (7, 11), 4)
                with pytest.raises(ValueError, match="publication"):
                    other.prefetch(hashes)
            finally:
                other.close()
    finally:
        if backend is not None:
            backend.close()
        owner.close()
