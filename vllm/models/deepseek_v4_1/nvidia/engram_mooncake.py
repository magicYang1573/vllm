# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batched Mooncake RDMA lookup for DeepSeek V4.1 Engram rows."""

from __future__ import annotations

import json
import threading
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart

from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphCapture,
    eager_break_during_capture,
)
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

if TYPE_CHECKING:
    from .engram import ParallelEngramEmbedding

logger = init_logger(__name__)


def _gpudirect_flush_required() -> bool:
    error, device = cudart.cudaGetDevice()
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"Could not query the current CUDA device: {error}")
    error, ordering = cudart.cudaDeviceGetAttribute(
        cudart.cudaDeviceAttr.cudaDevAttrGPUDirectRDMAWritesOrdering, device
    )
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"Could not query GPUDirect RDMA ordering: {error}")
    owner_scope = (
        cudart.cudaFlushGPUDirectRDMAWritesScope.cudaFlushGPUDirectRDMAWritesToOwner
    )
    if int(ordering) >= int(owner_scope):
        return False
    error, options = cudart.cudaDeviceGetAttribute(
        cudart.cudaDeviceAttr.cudaDevAttrGPUDirectRDMAFlushWritesOptions, device
    )
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"Could not query GPUDirect RDMA flush support: {error}")
    flush_options = cudart.cudaFlushGPUDirectRDMAWritesOptions
    host_flush = flush_options.cudaFlushGPUDirectRDMAWritesOptionHost
    if not (int(options) & int(host_flush)):
        raise RuntimeError(
            "This CUDA device cannot make GPUDirect RDMA writes visible to kernels"
        )
    return True


def _flush_gpudirect_writes() -> None:
    result = cudart.cudaDeviceFlushGPUDirectRDMAWrites(
        cudart.cudaFlushGPUDirectRDMAWritesTarget.cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
        cudart.cudaFlushGPUDirectRDMAWritesScope.cudaFlushGPUDirectRDMAWritesToOwner,
    )[0]
    if result != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA GPUDirect RDMA write flush failed: {result}")


def engram_head_shard(
    head_sizes: tuple[int, ...], num_shards: int, shard_rank: int
) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    """Return the first head, local sizes, and global offsets for one shard."""
    if not head_sizes or num_shards <= 0 or not 0 <= shard_rank < num_shards:
        raise ValueError("invalid Engram head shard geometry")
    part_heads = (len(head_sizes) + num_shards - 1) // num_shards
    head_start = shard_rank * part_heads
    if head_start >= len(head_sizes):
        raise ValueError(f"Engram sharding leaves rank {shard_rank} without hash heads")
    head_end = min(head_start + part_heads, len(head_sizes))
    offsets = np.cumsum((0, *head_sizes[:-1]), dtype=np.int64)
    return (
        head_start,
        head_sizes[head_start:head_end],
        tuple(int(value) for value in offsets[head_start:head_end]),
    )


def create_store(*, reader: bool = False):
    from mooncake.mooncake_config import MooncakeConfig
    from mooncake.store import MooncakeDistributedStore

    config = MooncakeConfig.load_from_env()
    store = MooncakeDistributedStore()
    rc = store.setup(
        local_hostname=config.local_hostname,
        metadata_server=config.metadata_server,
        global_segment_size=0 if reader else config.global_segment_size,
        local_buffer_size=config.local_buffer_size,
        protocol=config.protocol,
        rdma_devices=config.device_name or "",
        master_server_addr=config.master_server_address,
        enable_ssd_offload=config.enable_ssd_offload,
        ssd_offload_path=config.ssd_offload_path,
        tenant_id=config.tenant_id,
        enable_client_http_server=config.enable_client_http_server,
        client_http_port=config.client_http_port,
    )
    if rc != 0:
        store.close()
        raise RuntimeError(f"Mooncake Store setup failed, rc={rc}")
    return store


@triton.jit(do_not_specialize=["num_rows"])
def _dequant_packed_engram_rows(
    packed,
    dead,
    output,
    num_rows,
    ACTUAL_HEADS: tl.constexpr,
    PART_HEADS: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    cols = tl.arange(0, DIM)
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    valid = rows < num_rows
    row = rows.to(tl.int64)
    alive = tl.load(dead + row, mask=valid, other=1) == 0
    values = (
        tl.load(
            packed + row[:, None] * ROW_BYTES + cols[None, :],
            mask=valid[:, None] & alive[:, None],
            other=0,
        )
        .to(tl.float8e4nv, bitcast=True)
        .to(tl.float32)
    )
    scale = tl.load(
        packed + row[:, None] * ROW_BYTES + DIM + cols[None, :] // QUANT_BLOCK,
        mask=valid[:, None] & alive[:, None],
        other=0,
    )
    scale = (scale.to(tl.int32) << 23).to(tl.float32, bitcast=True)
    out_row = row // ACTUAL_HEADS * PART_HEADS + row % ACTUAL_HEADS
    tl.store(
        output + out_row[:, None] * DIM + cols[None, :],
        (values * scale).to(tl.bfloat16),
        mask=valid[:, None],
    )


@dataclass
class _LayerBuffers:
    host_ids: torch.Tensor
    local_ids: np.ndarray
    host_dead: torch.Tensor
    packed: torch.Tensor
    dead: torch.Tensor
    rows: torch.Tensor


@dataclass
class _Layer:
    model_id: int
    store_id: int
    hash_index: int
    head_start: int
    head_sizes: np.ndarray
    offsets: np.ndarray
    block_size: int
    buffers: list[_LayerBuffers]


@dataclass
class _LookupSlot:
    ids_ready: torch.cuda.Event = field(default_factory=torch.cuda.Event)
    rows_consumed: torch.cuda.Event = field(default_factory=torch.cuda.Event)
    future: Future[int] | None = None
    num_tokens: int = 0
    writes_flushed: bool = True


def _consume_previous_lookup(slot: _LookupSlot) -> None:
    future, slot.future = slot.future, None
    if future is not None:
        future.result()


def _release_backend(resources: dict[str, Any], executor: ThreadPoolExecutor) -> None:
    executor.shutdown(wait=True)
    store = resources.get("store")
    if store is not None:
        for buffer in resources["registered"]:
            rc = store.unregister_buffer(buffer.data_ptr())
            if rc != 0:
                logger.warning("Mooncake Engram buffer unregister failed: %s", rc)
        store.close()


def _lookup(
    table: Any,
    layers: dict[int, _Layer],
    slot_index: int,
    num_tokens: int,
    shard_rank: int,
) -> int:
    active_end = 0
    for layer in layers.values():
        buffer = layer.buffers[slot_index]
        ids = buffer.host_ids[:num_tokens].numpy()
        np.equal(ids, -1, out=buffer.host_dead[:num_tokens].numpy())
        active = np.flatnonzero(np.any(ids != -1, axis=1))
        if active.size:
            active_end = max(active_end, int(active[-1]) + 1)
    if not active_end:
        return 0
    layer_ids, local_ids, addresses, sizes = [], [], [], []
    for layer in layers.values():
        buffer = layer.buffers[slot_index]
        ids = buffer.host_ids[:active_end].numpy()
        local = buffer.local_ids[:active_end]
        np.subtract(ids, layer.offsets, out=local)
        dead = buffer.host_dead[:active_end].numpy()
        valid = dead | ((local >= 0) & (local < layer.head_sizes))
        if not valid.all():
            row, head = np.argwhere(~valid)[0]
            raise ValueError(
                f"Engram hash outside layer {layer.model_id}'s owned heads: "
                f"shard={shard_rank}, token={row}, head={head}, "
                f"id={ids[row, head]}"
            )
        local[dead] = 0
        layer_ids.append(layer.store_id)
        local_ids.append(local[None])
        addresses.append(buffer.packed.data_ptr())
        sizes.append(active_end * buffer.packed.stride(0))
    table.lookup_many_into_registered(layer_ids, local_ids, addresses, sizes)
    return active_end


class MooncakeEngramBackend:
    """Model-owned batched Store reads into reusable CUDA buffers."""

    def __init__(
        self,
        config_path: str,
        num_shards: int,
        shard_rank: int,
        num_slots: int,
        model: str,
    ) -> None:
        self.manifest = json.loads(Path(config_path).expanduser().read_text())
        if self.manifest.get("version") != 2 or not self.manifest.get("publication"):
            raise ValueError("Republish Engram tables with the version 2 publisher")
        if self.manifest.get("model") != str(Path(model).expanduser().resolve()):
            raise ValueError("Mooncake Engram manifest belongs to another checkpoint")
        if self.manifest.get("num_shards") != num_shards:
            raise ValueError("Mooncake Engram num_shards does not match TP/Engram-DP")
        self.num_shards, self.shard_rank = num_shards, shard_rank
        self._layers: dict[int, _Layer] = {}
        self._slots = [_LookupSlot() for _ in range(num_slots)]
        self._connect_lock = threading.Lock()
        self._store: Any = None
        self._table: Any = None
        self._needs_gpudirect_flush = False
        self._executor = ThreadPoolExecutor(
            max_workers=num_slots, thread_name_prefix="vllm-engram-mooncake"
        )
        self._resources: dict[str, Any] = {"store": None, "registered": []}
        self._finalizer = weakref.finalize(
            self, _release_backend, self._resources, self._executor
        )

    def close(self) -> None:
        self._finalizer()

    def attach(
        self,
        embedding: ParallelEngramEmbedding,
        model_layer_id: int,
        hash_index: int,
        head_sizes: tuple[int, ...],
        max_tokens: int,
    ) -> None:
        if self._store is not None or hash_index in self._layers:
            raise ValueError("Engram layers must be attached once before connecting")
        dim, block = embedding.dim, embedding.block_size
        row_bytes = dim + dim // block
        expected = dict(
            table_vocab_sizes=list(head_sizes), head_dim=dim, row_bytes=row_bytes
        )
        if self.manifest["layers"].get(str(model_layer_id)) != expected:
            raise ValueError(
                f"Mooncake Engram layout mismatch for layer {model_layer_id}"
            )
        start, sizes, offsets = engram_head_shard(
            head_sizes, self.num_shards, self.shard_rank
        )
        if start != embedding.head_start:
            raise ValueError("Mooncake and vLLM Engram head sharding disagree")
        device = torch.device("cuda", torch.accelerator.current_device_index())
        shape = (max_tokens, len(sizes))
        buffers = [
            _LayerBuffers(
                host_ids=torch.empty(
                    shape, dtype=torch.int32, device="cpu", pin_memory=True
                ),
                local_ids=np.empty(shape, dtype=np.int64),
                host_dead=torch.empty(
                    shape, dtype=torch.bool, device="cpu", pin_memory=True
                ),
                packed=torch.empty(
                    (*shape, row_bytes), dtype=torch.uint8, device=device
                ),
                dead=torch.empty(shape, dtype=torch.bool, device=device),
                rows=torch.zeros(
                    (max_tokens, embedding.part_n_hash_cols, dim),
                    dtype=torch.bfloat16,
                    device=device,
                ),
            )
            for _ in self._slots
        ]
        self._layers[hash_index] = _Layer(
            model_layer_id,
            model_layer_id * self.num_shards + self.shard_rank,
            hash_index,
            start,
            np.asarray(sizes, dtype=np.int64),
            np.asarray(offsets, dtype=np.int64),
            block,
            buffers,
        )

    def _connect(self) -> None:
        with self._connect_lock:
            if not self._finalizer.alive:
                raise RuntimeError("Mooncake Engram backend is closed")
            if self._store is not None:
                return
            from mooncake.store import EngramStore, EngramStoreConfig

            if not self._layers:
                raise RuntimeError("Mooncake Engram has no attached layers")
            if not hasattr(EngramStore, "lookup_many_into_registered"):
                raise RuntimeError(
                    "Mooncake Engram requires registered CUDA output support"
                )
            store = create_store(reader=True)
            registered = self._resources["registered"]
            try:
                ready = store.get("engram:manifest")
                if not ready or json.loads(ready) != self.manifest:
                    raise ValueError(
                        "Mooncake Engram publication does not match the manifest"
                    )
                self._needs_gpudirect_flush = _gpudirect_flush_required()
                configs = {}
                for layer in self._layers.values():
                    config = EngramStoreConfig()
                    config.table_vocab_sizes = layer.head_sizes.tolist()
                    config.row_bytes = layer.buffers[0].packed.shape[-1]
                    configs[layer.store_id] = config
                table = EngramStore(configs, store_client=store)
                keys = [
                    key
                    for layer_id in configs
                    for key in table.get_store_keys(layer_id)
                ]
                exists = store.batch_is_exist(keys)
                if len(exists) != len(keys) or any(value != 1 for value in exists):
                    raise RuntimeError("Mooncake Engram tables are incomplete")
                for layer in self._layers.values():
                    for buffer in layer.buffers:
                        packed = buffer.packed
                        if (
                            store.register_buffer(packed.data_ptr(), packed.numel())
                            != 0
                        ):
                            raise RuntimeError(
                                "Could not register Mooncake Engram CUDA buffer"
                            )
                        registered.append(packed)
            except Exception:
                for packed in registered:
                    store.unregister_buffer(packed.data_ptr())
                registered.clear()
                store.close()
                raise
            self._store, self._table = store, table
            self._resources["store"] = store

    @eager_break_during_capture
    def prefetch(self, gathered_hashes: torch.Tensor) -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Mooncake Engram requires breakable CUDA graphs or eager mode"
            )
        self._connect()
        slot_index = dbo_current_ubatch_id()
        slot = self._slots[slot_index]
        _consume_previous_lookup(slot)
        # The background thread must not synchronize CUDA while capture resumes.
        slot.rows_consumed.synchronize()
        slot.num_tokens = gathered_hashes.shape[0]
        capturing = BreakableCUDAGraphCapture.is_active()
        for layer in self._layers.values():
            buffer = layer.buffers[slot_index]
            if slot.num_tokens > buffer.host_ids.shape[0]:
                raise RuntimeError("Mooncake Engram staging buffer is too small")
            if capturing:
                buffer.host_dead[: slot.num_tokens].fill_(True)
            else:
                buffer.host_ids[: slot.num_tokens].copy_(
                    gathered_hashes[
                        :,
                        layer.hash_index,
                        layer.head_start : layer.head_start + len(layer.head_sizes),
                    ],
                    non_blocking=True,
                )
        slot.writes_flushed = capturing
        if capturing:
            # Hash producers are recorded, not executed, on the first capture.
            slot.future = Future()
            slot.future.set_result(0)
        else:
            slot.ids_ready.record(torch.cuda.current_stream())
            slot.ids_ready.synchronize()
            slot.future = self._executor.submit(
                _lookup,
                self._table,
                self._layers,
                slot_index,
                slot.num_tokens,
                self.shard_rank,
            )

    @eager_break_during_capture
    def wait(self) -> None:
        slot = self._slots[dbo_current_ubatch_id()]
        if slot.future is None:
            raise RuntimeError("Mooncake Engram rows were not prefetched")
        if slot.future.result() and not slot.writes_flushed:
            if self._needs_gpudirect_flush:
                _flush_gpudirect_writes()
            slot.writes_flushed = True

    def rows(self, hash_index: int) -> torch.Tensor:
        self.wait()
        slot_index = dbo_current_ubatch_id()
        layer = self._layers[hash_index]
        buffer = layer.buffers[slot_index]
        tokens = self._slots[slot_index].num_tokens
        buffer.dead[:tokens].copy_(buffer.host_dead[:tokens], non_blocking=True)
        rows = tokens * len(layer.head_sizes)
        if rows:
            _dequant_packed_engram_rows[(triton.cdiv(rows, 16),)](
                buffer.packed,
                buffer.dead,
                buffer.rows,
                rows,
                ACTUAL_HEADS=len(layer.head_sizes),
                PART_HEADS=buffer.rows.shape[1],
                ROW_BYTES=buffer.packed.shape[-1],
                DIM=buffer.rows.shape[-1],
                QUANT_BLOCK=layer.block_size,
                BLOCK_R=16,
            )
        self._record_rows_consumed()
        return buffer.rows[:tokens]

    @eager_break_during_capture
    def _record_rows_consumed(self) -> None:
        self._slots[dbo_current_ubatch_id()].rows_consumed.record(
            torch.cuda.current_stream()
        )
