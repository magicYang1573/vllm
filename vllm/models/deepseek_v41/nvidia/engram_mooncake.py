# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Global Engram table publication and per-layer Mooncake RDMA prefetch."""

from __future__ import annotations

import json
import threading
import time
import uuid
import weakref
from contextlib import ExitStack
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

import vllm.envs as envs
from vllm.distributed import get_tensor_model_parallel_rank, get_world_group
from vllm.logger import init_logger
from vllm.models.deepseek_v41.common.engram import Engram
from vllm.models.deepseek_v41.common.engram import (
    ParallelEngramEmbedding as BaseParallelEngramEmbedding,
)
from vllm.triton_utils import tl, triton
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.models.deepseek_v41.common.engram import EngramLayout

    from .model import DeepseekV4Model

logger = init_logger(__name__)


def _head_sizes(layout: EngramLayout, layer_index: int) -> tuple[int, ...]:
    return tuple(size for order in layout.primes[layer_index] for size in order)


def _layer_manifest(head_sizes, dim: int, block_size: int = 32) -> dict:
    return dict(
        table_vocab_sizes=list(head_sizes),
        head_dim=dim,
        row_bytes=dim + dim // block_size,
    )


def _table_ids(model_layer_id: int, num_heads: int) -> tuple[int, ...]:
    return tuple(model_layer_id * num_heads + head for head in range(num_heads))


def _store_configs(table_ids, head_sizes, row_bytes: int) -> dict:
    from mooncake.store import EngramStoreConfig

    configs = {}
    for table_id, size in zip(table_ids, head_sizes):
        config = EngramStoreConfig()
        config.table_vocab_sizes = [int(size)]
        config.row_bytes = row_bytes
        configs[table_id] = config
    return configs


class MooncakeEngramEmbedding(BaseParallelEngramEmbedding):
    """Keep TP head geometry and checkpoint names without allocating a table."""

    def __init__(self, layout: EngramLayout, layer_hash_index: int) -> None:
        super().__init__(
            layout.num_embeddings[layer_hash_index],
            layout.head_dim,
            _head_sizes(layout, layer_hash_index),
        )

    def _get_shard_info(self) -> tuple[int, int]:
        # Remote tables are global; DP ranks read only their own request IDs.
        self.dp_size = 1
        return self.tp_size, get_tensor_model_parallel_rank()

    def _allocate_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        # The native weight loader loads zero rows into these placeholders.
        return (
            torch.empty(0, self.dim, dtype=torch.float8_e4m3fn),
            torch.empty(0, self.dim // self.block_size, dtype=torch.uint8),
        )

    def _storage(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("Mooncake Engram rows require model-level prefetch")


class EngramMooncake(Engram):
    """Shared Engram computation with embeddings fetched from Mooncake Store."""

    mooncake_backend: MooncakeEngramBackend

    def _create_embedding(
        self, layout: EngramLayout, layer_hash_index: int
    ) -> MooncakeEngramEmbedding:
        return MooncakeEngramEmbedding(layout, layer_hash_index)

    def _init_staging(self, max_tokens: int, head_dim: int) -> None:
        # The model attaches Store buffers after constructing its layers.
        pass

    def prepare_embeddings(self, hash_ids: torch.Tensor) -> None:
        self.mooncake_backend.prefetch(hash_ids)

    def _ready_rows(self, num_tokens: int) -> torch.Tensor:
        return self.mooncake_backend.rows()[:num_tokens]


def _release_engram_store(backends: list[MooncakeEngramBackend], store: Any) -> None:
    with ExitStack() as cleanup:
        cleanup.callback(store.close)
        for backend in backends:
            cleanup.callback(backend.close)


def init_mooncake_engram(model: DeepseekV4Model, config: VllmConfig) -> None:
    """Initialize global tables and attach readers through the shared-table path."""
    if model.engram_layout is None:
        return
    publisher = (
        envs.VLLM_ENGRAM_MOONCAKE_PUBLISH
        and config.parallel_config.data_parallel_rank == 0
        and get_world_group().rank == 0
    )
    # Check for an existing publication before allocating any table memory.
    store = create_store(reader=True)
    backends = []
    try:
        ready = store.get("engram:manifest")
        if not ready and publisher:
            owner = create_store()
            store.close()
            store = owner
            manifest = publish_engram_tables(
                config.model_config.model, model.engram_layout, store
            )
        else:
            timeout = config.parallel_config.cpu_distributed_timeout_seconds or 1800
            deadline = time.monotonic() + timeout
            if not ready:
                logger.info("Waiting for the Mooncake Engram publisher")
            while not ready:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Timed out waiting for Mooncake Engram tables")
                time.sleep(1)
                ready = store.get("engram:manifest")
            manifest = json.loads(ready)
        _validate_manifest(manifest, config.model_config.model)
        expected = {
            str(layer_id): _layer_manifest(
                _head_sizes(model.engram_layout, index), model.engram_layout.head_dim
            )
            for index, layer_id in enumerate(model.engram_layout.layer_ids)
        }
        if manifest["layers"] != expected:
            raise ValueError("Mooncake Engram table layout does not match the model")
        from mooncake.store import EngramStore

        configs = _manifest_configs(manifest)
        table = EngramStore(configs, store_client=store)
        keys = [key for table_id in configs for key in table.get_store_keys(table_id)]
        exists = store.batch_is_exist(keys)
        if len(exists) != len(keys) or any(value != 1 for value in exists):
            raise RuntimeError("Mooncake Engram tables are incomplete")
        for layer in islice(model.layers, model.start_layer, model.end_layer):
            engram = getattr(layer, "engram", None)
            if engram is not None:
                backend = MooncakeEngramBackend(
                    manifest,
                    2 if config.parallel_config.enable_dbo else 1,
                    config.model_config.model,
                )
                backends.append(backend)
                backend.attach(
                    engram,
                    model.engram_layout,
                    config.scheduler_config.max_num_batched_tokens,
                )
    except BaseException:
        _release_engram_store(backends, store)
        raise
    # Drain outstanding reads before releasing the publisher's table segment.
    weakref.finalize(model, _release_engram_store, backends, store)
    model.engram_dp_shared_memory = True


def create_store(*, reader: bool = False):
    from mooncake.mooncake_config import MooncakeConfig
    from mooncake.store import MooncakeDistributedStore

    config = MooncakeConfig.load_from_env()
    if not reader and config.global_segment_size <= 0:
        raise ValueError("Engram publication requires a positive Store segment size")
    store = MooncakeDistributedStore()
    rc = store.setup(
        local_hostname=config.local_hostname,
        metadata_server=config.metadata_server,
        global_segment_size=0 if reader else config.global_segment_size,
        local_buffer_size=config.local_buffer_size,
        protocol=config.protocol,
        rdma_devices=config.device_name or "",
        master_server_addr=config.master_server_address,
        tenant_id=config.tenant_id,
    )
    if rc != 0:
        store.close()
        raise RuntimeError(f"Mooncake Store setup failed, rc={rc}")
    return store


def _weight_map(model: Path) -> dict[str, str]:
    from safetensors import safe_open

    index_path = model / "model.safetensors.index.json"
    if index_path.exists():
        return json.loads(index_path.read_text())["weight_map"]
    result = {}
    for shard in sorted(model.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as checkpoint:
            result.update(dict.fromkeys(checkpoint.keys(), shard.name))
    if not result:
        raise ValueError(f"No safetensors checkpoints found under {model}")
    return result


def _find_weight(weight_map: dict[str, str], suffix: str) -> tuple[str, str]:
    matches = [
        (name, file) for name, file in weight_map.items() if name.endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one checkpoint weight ending in {suffix!r}")
    return matches[0]


def _pack_head(weight, scale, offset: int, rows: int, metadata: dict) -> np.ndarray:
    dim = metadata["head_dim"]
    packed = np.empty((rows, metadata["row_bytes"]), dtype=np.uint8)
    for start in range(0, rows, 65536):
        end = min(start + 65536, rows)
        packed[start:end, :dim] = (
            weight[offset + start : offset + end].view(torch.uint8).numpy()
        )
        packed[start:end, dim:] = (
            scale[offset + start : offset + end].view(torch.uint8).numpy()
        )
    return packed


def _verify_published_head(store, table, table_id: int, source: np.ndarray) -> None:
    """Read the boundary rows back before releasing the source buffer."""
    row_ids = np.array([[[0], [len(source) - 1]]], dtype=np.int64)
    output = np.empty((1, 2, 1, source.shape[1]), dtype=np.uint8)
    rc = store.register_buffer(output.ctypes.data, output.nbytes)
    if rc != 0:
        raise RuntimeError(f"Could not register verification buffer, rc={rc}")
    try:
        table.lookup_into(table_id, row_ids, output)
        if not np.array_equal(output[0, :, 0], source[[0, -1]]):
            raise RuntimeError(f"Mooncake verification failed for table {table_id}")
    finally:
        rc = store.unregister_buffer(output.ctypes.data)
        if rc != 0:
            raise RuntimeError(f"Could not unregister verification buffer, rc={rc}")


def _manifest_configs(manifest: dict) -> dict:
    configs = {}
    for layer_id, metadata in manifest["layers"].items():
        sizes = metadata["table_vocab_sizes"]
        configs.update(
            _store_configs(
                _table_ids(int(layer_id), len(sizes)), sizes, metadata["row_bytes"]
            )
        )
    return configs


def publish_engram_tables(model: str | Path, layout: EngramLayout, store: Any) -> dict:
    """Upload checkpoint heads and return their ready manifest.

    The caller owns the Store connection and must keep its table segment alive.
    Use an immutable checkpoint and exactly one publisher per Store tenant.
    """
    from mooncake.store import EngramStore, ReplicateConfig
    from safetensors import safe_open

    model = Path(model).expanduser().resolve()
    weight_map = _weight_map(model)
    manifest: dict[str, Any] = dict(
        version=3, model=str(model), publication=uuid.uuid4().hex, layers={}
    )
    for index, layer_id in enumerate(layout.layer_ids):
        metadata = _layer_manifest(_head_sizes(layout, index), layout.head_dim)
        manifest["layers"][str(layer_id)] = metadata

    configs = _manifest_configs(manifest)
    replicate = ReplicateConfig()
    replicate.with_hard_pin = True
    replicate.preferred_segment = store.get_hostname()
    table = EngramStore(configs, store_client=store)
    keys = [key for table_id in configs for key in table.get_store_keys(table_id)]
    exists = store.batch_is_exist(keys)
    if len(exists) != len(keys) or any(value != 0 for value in exists):
        raise RuntimeError("Mooncake destination already contains Engram table keys")
    for index, layer_id in enumerate(layout.layer_ids):
        metadata = manifest["layers"][str(layer_id)]
        sizes = metadata["table_vocab_sizes"]
        if sum(sizes) > layout.num_embeddings[index]:
            raise ValueError(f"Layer {layer_id} head sizes exceed its table")
        weight_name, weight_file = _find_weight(
            weight_map, f"layers.{layer_id}.engram.embed.weight"
        )
        scale_name, scale_file = _find_weight(
            weight_map, f"layers.{layer_id}.engram.embed.scale"
        )
        with (
            torch.device("cpu"),
            safe_open(model / weight_file, framework="pt", device="cpu") as weights,
            safe_open(model / scale_file, framework="pt", device="cpu") as scales,
        ):
            weight, scale = (
                weights.get_slice(weight_name),
                scales.get_slice(scale_name),
            )
            offset = 0
            for table_id, rows in zip(_table_ids(layer_id, len(sizes)), sizes):
                packed = _pack_head(weight, scale, offset, rows, metadata)
                table.populate(table_id, [packed], replicate)
                _verify_published_head(store, table, table_id, packed)
                logger.info(
                    "Published Engram table %d: %d bytes", table_id, packed.nbytes
                )
                offset += rows
                del packed

    encoded = json.dumps(manifest, indent=2) + "\n"
    if store.put("engram:manifest", encoded.encode(), replicate) != 0:
        raise RuntimeError("Could not publish the Engram manifest")
    return manifest


@triton.jit(do_not_specialize=["num_rows"])
def _dequant_packed_engram_rows(
    packed,
    hash_ids,
    output,
    num_rows,
    ACTUAL_HEADS: tl.constexpr,
    PART_HEADS: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    HEAD_STRIDE: tl.constexpr,
    HASH_STRIDE: tl.constexpr,
    DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    cols = tl.arange(0, DIM)
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    valid = rows < num_rows
    row = rows.to(tl.int64)
    packed_row = row % ACTUAL_HEADS * HEAD_STRIDE + row // ACTUAL_HEADS * ROW_BYTES
    alive = (
        tl.load(
            hash_ids + row // ACTUAL_HEADS * HASH_STRIDE + row % ACTUAL_HEADS,
            mask=valid,
            other=-1,
        )
        != -1
    )
    values = (
        tl.load(
            packed + packed_row[:, None] + cols[None, :],
            mask=valid[:, None] & alive[:, None],
            other=0,
        )
        .to(tl.float8e4nv, bitcast=True)
        .to(tl.float32)
    )
    scale = tl.load(
        packed + packed_row[:, None] + DIM + cols[None, :] // QUANT_BLOCK,
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
    packed: torch.Tensor
    rows: torch.Tensor


@dataclass
class _Layer:
    store_ids: tuple[int, ...]
    head_start: int
    head_sizes: np.ndarray
    offsets: tuple[int, ...]
    block_size: int
    buffers: list[_LayerBuffers]


@dataclass
class _LookupSlot:
    rows_consumed: torch.cuda.Event = field(
        default_factory=lambda: torch.cuda.Event(external=True)
    )
    lookup: Any = None
    hash_ids: torch.Tensor | None = None


def _release_backend(resources: dict[str, Any], slots: list[_LookupSlot]) -> None:
    store = resources.get("store")
    if store is not None:
        torch.accelerator.synchronize(resources["registered"][0].device.index)
        for slot in slots:
            slot.lookup = None
        resources.pop("table", None)
        for buffer in resources["registered"]:
            rc = store.unregister_buffer(buffer.data_ptr())
            if rc != 0:
                logger.warning("Mooncake Engram buffer unregister failed: %s", rc)
        store.close()


def _validate_manifest(manifest: dict, model: str) -> None:
    if manifest.get("version") != 3 or not manifest.get("publication"):
        raise ValueError(
            "Mooncake Engram requires a version 3 manifest with a publication ID"
        )
    if manifest.get("model") != str(Path(model).expanduser().resolve()):
        raise ValueError("Mooncake Engram manifest belongs to another checkpoint")


class MooncakeEngramBackend:
    """Layer-owned Store reads into reusable CUDA buffers."""

    def __init__(
        self,
        manifest: dict,
        num_slots: int,
        model: str,
    ) -> None:
        _validate_manifest(manifest, model)
        self.manifest = manifest
        self._layer: _Layer | None = None
        self._slots = [_LookupSlot() for _ in range(num_slots)]
        self._connect_lock = threading.Lock()
        self._store: Any = None
        self._table: Any = None
        self._resources: dict[str, Any] = {"store": None, "registered": []}
        self._finalizer = weakref.finalize(
            self,
            _release_backend,
            self._resources,
            self._slots,
        )

    def close(self) -> None:
        self._table = None
        self._finalizer()

    def attach(
        self,
        engram: EngramMooncake,
        layout: EngramLayout,
        max_tokens: int,
    ) -> None:
        embedding = engram.embed_tokens
        hash_index = engram.layer_hash_index
        model_layer_id = layout.layer_ids[hash_index]
        head_sizes = _head_sizes(layout, hash_index)
        if self._layer is not None:
            raise ValueError("An Engram backend must be attached to one layer")
        dim, block = embedding.dim, embedding.block_size
        expected = _layer_manifest(head_sizes, dim, block)
        row_bytes = expected["row_bytes"]
        if self.manifest["layers"].get(str(model_layer_id)) != expected:
            raise ValueError(
                f"Mooncake Engram layout mismatch for layer {model_layer_id}"
            )
        start = embedding.head_start
        end = min(start + embedding.part_n_hash_cols, len(head_sizes))
        sizes = head_sizes[start:end]
        offsets = np.cumsum((0, *head_sizes[:-1]), dtype=np.int64)[start:end]
        device = torch.device("cuda", torch.accelerator.current_device_index())
        shape = (max_tokens, len(sizes))
        buffers = [
            _LayerBuffers(
                packed=torch.empty(
                    (*shape[::-1], row_bytes), dtype=torch.uint8, device=device
                ),
                rows=torch.zeros(
                    (max_tokens, embedding.part_n_hash_cols, dim),
                    dtype=torch.bfloat16,
                    device=device,
                ),
            )
            for _ in self._slots
        ]
        self._layer = _Layer(
            _table_ids(model_layer_id, len(head_sizes))[start:end],
            start,
            np.asarray(sizes, dtype=np.int64),
            tuple(int(offset) for offset in offsets),
            block,
            buffers,
        )
        engram.mooncake_backend = self

    def _connect(self) -> None:
        with self._connect_lock:
            if not self._finalizer.alive:
                raise RuntimeError("Mooncake Engram backend is closed")
            if self._store is not None:
                return
            from mooncake.store import EngramStore

            layer = self._layer
            if layer is None:
                raise RuntimeError("Mooncake Engram has no attached layer")
            if not hasattr(EngramStore, "lookup"):
                raise RuntimeError(
                    "Mooncake Engram requires the lookup registered-output API"
                )
            store = create_store(reader=True)
            registered = self._resources["registered"]
            try:
                ready = store.get("engram:manifest")
                if not ready or json.loads(ready) != self.manifest:
                    raise ValueError(
                        "Mooncake Engram publication does not match the manifest"
                    )
                configs = _store_configs(
                    layer.store_ids, layer.head_sizes, layer.buffers[0].packed.shape[-1]
                )
                table = EngramStore(configs, store_client=store)
                keys = [
                    key
                    for layer_id in configs
                    for key in table.get_store_keys(layer_id)
                ]
                exists = store.batch_is_exist(keys)
                if len(exists) != len(keys) or any(value != 1 for value in exists):
                    raise RuntimeError("Mooncake Engram tables are incomplete")
                for buffer in layer.buffers:
                    packed = buffer.packed
                    if store.register_buffer(packed.data_ptr(), packed.numel()) != 0:
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
            self._resources["table"] = table

    def prefetch(self, hash_ids: torch.Tensor) -> None:
        self._connect()
        layer = self._layer
        assert layer is not None
        slot_index = dbo_current_ubatch_id()
        slot = self._slots[slot_index]
        torch.cuda.current_stream().wait_event(slot.rows_consumed)
        slot.hash_ids = hash_ids
        tokens = hash_ids.shape[0]
        buffer = layer.buffers[slot_index]
        if tokens > buffer.rows.shape[0]:
            raise RuntimeError("Mooncake Engram output buffer is too small")
        heads, width = len(layer.store_ids), buffer.packed.shape[-1]
        packed = buffer.packed.view(-1)[: heads * tokens * width].view(
            heads, tokens, width
        )
        slot.lookup = self._table.lookup(
            layer.store_ids,
            hash_ids[:, layer.head_start : layer.head_start + heads],
            packed,
            stream=torch.cuda.current_stream().cuda_stream,
            offsets=layer.offsets,
        )

    def wait(self) -> None:
        slot = self._slots[dbo_current_ubatch_id()]
        if slot.lookup is None:
            raise RuntimeError("Mooncake Engram rows were not prefetched")
        slot.lookup.wait(torch.cuda.current_stream().cuda_stream)

    def rows(self) -> torch.Tensor:
        self.wait()
        slot_index = dbo_current_ubatch_id()
        layer = self._layer
        assert layer is not None
        buffer = layer.buffers[slot_index]
        hash_ids = self._slots[slot_index].hash_ids
        assert hash_ids is not None
        hash_ids = hash_ids[:, layer.head_start :]
        tokens = hash_ids.shape[0]
        rows = tokens * len(layer.head_sizes)
        if rows:
            _dequant_packed_engram_rows[(triton.cdiv(rows, 16),)](
                buffer.packed,
                hash_ids,
                buffer.rows,
                rows,
                ACTUAL_HEADS=len(layer.head_sizes),
                PART_HEADS=buffer.rows.shape[1],
                ROW_BYTES=buffer.packed.shape[-1],
                HEAD_STRIDE=tokens * buffer.packed.shape[-1],
                HASH_STRIDE=hash_ids.stride(0),
                DIM=buffer.rows.shape[-1],
                QUANT_BLOCK=layer.block_size,
                BLOCK_R=16,
            )
        self._slots[slot_index].rows_consumed.record(torch.cuda.current_stream())
        return buffer.rows[:tokens]
