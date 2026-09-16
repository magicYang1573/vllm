# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from pydantic import Field, model_validator
from typing_extensions import Self

import vllm.envs as envs
from vllm.config.utils import config, get_hash_factors, hash_factors

if TYPE_CHECKING:
    from vllm.config.load import LoadConfig
    from vllm.config.model import ModelConfig
    from vllm.config.parallel import ParallelConfig

# Architecture -> the hf_text_config field naming its n-gram layers. A model is
# only configurable here if it actually has such layers to store.
_NGRAM_LAYER_FIELDS = {
    "DeepseekV41ForCausalLM": "engram_layer_ids",
    "Qwen4ExpForCausalLM": "ple_layer_ids",
    "Qwen4ExpForConditionalGeneration": "ple_layer_ids",
}


def _default_cpu_offload() -> bool:
    return envs.VLLM_PLE_CPU_OFFLOAD


def model_has_engram_layers(model_config: "ModelConfig | None") -> bool:
    """Whether the model carries n-gram embedding layers."""
    if model_config is None:
        return False
    field = _NGRAM_LAYER_FIELDS.get(model_config.architecture)
    if field is None:
        return False
    return bool(getattr(model_config.hf_text_config, field, None))


def _default_mooncake_config_path() -> str | None:
    return envs.VLLM_ENGRAM_MOONCAKE_CONFIG or None


@config
class EngramConfig:
    """Configuration for Engram embedding storage and sharding."""

    cpu_offload: bool = Field(default_factory=_default_cpu_offload)
    """Store embedding weights in pinned CPU memory for UVA lookup.
    Defaults to VLLM_PLE_CPU_OFFLOAD, which is enabled by default. An explicit
    value takes precedence over the environment variable."""

    embedding_across_dp: bool = False
    """Shard embeddings across TP and all DP ranks when enabled.
    Otherwise, each DP rank has a separate TP-sharded embedding replica."""

    dp_shared_memory: bool = False
    """Share CPU-offloaded embedding weights between co-located
    DP replicas. Each node stores one copy of every TP shard, reducing host
    memory without per-step Engram DP collectives. Requires sufficient
    /dev/shm capacity and a shared IPC namespace."""

    mooncake_config_path: str | None = Field(
        default_factory=_default_mooncake_config_path
    )
    """Published manifest for Engram tables served by Mooncake Store.

    This placement takes precedence over cpu_offload and reads directly into CUDA.

    Mooncake connection settings are read from the standard ``MOONCAKE_*``
    environment variables. The backend batches all local Engram layers into
    one ranged-read submission and only fetches the hash heads owned by this
    TP/Engram-DP rank. Defaults to VLLM_ENGRAM_MOONCAKE_CONFIG."""

    @model_validator(mode="after")
    def _validate_shared_memory(self) -> Self:
        if self.dp_shared_memory and not self.cpu_offload:
            raise ValueError("dp_shared_memory requires cpu_offload=True")
        if self.dp_shared_memory and self.mooncake_config_path:
            raise ValueError(
                "dp_shared_memory and Mooncake are alternative Engram placements"
            )
        return self

    def verify_model_config(self, model_config: "ModelConfig | None") -> None:
        """Reject Engram configuration for models without n-gram embeddings."""
        from vllm.platforms import current_platform

        field = (
            _NGRAM_LAYER_FIELDS.get(model_config.architecture)
            if model_config is not None
            else None
        )
        if (
            model_config is None
            or field is None
            or not current_platform.is_cuda()
            or not getattr(model_config.hf_text_config, field, None)
        ):
            raise ValueError(
                "EngramConfig requires a model with supported Engram "
                "embeddings, non-empty n-gram layer ids, and CUDA."
            )

        if (
            self.mooncake_config_path
            and model_config.architecture != "DeepseekV41ForCausalLM"
        ):
            raise ValueError("Mooncake Engram only supports DeepSeek V4.1")

    def verify_parallel_config(self, parallel_config: "ParallelConfig") -> None:
        """Reject unsupported embedding parallel topologies."""
        if self.dp_shared_memory:
            if parallel_config.data_parallel_size <= 1:
                raise ValueError("dp_shared_memory requires data_parallel_size > 1.")
            if parallel_config.enable_elastic_ep:
                raise ValueError("dp_shared_memory is not supported with elastic EP.")
        if (
            self.embedding_across_dp
            and parallel_config.data_parallel_size > 1
            and parallel_config.enable_elastic_ep
        ):
            raise ValueError(
                "Engram embedding_across_dp is not supported with elastic EP yet."
            )

    def verify_load_config(self, load_config: "LoadConfig") -> None:
        """Shared tables require a loader that invokes parameter weight callbacks."""
        if self.mooncake_config_path:
            if load_config.load_format not in ("safetensors", "dummy"):
                raise ValueError(
                    "mooncake_config_path requires load_format 'safetensors' "
                    f"or 'dummy'; got {load_config.load_format!r}."
                )
            if (
                load_config.load_format == "safetensors"
                and load_config.safetensors_load_strategy != "lazy"
            ):
                raise ValueError(
                    "mooncake_config_path requires "
                    "safetensors_load_strategy='lazy'; automatic, eager, or "
                    "prefetch loading may read the external Engram tables "
                    "into local memory."
                )
        if self.dp_shared_memory and load_config.load_format not in (
            "auto",
            "safetensors",
            "pt",
        ):
            raise ValueError(
                "dp_shared_memory requires load_format 'auto', "
                f"'safetensors' or 'pt'; got {load_config.load_format!r}."
            )

    def get_parallel_size(self, parallel_config: "ParallelConfig") -> int:
        """Derive the embedding group size from the parallel configuration."""
        size = parallel_config.tensor_parallel_size
        if self.embedding_across_dp and parallel_config.data_parallel_size > 1:
            size *= parallel_config.data_parallel_size
        return size

    def compute_hash(self) -> str:
        """Hash settings that affect embedding execution and graph structure."""
        return hash_factors(get_hash_factors(self, set()))
