import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheGroup,
    KVCacheLayerConfig,
    KVQuantMode,
)
from vllm.v1.request import Request


def test_hybrid_eagle_cache_hit_shrink_aligns_groups():
    init_none_hash(sha256)
    bs = 16
    spec = AttentionSpec(
        block_size=bs,
        num_kv_heads=2,
        head_size=16,
        dtype=torch.float16,
        page_size_padded=None,
        kv_quant_mode=KVQuantMode.NONE,
    )
    full_cfg = KVCacheLayerConfig(spec=spec, manager_cfg=FullAttentionManager.Config())
    swa_cfg = KVCacheLayerConfig(
        spec=spec,
        manager_cfg=SlidingWindowManager.Config(
            SlidingWindowManager.Params(sliding_window=2 * bs)
        ),
    )
    g0 = KVCacheGroup({"l0": spec}, full_cfg.manager_config, bs, False, is_eagle_group=True)
    g1 = KVCacheGroup({"l1": spec}, swa_cfg.manager_config, bs, False, is_eagle_group=True)
    mgr = KVCacheManager(
        KVCacheConfig(num_blocks=6, kv_cache_tensors=[], kv_cache_groups=[g0, g1]),
        max_model_len=512,
        hash_block_size=bs,
        enable_caching=True,
        use_eagle=True,
    )

    hasher = get_request_block_hasher(bs, sha256)
    toks_a = [100 + i for i in range(64)]
    req_a = Request("req_a", toks_a, SamplingParams(max_tokens=4), None, block_hasher=hasher)
    hits_a, _ = mgr.get_computed_blocks(req_a)
    assert mgr.allocate_slots(req_a, 64, new_computed_blocks=hits_a) is not None
    req_a.num_computed_tokens = 64
    mgr.free(req_a)

    # Evict middle block via pressure request so SWA group shrinks hit_length cleanly.
    toks_ev = [900 + i for i in range(48)]
    req_ev = Request("req_ev", toks_ev, SamplingParams(max_tokens=4), None, block_hasher=hasher)
    hits_ev, _ = mgr.get_computed_blocks(req_ev)
    assert mgr.allocate_slots(req_ev, 48, new_computed_blocks=hits_ev) is not None
    mgr.free(req_ev)

    req_b = Request("req_b", toks_a + [777], SamplingParams(max_tokens=4), None, block_hasher=hasher)
    hits_b, num_hit = mgr.get_computed_blocks(req_b)
    assert len(hits_b.blocks[0]) == len(hits_b.blocks[1])
    assert len(hits_b.blocks[0]) * bs == num_hit
