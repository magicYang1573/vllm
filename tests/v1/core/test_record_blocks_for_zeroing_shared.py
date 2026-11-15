import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheGroup,
    KVQuantMode,
)
from vllm.v1.request import Request


def test_record_blocks_for_zeroing_preserves_shared_prefix_blocks():
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
    grp = KVCacheGroup({"l0": spec}, FullAttentionManager.Config(), bs, False)
    mgr = KVCacheManager(
        KVCacheConfig(num_blocks=16, kv_cache_tensors=[], kv_cache_groups=[grp]),
        max_model_len=256,
        hash_block_size=bs,
        enable_caching=True,
        needs_kv_cache_zeroing=True,
    )
    hasher = get_request_block_hasher(bs, sha256)
    shared = [300 + i for i in range(32)]

    req_a = Request("req_a", shared + [1, 2], SamplingParams(max_tokens=4), None, block_hasher=hasher)
    hits_a, _ = mgr.get_computed_blocks(req_a)
    slots_a = mgr.allocate_slots(req_a, len(req_a.prompt_token_ids), new_computed_blocks=hits_a)
    assert slots_a is not None
    req_a.num_computed_tokens = 32
    shared_ids = {b.block_id for b in slots_a.blocks[0][:2]}
    mgr.take_new_block_ids()

    # Tenant B hits shared prefix [0..32) + allocates private tail [32..48), then encounters load failure.
    req_b = Request("req_b", shared + [400 + i for i in range(16)], SamplingParams(max_tokens=4), None, block_hasher=hasher)
    hits_b, num_b = mgr.get_computed_blocks(req_b)
    assert num_b == 32
    slots_b = mgr.allocate_slots(req_b, 16, new_computed_blocks=hits_b)
    assert slots_b is not None
    mgr.take_new_block_ids()

    mgr.record_blocks_for_zeroing("req_b")
    zeroed = set(mgr.take_new_block_ids()[0])
    assert zeroed.isdisjoint(shared_ids)
    assert 0 not in zeroed
