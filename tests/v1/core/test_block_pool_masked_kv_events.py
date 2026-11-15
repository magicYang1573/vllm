from vllm.distributed.kv_events import BlockStored
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.request import Request


def test_block_pool_cache_full_blocks_skips_null_blocks_cleanly():
    init_none_hash(sha256)
    bs = 16
    pool = BlockPool(
        num_gpu_blocks=16,
        enable_caching=True,
        hash_block_size=bs,
        enable_kv_cache_events=True,
    )
    toks = [200 + i for i in range(48)]
    req = Request(
        "req_mask",
        toks,
        SamplingParams(max_tokens=4),
        None,
        block_hasher=get_request_block_hasher(bs, sha256),
    )
    b0, b2 = pool.get_new_blocks(2)
    blocks = [b0, pool.null_block, b2]

    pool.cache_full_blocks(
        request=req,
        blocks=blocks,
        num_cached_blocks=0,
        num_full_blocks=3,
        block_size=bs,
        kv_cache_group_id=0,
    )
    events = [e for e in pool.take_events() if isinstance(e, BlockStored)]
    assert len(events) == 2
    for ev in events:
        assert len(ev.block_hashes) == 1
        assert len(ev.token_ids) == bs
    assert list(events[0].token_ids) == toks[:bs]
    assert list(events[1].token_ids) == toks[2 * bs : 3 * bs]
