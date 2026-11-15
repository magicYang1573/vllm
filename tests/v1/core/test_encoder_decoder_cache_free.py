from types import SimpleNamespace
from vllm.v1.core.encoder_cache_manager import EncoderDecoderCacheManager


def _make_enc_dec_request(req_id: str, num_embeds: int) -> SimpleNamespace:
    mm_feat = SimpleNamespace(
        identifier=f"audio_{req_id}",
        mm_position=SimpleNamespace(offset=0, length=num_embeds),
    )
    return SimpleNamespace(
        request_id=req_id,
        mm_features=[mm_feat],
        get_num_encoder_embeds=lambda idx: num_embeds,
    )


def test_encoder_decoder_cache_manager_no_duplicate_free_inflation():
    mgr = EncoderDecoderCacheManager(cache_size=256)
    req0 = _make_enc_dec_request("req_0", 192)

    assert mgr.check_and_update_cache(req0, 0) is False
    assert mgr.can_allocate(req0, 0, 0, 0) is True
    mgr.allocate(req0, 0)
    assert mgr.num_free_slots == 64
    assert mgr.get_cached_input_ids(req0) == {0}

    # End of prefill frees encoder input 0 once.
    for input_id in list(mgr.get_cached_input_ids(req0)):
        mgr.free_encoder_input(req0, input_id)
    assert mgr.num_free_slots == 256
    assert mgr.get_cached_input_ids(req0) == set()

    # Subsequent decode steps and final request completion must not re-credit slots.
    for _ in range(3):
        for input_id in list(mgr.get_cached_input_ids(req0)):
            mgr.free_encoder_input(req0, input_id)
    mgr.free_encoder_input(req0, 0)
    mgr.free(req0)
    assert mgr.num_free_slots == 256

    # Subsequent admission must respect physical 256-slot capacity.
    req_a = _make_enc_dec_request("req_a", 192)
    req_b = _make_enc_dec_request("req_b", 192)
    assert mgr.can_allocate(req_a, 0, 0, 0) is True
    mgr.allocate(req_a, 0)
    assert mgr.num_free_slots == 64
    assert mgr.can_allocate(req_b, 0, 0, 0) is False
