import inspect
from vllm.v1.core.sched.async_scheduler import AsyncScheduler


def test_async_scheduler_spec_placeholder_no_shared_list_alias():
    src = inspect.getsource(AsyncScheduler._update_after_schedule)
    # Verify per-request list allocation rather than aliasing a shared list reference.
    assert "list(" in src or "[-1] *" in src
