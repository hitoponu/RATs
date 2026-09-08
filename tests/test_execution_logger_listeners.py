from __future__ import annotations

import pytest

from rats.utils import execution_logger as el


@pytest.fixture
def events():
    got: list[dict] = []
    el.register_policy_step_listener(got.append)
    try:
        yield got
    finally:
        el.unregister_policy_step_listener(got.append)


def test_begin_end_events(events):
    el.init_execution_context(code_block_index=0)
    with el.policy_step_context("step-1", "grasp", step_index=0):
        pass
    el.finalize_execution_context()
    assert [e["phase"] for e in events] == ["begin", "end"]
    assert events[0]["step_index"] == 0 and events[0]["step_id"] == "step-1"
    assert events[1]["exc_in_flight"] is False


def test_exception_marks_end_event(events):
    el.init_execution_context(code_block_index=0)
    with pytest.raises(RuntimeError):
        with el.policy_step_context("step-2", "place", step_index=1):
            raise RuntimeError("boom")
    el.finalize_execution_context()
    assert events[-1]["phase"] == "end"
    assert events[-1]["exc_in_flight"] is True
    assert events[-1]["step_index"] == 1


def test_registry_survives_context_reset(events):
    el.init_execution_context(code_block_index=0)
    el.finalize_execution_context()
    el.clear_all_histories()
    el.init_execution_context(code_block_index=1)
    with el.policy_step_context("step-1", "x", step_index=0):
        pass
    el.finalize_execution_context()
    assert len(events) == 2


def test_listener_errors_do_not_propagate():
    def bad(_e):
        raise ValueError("listener bug")

    el.register_policy_step_listener(bad)
    try:
        el.init_execution_context(code_block_index=0)
        with el.policy_step_context("step-1", "x", step_index=0):
            pass
        el.finalize_execution_context()
    finally:
        el.unregister_policy_step_listener(bad)
