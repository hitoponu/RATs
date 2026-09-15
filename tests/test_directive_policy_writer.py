"""The strategy directive has to actually reach the writer's prompt."""

from __future__ import annotations

from pathlib import Path

import pytest

# PolicyWriter pulls in the LLM client stack; skip rather than fail on a venv
# that only has the pure-python test deps.
pytest.importorskip("requests")

from rats.agents.directive_policy_writer import DirectivePolicyWriter, make_policy_writer  # noqa: E402
from rats.agents.policy_writer import PolicyWriter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DIRECTIVE = "## STRATEGY DIRECTIVE (HARD — priority 0, overrides all other guidance below)\n  GRASP — ask plan_grasp\n"
PLAN = {"steps": [{"id": "step-1", "description": "grasp the bowl"}]}
SCENE = {"env_type": "libero", "available_functions": ["plan_grasp", "goto_pose", "close_gripper"]}


@pytest.fixture
def writer(monkeypatch):
    import rats.agents.policy_writer as pw

    monkeypatch.chdir(ROOT)  # write() reads rats/prompts/policy_writer.txt relatively
    monkeypatch.setattr(pw, "query_llm_text", lambda system, prompt, **kw: "def main(env):\n    pass\n")
    return DirectivePolicyWriter(max_retries=1, ensemble_n=0)


def test_pending_directive_lands_in_the_priority_zero_slot(writer):
    writer.write(PLAN, SCENE)
    assert "STRATEGY DIRECTIVE" not in writer.last_user_prompt_text

    writer.pending_directive = DIRECTIVE
    writer.write(PLAN, SCENE)
    prompt = writer.last_user_prompt_text
    assert "STRATEGY DIRECTIVE" in prompt
    # priority 0 == above the advisory lessons block
    assert prompt.index("STRATEGY DIRECTIVE") < prompt.index("LESSONS FROM PAST FAILURES")


def test_explicit_subagent_directive_still_wins(writer):
    """The parallel sub-agent path must not be hijacked by the sticky one."""
    writer.pending_directive = DIRECTIVE
    writer.write(PLAN, SCENE, subagent_directive="## SUB-AGENT DIRECTIVE (HARD)\n  Strategy: push it\n")
    assert "SUB-AGENT DIRECTIVE" in writer.last_user_prompt_text
    assert "STRATEGY DIRECTIVE" not in writer.last_user_prompt_text


def test_empty_directive_is_the_plain_writer(writer):
    writer.pending_directive = ""
    writer.write(PLAN, SCENE)
    assert "STRATEGY DIRECTIVE" not in writer.last_user_prompt_text


def test_factory_returns_a_plain_writer_unless_diversity_is_on(monkeypatch):
    monkeypatch.delenv("RATS_STEP_GROWTH", raising=False)
    monkeypatch.delenv("RATS_STEP_GROWTH_DIVERSITY", raising=False)
    assert type(make_policy_writer(max_retries=1, ensemble_n=0)) is PolicyWriter

    # the arm alone is not enough: diversity ships off (separate arm)
    monkeypatch.setenv("RATS_STEP_GROWTH", "1")
    assert type(make_policy_writer(max_retries=1, ensemble_n=0)) is PolicyWriter

    monkeypatch.setenv("RATS_STEP_GROWTH_DIVERSITY", "1")
    assert isinstance(make_policy_writer(max_retries=1, ensemble_n=0), DirectivePolicyWriter)

    # and the env var can force it off again
    monkeypatch.setenv("RATS_STEP_GROWTH_DIVERSITY", "0")
    assert type(make_policy_writer(max_retries=1, ensemble_n=0)) is PolicyWriter
