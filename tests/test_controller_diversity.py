"""The diversity half, wired through the controller (fake env, fake writer)."""

from __future__ import annotations

import pytest
from conftest import FakeLowLevelEnv
from test_controller import AVAIL, BOWL, CODE, PLAN, PLATE, SCRIPT, _simulate_policy, _skill

from rats.step_growth.config import StepGrowthConfig
from rats.step_growth.controller import StepGrowthController
from rats.step_growth.strategy_portfolio import Family, StrategyBank, StrategyPortfolio
from rats.utils import execution_logger as el
from skill_library.step_credit_library import StepCreditSkillLibrary

GOAL = [("on", BOWL, PLATE)]
TASK = {"language": "put the black bowl on the plate", "activity_name": "put_bowl_on_plate"}
SCENE = {"available_functions": AVAIL, "bddl_path": "/tmp/x.bddl"}


class FakeWriter:
    """Stands in for DirectivePolicyWriter (no LLM stack in the test venv)."""

    def __init__(self) -> None:
        self.pending_directive = ""
        self.pending_directive_meta: dict = {}


def _bank() -> StrategyBank:
    return StrategyBank(
        [
            Family("A", ("grasp",), "use the planner", ("plan_grasp",), (), ("plan_grasp",)),
            Family("B", ("grasp",), "use the OBB", ("goto_pose",), (), ("obb_yaw",)),
            Family("P", ("place",), "hover and release", ("open_gripper",), (), ("place",)),
        ],
        ["never hard-code a quaternion"], source="test-bank",
    )


@pytest.fixture
def ctrl(tmp_path):
    """Controller with the diversity half on, over the same fake env as test_controller."""
    lib = StepCreditSkillLibrary(storage_path=str(tmp_path / "skills.json"))
    lib._find_semantic_duplicate = lambda *a, **k: None
    lib.add_skill(_skill("grasp_helper", "    g = plan_grasp(segment_sam3_text_prompt(object_name))\n"
                                         "    goto_pose(g)\n    close_gripper()\n    return {'grasp': g}\n"),
                  available_functions=AVAIL)
    writer = FakeWriter()
    portfolio = StrategyPortfolio(_bank(), tmp_path / "run" / "step_growth" / "strategy_state.json",
                                  rng_seed=0, min_pulls=1, collapse_k=3)
    cfg = StepGrowthConfig(extraction_enabled=False)
    c = StepGrowthController(cfg, output_dir=tmp_path / "run", library_getter=lambda: lib,
                             writer_getter=lambda: writer, portfolio=portfolio)
    yield c, lib, writer, portfolio
    el.unregister_policy_step_listener(c.recorder.on_step_event)


def _run_attempt(c, *, attempt: int, attempt_in_iter: int, iteration_data: dict, code: str = CODE):
    env = FakeLowLevelEnv(SCRIPT, goal_state=GOAL)
    env._pick_baseline_z = {BOWL: 0.90, PLATE: 0.88}
    c.on_attempt_start(env, iteration=7, attempt=attempt, attempt_in_iter=attempt_in_iter,
                       turn_in_attempt=0, env_reset=True)
    c.note_code(iteration=7, attempt=attempt, attempt_in_iter=attempt_in_iter, code=code,
                scene_context=SCENE, iteration_data=iteration_data)
    c.on_execution_start(iteration=7, attempt=attempt, attempt_in_iter=attempt_in_iter,
                         turn_in_attempt=0, env_reset=True)
    _simulate_policy(env)
    return c.on_attempt_executed(
        execution_result={"artifacts": {}}, plan=PLAN, code=code, attempt=attempt,
        attempt_in_iter=attempt_in_iter, turn_in_attempt=0, scene_context=SCENE,
        task_proposal=TASK, iteration_data=iteration_data,
    )


def test_directive_is_armed_scored_and_disarmed(ctrl):
    c, _, writer, portfolio = ctrl
    iteration_data = {"iteration": 7}
    c.on_plan_ready(plan=PLAN, task_proposal=TASK, scene_context=SCENE, iteration_data=iteration_data)
    assert writer.pending_directive == ""

    summary = _run_attempt(c, attempt=0, attempt_in_iter=0, iteration_data=iteration_data)

    # the writer is armed for the attempt that is about to be written
    assert writer.pending_directive.startswith("## STRATEGY DIRECTIVE (HARD")
    assert "never hard-code a quaternion" in writer.pending_directive
    assert writer.pending_directive_meta["families"]["grasp"] in {"A", "B"}
    # attempt 0 leaves no policy-writer prompt on disk (no retry feedback), so
    # the arm saves the armed text itself for the audit
    armed = c.output_dir / "iteration_007" / "strategy_directive_attempt00.txt"
    assert armed.exists() and armed.read_text() == writer.pending_directive

    div = iteration_data["step_growth"]["diversity"]
    assert div["enabled"] is True and div["bank"] == "test-bank"
    assert div["selections"]["0"]["families"] == summary["strategy"]["families"]
    fp = div["fingerprints"]["0"]
    assert fp["step_markers"] == 5 and fp["skill_calls"] == {"grasp_helper": 1}

    # the bandit is not paid until the attempt is over ...
    assert portfolio.stats_snapshot() == {}
    c.on_iteration_end(success=False, iteration_data=iteration_data, scene_context=SCENE, task_proposal=TASK)
    # ... and then it is, from the milestones the oracle recorded (lift reached,
    # place dropped) -- grasp rewarded, place not.
    stats = portfolio.stats_snapshot()
    grasp_fam = div["selections"]["0"]["families"]["grasp"]
    assert stats[grasp_fam]["pulls"] == 1 and stats[grasp_fam]["rewards"] == 1
    assert stats["P"]["pulls"] == 1 and stats["P"]["rewards"] == 0
    assert div["misses"] == {"grasp": 0, "place": 1}
    assert writer.pending_directive == ""          # disarmed for the next iteration
    assert (div["family_stats"] or {})[grasp_fam]["pulls"] == 1


def test_families_are_sticky_across_turns_and_switch_on_repeated_misses(ctrl):
    c, _, writer, _ = ctrl
    iteration_data = {"iteration": 7}
    c.on_plan_ready(plan=PLAN, task_proposal=TASK, scene_context=SCENE, iteration_data=iteration_data)
    _run_attempt(c, attempt=0, attempt_in_iter=0, iteration_data=iteration_data)
    first = dict(c._iter["selection"].families)

    # a within-attempt turn transition never re-rolls the strategy
    assert c.on_retry(iteration=7, attempt=1, attempt_in_iter=0, attempt_boundary=False, plan=PLAN,
                      retry_feedback={"failure_mode": "grasp_failure"}, diagnosis=None,
                      iteration_data=iteration_data) is None
    assert c._iter["selection"].families == first

    # one miss at an attempt boundary: keep the family (could be an argument fix)
    out = c.on_retry(iteration=7, attempt=1, attempt_in_iter=0, attempt_boundary=True, plan=PLAN,
                     retry_feedback={"failure_mode": "grasp_failure"}, diagnosis=None,
                     iteration_data=iteration_data)
    assert out["families"]["place"] == first["place"]
    assert not out["notes"]

    # two misses of the same type with a physical failure mode: switch
    c._iter["misses"] = {"grasp": 2}
    out = c.on_retry(iteration=7, attempt=2, attempt_in_iter=1, attempt_boundary=True, plan=PLAN,
                     retry_feedback={"failure_mode": "grasp_failure"}, diagnosis=None,
                     iteration_data=iteration_data)
    assert out["families"]["grasp"] != first["grasp"]
    assert any("FORBIDDEN" in n for n in out["notes"])
    assert "SWITCH:" in writer.pending_directive


def test_collapse_override_reaches_the_next_iteration(ctrl):
    c, _, _, portfolio = ctrl
    handbuilt = 'q = np.array([1.0, 0.0, 0.0, 0.0])\ngoto_pose([0, 0, 0], q)\nclose_gripper()\n'
    for it in range(3):
        data = {"iteration": it}
        c.on_plan_ready(plan=PLAN, task_proposal=TASK, scene_context=SCENE, iteration_data=data)
        c.note_code(iteration=it, attempt=0, attempt_in_iter=0, code=handbuilt,
                    scene_context=SCENE, iteration_data=data)
    assert portfolio.collapse and portfolio.collapse["family"] == "handbuilt_identity"


def test_arm_without_diversity_is_untouched(tmp_path):
    """Same controller, portfolio=None: no diversity key, no directive, no crash."""
    lib = StepCreditSkillLibrary(storage_path=str(tmp_path / "skills.json"))
    writer = FakeWriter()
    c = StepGrowthController(StepGrowthConfig(extraction_enabled=False), output_dir=tmp_path / "run",
                             library_getter=lambda: lib, writer_getter=lambda: writer,
                             register_listener=False)
    assert c.portfolio is None
    iteration_data = {"iteration": 1}
    c.on_plan_ready(plan=PLAN, task_proposal=TASK, scene_context=SCENE, iteration_data=iteration_data)
    assert c.note_code(iteration=1, attempt=0, attempt_in_iter=0, code=CODE,
                       scene_context=SCENE, iteration_data=iteration_data) is None
    assert c.on_retry(iteration=1, attempt=1, attempt_in_iter=0, attempt_boundary=True, plan=PLAN,
                      retry_feedback={"failure_mode": "grasp_failure"}, diagnosis=None,
                      iteration_data=iteration_data) is None
    c.on_iteration_end(success=False, iteration_data=iteration_data, scene_context=SCENE, task_proposal=TASK)
    assert "diversity" not in iteration_data["step_growth"]
    assert writer.pending_directive == ""


def test_diversity_survives_a_broken_writer(ctrl, tmp_path):
    """A writer that cannot carry a directive must not take the loop down."""
    c, lib, _, portfolio = ctrl
    c._writer_getter = lambda: object()
    iteration_data = {"iteration": 7}
    c.on_plan_ready(plan=PLAN, task_proposal=TASK, scene_context=SCENE, iteration_data=iteration_data)
    summary = _run_attempt(c, attempt=0, attempt_in_iter=0, iteration_data=iteration_data)
    assert summary["strategy"]["families"]           # selection still happened
    c.on_iteration_end(success=False, iteration_data=iteration_data, scene_context=SCENE, task_proposal=TASK)
    assert portfolio.stats_snapshot()                 # and was still scored


def test_env_var_gates_the_diversity_half(monkeypatch):
    """Three states: env var wins both ways, config decides when it is unset."""
    from rats.step_growth.config import diversity_enabled, load_config

    monkeypatch.delenv("RATS_STEP_GROWTH", raising=False)
    monkeypatch.delenv("RATS_STEP_GROWTH_DIVERSITY", raising=False)
    assert diversity_enabled() is False
    monkeypatch.setenv("RATS_STEP_GROWTH", "1")
    assert diversity_enabled() is False              # rats/config/step_growth.yaml ships it off
    monkeypatch.setenv("RATS_STEP_GROWTH_DIVERSITY", "1")
    assert diversity_enabled() is True

    cfg = load_config()
    cfg.diversity_enabled = True
    monkeypatch.setenv("RATS_STEP_GROWTH_DIVERSITY", "0")
    assert diversity_enabled(cfg) is False
    monkeypatch.delenv("RATS_STEP_GROWTH_DIVERSITY")
    assert diversity_enabled(cfg) is True
    monkeypatch.delenv("RATS_STEP_GROWTH")
    assert diversity_enabled(cfg) is False           # never without the arm itself


def test_maybe_create_builds_the_shipped_portfolio(tmp_path, monkeypatch):
    monkeypatch.setenv("RATS_STEP_GROWTH", "1")
    monkeypatch.setenv("RATS_STEP_GROWTH_DIVERSITY", "1")
    c = StepGrowthController.maybe_create(
        output_dir=tmp_path, env_type="libero", library_getter=lambda: None,
    )
    assert c is not None
    try:
        assert c.portfolio is not None
        assert {"graspnet_6dof", "obb_topdown_yaw"} <= {f.id for f in c.portfolio.bank.families}
        assert c.portfolio.state_path == tmp_path / "step_growth" / "strategy_state.json"
    finally:
        el.unregister_policy_step_listener(c.recorder.on_step_event)

    monkeypatch.setenv("RATS_STEP_GROWTH_DIVERSITY", "0")
    off = StepGrowthController.maybe_create(
        output_dir=tmp_path, env_type="libero", library_getter=lambda: None,
    )
    assert off is not None and off.portfolio is None
    el.unregister_policy_step_listener(off.recorder.on_step_event)


def test_switching_restarts_the_miss_streak(ctrl):
    """Regression (smoke 5258527): the streak never reset, so every later retry
    switched again and one iteration burned the whole family pool."""
    c, _, _, _ = ctrl
    iteration_data = {"iteration": 7}
    c.on_plan_ready(plan=PLAN, task_proposal=TASK, scene_context=SCENE, iteration_data=iteration_data)
    _run_attempt(c, attempt=0, attempt_in_iter=0, iteration_data=iteration_data)
    # first retry flushes the attempt's reward (which recomputes the streaks)
    c.on_retry(iteration=7, attempt=1, attempt_in_iter=0, attempt_boundary=True, plan=PLAN,
               retry_feedback={"failure_mode": "grasp_failure"}, diagnosis=None,
               iteration_data=iteration_data)
    c._iter["misses"] = {"grasp": 2, "place": 1}
    out = c.on_retry(iteration=7, attempt=2, attempt_in_iter=1, attempt_boundary=True, plan=PLAN,
                     retry_feedback={"failure_mode": "grasp_failure"}, diagnosis=None,
                     iteration_data=iteration_data)
    assert out["reasons"]["grasp"].startswith("switched")
    assert c._iter["misses"]["grasp"] == 0        # the new family starts fresh
    assert c._iter["misses"]["place"] == 1        # untouched type keeps its count

    # ... so the very next retry does NOT switch again
    again = c.on_retry(iteration=7, attempt=3, attempt_in_iter=2, attempt_boundary=True, plan=PLAN,
                       retry_feedback={"failure_mode": "grasp_failure"}, diagnosis=None,
                       iteration_data=iteration_data)
    assert again["families"]["grasp"] == out["families"]["grasp"]
    assert not again["notes"]
