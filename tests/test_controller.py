"""End-to-end: fake env + real execution_logger markers + real library."""

from __future__ import annotations

import json

import pytest
from conftest import FakeLowLevelEnv, snap

from rats.step_growth.config import StepGrowthConfig
from rats.step_growth.controller import StepGrowthController
from rats.utils import execution_logger as el
from skill_library.step_credit_library import StepCreditSkillLibrary

BOWL, PLATE = "akita_black_bowl_1", "plate_1"
ON = f"[on {BOWL} {PLATE}]"
AVAIL = ["segment_sam3_text_prompt", "plan_grasp", "goto_pose", "close_gripper", "open_gripper", "get_observation"]

BASE = {BOWL: (0.0, 0.0, 0.90), PLATE: (0.30, 0.0, 0.88)}
LIFTED = {BOWL: (0.0, 0.0, 0.95), PLATE: (0.30, 0.0, 0.88)}
OVER = {BOWL: (0.30, 0.0, 0.95), PLATE: (0.30, 0.0, 0.88)}
FELL = {BOWL: (0.30, 0.18, 0.88), PLATE: (0.30, 0.0, 0.88)}

# one snapshot per boundary: (begin s0, end s0, begin s1, end s1, ...)
SCRIPT = [
    snap(BASE, goal={ON: False}), snap(BASE, goal={ON: False}),
    snap(BASE, goal={ON: False}), snap(BASE, contact=[BOWL], goal={ON: False}),
    snap(BASE, contact=[BOWL], goal={ON: False}), snap(LIFTED, contact=[BOWL], goal={ON: False}),
    snap(LIFTED, contact=[BOWL], goal={ON: False}), snap(OVER, contact=[BOWL], goal={ON: False}),
    snap(OVER, contact=[BOWL], goal={ON: False}), snap(FELL, goal={ON: False}),
]

CODE = '''def main():
    with step_context("step-1", "localize the bowl", step_index=0):
        obs = get_observation()
        mask = segment_sam3_text_prompt(obs, "black bowl")
    with step_context("step-2", "grasp the bowl", step_index=1):
        r = grasp_helper("black bowl")
    with step_context("step-3", "lift", step_index=2):
        goto_pose(r["grasp"], dz=0.08)
    with step_context("step-4", "move over the plate", step_index=3):
        goto_pose([0.3, 0.0, 0.95])
    with step_context("step-5", "place on the plate", step_index=4):
        place_helper("plate")
    RESULT = {"success": False}
main()
'''

PLAN = {"steps": [{"id": f"step-{i + 1}", "description": d} for i, d in enumerate(
    ["localize", "grasp", "lift", "move", "place"])]}

GOOD = '''def grasp_and_lift_with_planned_pose(object_name: str, lift_height: float = 0.08) -> dict:
    obs = get_observation()
    mask = segment_sam3_text_prompt(obs, object_name)
    grasp = plan_grasp(obs, mask)
    goto_pose(grasp)
    close_gripper()
    goto_pose(grasp, dz=lift_height)
    return {"success": True}
'''


class FakeExtractor:
    def __init__(self):
        self.calls = []

    def extract(self, **kw):
        self.calls.append(kw)
        return {"skill": {"name": "grasp_and_lift_with_planned_pose", "description": "grasp and lift", "code": GOOD,
                          "strategy_tag": "graspnet_6dof", "api_primitives_used": ["plan_grasp"], "preconditions": [],
                          "effects": [], "params": [], "returns": {}, "usage_example": "", "extraction_rationale": ""},
                "rejected_reason": None, "llm_reason": "ok", "raw": {}, "llm_called": True}


def _skill(name: str, body: str) -> dict:
    return {"name": name, "description": name, "code": f"def {name}(object_name: str) -> dict:\n{body}",
            "api_primitives_used": [], "preconditions": [], "effects": [], "source_task": "t", "learned_iteration": 0}


@pytest.fixture
def ctrl(tmp_path):
    lib = StepCreditSkillLibrary(storage_path=str(tmp_path / "skills.json"))
    lib._find_semantic_duplicate = lambda *a, **k: None
    assert lib.add_skill(_skill("grasp_helper", "    g = plan_grasp(segment_sam3_text_prompt(object_name))\n    goto_pose(g)\n    close_gripper()\n    return {'grasp': g}\n"), available_functions=AVAIL)
    assert lib.add_skill(_skill("place_helper", "    goto_pose([0, 0, 0])\n    open_gripper()\n    return {'success': True}\n"), available_functions=AVAIL)
    extractor = FakeExtractor()
    c = StepGrowthController(StepGrowthConfig(), output_dir=tmp_path / "run", library_getter=lambda: lib, extractor=extractor)
    yield c, lib, extractor
    el.unregister_policy_step_listener(c.recorder.on_step_event)


def _simulate_policy(env: FakeLowLevelEnv) -> None:
    el.init_execution_context(code_block_index=0)
    with el.policy_step_context("step-1", "localize the bowl", step_index=0):
        env._sim_step_count = 100
    with el.policy_step_context("step-2", "grasp the bowl", step_index=1):
        env._gripper_fraction = 0.0
        env._sim_step_count = 300
    with el.policy_step_context("step-3", "lift", step_index=2):
        env._pick_events.append({"object": BOWL, "sim_step": 350, "z0": 0.90, "z": 0.94, "dz": 0.04})
        env._sim_step_count = 500
    with el.policy_step_context("step-4", "move over the plate", step_index=3):
        env._sim_step_count = 800
    with el.policy_step_context("step-5", "place on the plate", step_index=4):
        env._gripper_fraction = 1.0
        env._sim_step_count = 1000
    el.finalize_execution_context()


def test_end_to_end_partial_success(ctrl, tmp_path):
    c, lib, extractor = ctrl
    env = FakeLowLevelEnv(SCRIPT, goal_state=[("on", BOWL, PLATE)])
    env._pick_baseline_z = {BOWL: 0.90, PLATE: 0.88}
    iteration_data = {"iteration": 7}
    task_proposal = {"language": "put the black bowl on the plate", "activity_name": "put_black_bowl_on_plate"}
    scene_context = {"available_functions": AVAIL, "bddl_path": "/tmp/x.bddl"}

    c.on_plan_ready(plan=PLAN, task_proposal=task_proposal, scene_context=scene_context, iteration_data=iteration_data)
    c.on_attempt_start(env, iteration=7, attempt=0, attempt_in_iter=0, turn_in_attempt=0, env_reset=True)
    # the self-check dry run happens before the window opens -> ignored
    el.init_execution_context(code_block_index=0)
    with el.policy_step_context("step-1", "dry", step_index=0):
        pass
    el.finalize_execution_context()
    assert env._calls == 0
    c.on_execution_start(iteration=7, attempt=0, attempt_in_iter=0, turn_in_attempt=0, env_reset=True)
    _simulate_policy(env)
    assert env._calls == 10
    execution_result = {"artifacts": {"grounded_state": {"before": {"object_state": SCRIPT[0]}, "after": {"object_state": SCRIPT[-1]}}}}
    summary = c.on_attempt_executed(execution_result=execution_result, plan=PLAN, code=CODE, attempt=0, attempt_in_iter=0,
                                    turn_in_attempt=0, scene_context=scene_context, task_proposal=task_proposal,
                                    iteration_data=iteration_data)
    assert summary is not None
    assert summary["achieved"] == [f"grasped({BOWL})", f"lifted({BOWL})", f"near({BOWL}, {PLATE})"]
    assert summary["s_star"] == 3 and summary["fail_step"] == 4 and summary["fail_reason"] == "failure_event:dropped"
    assert summary["code_slicing"] == "step_context" and summary["sliced_steps"] == [0, 1, 2, 3, 4]
    credited = {(x["skill"], x["ok"]) for x in summary["credited"] if "skill" in x}
    assert credited == {("grasp_helper", True), ("place_helper", False)}
    assert execution_result["artifacts"]["step_oracle"]["progress"] == 0.75
    sidecar = tmp_path / "run" / "iteration_007" / "attempt_00" / "step_oracle.json"
    assert sidecar.exists()
    data = json.loads(sidecar.read_text())
    assert len(data["record"]["boundaries"]) == 10 and data["record"]["pick_events"][0]["object"] == BOWL
    assert iteration_data["step_growth"]["attempts"]["0"]["progress"] == 0.75
    assert iteration_data["step_growth"]["bddl_path"] == "/tmp/x.bddl"

    skills = {s["name"]: s for s in lib.get_full_skills_for_planner(include_deprecated=True)}
    assert skills["grasp_helper"]["step_usage_count"] == 1 and skills["grasp_helper"]["step_success_count"] == 1
    assert skills["place_helper"]["step_usage_count"] == 1 and skills["place_helper"]["step_success_count"] == 0

    out = c.on_iteration_end(success=False, iteration_data=iteration_data, scene_context=scene_context, task_proposal=task_proposal)
    assert out["attempted"] is True and out["stored_name"] == "grasp_and_lift_with_planned_pose"
    assert out["pass_steps"] == [0, 1, 2, 3]
    kw = extractor.calls[0]
    assert kw["achieved"] == [f"grasped({BOWL})", f"lifted({BOWL})", f"near({BOWL}, {PLATE})"]
    assert "grasp_helper(" in kw["prefix_code"] and "place_helper(" not in kw["prefix_code"]
    assert "[step 4]" in kw["prefix_code"] and "[step 5]" not in kw["prefix_code"]
    new = {s["name"]: s for s in lib.get_full_skills_for_planner()}["grasp_and_lift_with_planned_pose"]
    assert new["step_usage_count"] == 1 and new["strategy_tag"] == "graspnet_6dof"
    assert iteration_data["step_skills_learned"][0]["name"] == "grasp_and_lift_with_planned_pose"
    assert (tmp_path / "run" / "step_growth" / "state.json").exists()


def test_success_iteration_skips_extraction(ctrl):
    c, lib, extractor = ctrl
    iteration_data = {"iteration": 1}
    c.on_plan_ready(plan=PLAN, task_proposal={}, scene_context={}, iteration_data=iteration_data)
    out = c.on_iteration_end(success=True, iteration_data=iteration_data, scene_context={}, task_proposal={})
    assert out["attempted"] is False and out["reason"] == "iteration_succeeded" and not extractor.calls


def test_hooks_are_exception_safe(ctrl):
    c, _, _ = ctrl
    # nothing bound, garbage inputs: must not raise
    c.on_attempt_start(None, iteration=1, attempt=0, attempt_in_iter=0, turn_in_attempt=0, env_reset=True)
    c.on_execution_start(iteration=1, attempt=0, attempt_in_iter=0, turn_in_attempt=0, env_reset=True)
    out = c.on_attempt_executed(execution_result=None, plan=None, code="", attempt=0, attempt_in_iter=0,
                                turn_in_attempt=0, scene_context={}, task_proposal={}, iteration_data={"iteration": 1})
    # no env bound -> empty record, no markers, no credit; and no exception
    assert out is None or (out["markers_seen"] is False and out["credited"] == [])


def test_maybe_create_gates_on_env_and_flag(tmp_path, monkeypatch):
    monkeypatch.delenv("RATS_STEP_GROWTH", raising=False)
    assert StepGrowthController.maybe_create(output_dir=tmp_path, env_type="libero", library_getter=lambda: None) is None
    monkeypatch.setenv("RATS_STEP_GROWTH", "1")
    assert StepGrowthController.maybe_create(output_dir=tmp_path, env_type="molmospaces", library_getter=lambda: None) is None
    c = StepGrowthController.maybe_create(output_dir=tmp_path, env_type="libero", library_getter=lambda: None)
    assert c is not None
    el.unregister_policy_step_listener(c.recorder.on_step_event)
