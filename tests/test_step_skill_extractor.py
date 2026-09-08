from __future__ import annotations

from rats.agents.step_skill_extractor import StepSkillExtractor, lint_skill_code

AVAIL = ["segment_sam3_text_prompt", "plan_grasp", "goto_pose", "close_gripper", "open_gripper", "get_observation"]

GOOD = '''def grasp_and_lift_with_planned_pose(object_name: str, lift_height: float = 0.08) -> dict:
    obs = get_observation()
    mask = segment_sam3_text_prompt(obs, object_name)
    grasp = plan_grasp(obs, mask)
    goto_pose(grasp)
    close_gripper()
    goto_pose(grasp, dz=lift_height)
    return {"success": True, "grasp": grasp}
'''

POSE_IN = '''def grasp_with_wrist_gate(grasp_pos: list, grasp_quat: list, lift_height: float = 0.08) -> dict:
    goto_pose(grasp_pos, grasp_quat)
    close_gripper()
    goto_pose([grasp_pos[0], grasp_pos[1], grasp_pos[2] + lift_height], grasp_quat)
    return {"success": True}
'''


def test_lint_accepts_boundary_compliant_skill():
    assert lint_skill_code(GOOD, available_functions=AVAIL, achieved_kinds={"grasped", "lifted"}) is None


def test_lint_rejects_pose_in_wrapper():
    reason = lint_skill_code(POSE_IN, available_functions=AVAIL, achieved_kinds={"grasped"})
    assert reason is not None and reason.startswith("first_param_not_object_name")
    shifted = POSE_IN.replace("grasp_pos: list, grasp_quat: list", "object_name: str, grasp_pos: list, grasp_quat: list")
    reason = lint_skill_code(shifted, available_functions=AVAIL, achieved_kinds={"grasped"})
    assert reason == "pose_in_wrapper:grasp_pos"


def test_lint_allows_offset_params_and_requires_perception_for_effects():
    code = GOOD.replace("lift_height: float = 0.08", "lift_height: float = 0.08, xy_offset: tuple = (0.0, 0.0)")
    assert lint_skill_code(code, available_functions=AVAIL, achieved_kinds={"grasped"}) is None
    no_perc = '''def grasp_blind(object_name: str) -> dict:
    close_gripper()
    return {"success": True}
'''
    assert lint_skill_code(no_perc, available_functions=AVAIL, achieved_kinds={"grasped"}) == "effect_skill_without_perception"


def test_lint_structural_rules():
    assert lint_skill_code("x = 1\n" + GOOD, available_functions=AVAIL, achieved_kinds=set()) == "top_level_statements_outside_function"
    assert lint_skill_code(GOOD + GOOD.replace("grasp_and_lift_with_planned_pose", "other"), available_functions=AVAIL, achieved_kinds=set()).startswith("expected_one_function")
    loop = GOOD.replace("    close_gripper()\n", "    while True:\n        close_gripper()\n")
    assert lint_skill_code(loop, available_functions=AVAIL, achieved_kinds=set()) == "while_true"
    envcall = GOOD.replace("    close_gripper()\n", "    env.step('x')\n")
    assert lint_skill_code(envcall, available_functions=AVAIL, achieved_kinds=set()) == "env_method_call"
    assert lint_skill_code(GOOD, available_functions=AVAIL, achieved_kinds=set(), max_lines=3).startswith("too_long")


def _prefix() -> str:
    return (
        "# [step 2]\nobs = get_observation()\nmask = segment_sam3_text_prompt(obs, 'black bowl')\n"
        "grasp = plan_grasp(obs, mask)\ngoto_pose(grasp)\nclose_gripper()\n# [step 3]\ngoto_pose(grasp, dz=0.08)\n"
    )


def test_extract_with_fake_llm_returns_tagged_skill():
    calls = []

    def fake(system_prompt, user_prompt, model=None):
        calls.append(user_prompt)
        return {"skill": {"name": "grasp_and_lift_with_planned_pose", "description": "grasp then lift",
                          "code": GOOD, "strategy_tag": "graspnet_6dof", "usage_example": "grasp_and_lift_with_planned_pose('black bowl')"},
                "reason": "ok"}

    ex = StepSkillExtractor(llm_query=fake)
    out = ex.extract(prefix_code=_prefix(), achieved=["grasped(akita_black_bowl_1)", "lifted(akita_black_bowl_1)"],
                     task_language="put the bowl on the plate", existing_skills=[{"name": "old_helper", "description": "x"}],
                     available_functions=AVAIL)
    assert out["skill"] is not None and out["rejected_reason"] is None
    assert out["skill"]["credit_source"] == "step_oracle" and out["skill"]["strategy_tag"] == "graspnet_6dof"
    assert "grasped(akita_black_bowl_1), lifted(akita_black_bowl_1)" in calls[0]
    assert "old_helper" in calls[0] and "put the bowl on the plate" in calls[0]
    assert "{code}" not in calls[0] and "{max_lines}" not in calls[0]


def test_extract_rejects_null_and_bad_candidates():
    ex_null = StepSkillExtractor(llm_query=lambda s, u, model=None: {"skill": None, "reason": "dup"})
    out = ex_null.extract(prefix_code=_prefix(), achieved=["grasped(x)"], task_language="t", existing_skills=[], available_functions=AVAIL)
    assert out["skill"] is None and out["rejected_reason"] == "llm_returned_null" and out["llm_reason"] == "dup"
    ex_bad = StepSkillExtractor(llm_query=lambda s, u, model=None: {"skill": {"name": "w", "code": POSE_IN}})
    out = ex_bad.extract(prefix_code=_prefix(), achieved=["grasped(x)"], task_language="t", existing_skills=[], available_functions=AVAIL)
    assert out["skill"] is None and out["rejected_reason"].startswith("first_param_not_object_name")


def test_extract_prefix_gate_skips_llm():
    called = []
    ex = StepSkillExtractor(llm_query=lambda *a, **k: called.append(1) or {})
    out = ex.extract(prefix_code="x = 1\n", achieved=["grasped(x)"], task_language="t", existing_skills=[], available_functions=AVAIL)
    assert out["rejected_reason"].startswith("prefix_too_short") and not called and out["llm_called"] is False
