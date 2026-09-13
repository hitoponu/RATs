from __future__ import annotations

import json

from skill_library.library import SkillLibrary, merge_skill_library_files
from skill_library.step_credit_library import StepCreditSkillLibrary, make_skill_library

AVAIL = ["segment_sam3_text_prompt", "plan_grasp", "goto_pose", "close_gripper", "open_gripper"]


def _skill(name: str) -> dict:
    return {
        "name": name,
        "description": f"{name} helper",
        "code": (
            f"def {name}(object_name: str) -> dict:\n"
            "    m = segment_sam3_text_prompt(object_name)\n"
            "    g = plan_grasp(m)\n"
            "    goto_pose(g)\n"
            "    close_gripper()\n"
            "    return {'success': True}\n"
        ),
        "api_primitives_used": ["plan_grasp"], "preconditions": [], "effects": [],
        "source_task": "t", "learned_iteration": 1,
    }


def _lib(path, **kw) -> StepCreditSkillLibrary:
    lib = StepCreditSkillLibrary(storage_path=str(path), **kw)
    assert lib.add_skill(_skill("grasp_helper"), available_functions=AVAIL)
    return lib


def test_step_credit_promotes_at_3_uses(tmp_skills_path):
    lib = _lib(tmp_skills_path)
    for _ in range(2):
        lib.record_step_usage(["grasp_helper"], True, iteration=1, attempt=0, step_id="step-2")
    s = next(x for x in lib.get_full_skills_for_planner() if x["name"] == "grasp_helper")
    assert s["tier"] == "experimental" and s["step_usage_count"] == 2
    events = lib.record_step_usage(["grasp_helper"], True, iteration=2, attempt=0, step_id="step-2")
    s = next(x for x in lib.get_full_skills_for_planner() if x["name"] == "grasp_helper")
    assert s["tier"] == "verified" and s["step_success_count"] == 3
    assert events and events[0]["action"] == "tier:experimental->verified"
    # task-level counters untouched
    assert s["usage_count"] == 0 and s["success_count"] == 0


def test_step_credit_deprecates_and_respects_dependency_guard(tmp_skills_path):
    lib = _lib(tmp_skills_path)
    wrapper = _skill("grasp_wrapper")
    wrapper["code"] = "def grasp_wrapper(object_name: str) -> dict:\n    return grasp_helper(object_name)\n"
    lib._find_semantic_duplicate = lambda *a, **k: None  # no LLM
    assert lib.add_skill(wrapper, available_functions=AVAIL)
    for _ in range(8):
        lib.record_step_usage(["grasp_helper"], False, iteration=1, attempt=0, step_id="step-2")
    s = next(x for x in lib.get_full_skills_for_planner(include_deprecated=True) if x["name"] == "grasp_helper")
    assert s["tier"] == "experimental"  # guarded: grasp_wrapper depends on it
    lib2 = _lib(tmp_skills_path.parent / "b.json")
    for _ in range(8):
        lib2.record_step_usage(["grasp_helper"], False, iteration=1, attempt=0, step_id="step-2")
    s2 = next(x for x in lib2.get_full_skills_for_planner(include_deprecated=True) if x["name"] == "grasp_helper")
    assert s2["tier"] == "deprecated"


def test_wilson_prefers_step_stats_and_orders_planner(tmp_skills_path):
    lib = _lib(tmp_skills_path)
    lib._find_semantic_duplicate = lambda *a, **k: None
    assert lib.add_skill(_skill("other_helper"), available_functions=AVAIL)
    lib.record_usage(["other_helper"], True, iteration=1)   # task-level only
    for _ in range(3):
        lib.record_step_usage(["grasp_helper"], True, iteration=1, attempt=0, step_id="s")
    names = [s["name"] for s in lib.get_full_skills_for_planner() if not s["is_primitive"]]
    assert names[0] == "grasp_helper"
    g = next(s for s in lib.get_full_skills_for_planner() if s["name"] == "grasp_helper")
    assert g["wilson_score"] > 0.2 and g["step_wilson"] == g["wilson_score"]
    task_only = StepCreditSkillLibrary(storage_path=str(tmp_skills_path), tier_policy="task")
    g2 = next(s for s in task_only.get_full_skills_for_planner() if s["name"] == "grasp_helper")
    assert g2["wilson_score"] == 0.0  # task counters are 0/0 under the task policy


def test_roundtrip_and_merge_keep_step_counters(tmp_skills_path):
    lib = _lib(tmp_skills_path)
    lib.record_step_usage(["grasp_helper"], True, iteration=1, attempt=0, step_id="s")
    raw = json.loads(tmp_skills_path.read_text())
    entry = next(s for s in raw if s["name"] == "grasp_helper")
    assert entry["step_usage_count"] == 1 and entry["step_events"][0]["ok"] is True
    plain = SkillLibrary(storage_path=str(tmp_skills_path))  # base class loads the same file
    assert plain.get_learned_skill_count() == 1
    merged = merge_skill_library_files(tmp_skills_path, [])
    m = next(s for s in merged if s["name"] == "grasp_helper")
    assert m["step_usage_count"] == 1


def test_make_skill_library_respects_env(tmp_skills_path, monkeypatch):
    monkeypatch.delenv("RATS_STEP_GROWTH", raising=False)
    assert type(make_skill_library(str(tmp_skills_path))) is SkillLibrary
    monkeypatch.setenv("RATS_STEP_GROWTH", "1")
    lib = make_skill_library(str(tmp_skills_path))
    assert isinstance(lib, StepCreditSkillLibrary) and lib.tier_policy == "step"


def test_curator_payload_carries_step_counters(tmp_skills_path):
    """The curator must see the step record, not just the task counters.

    Regression for job 5252020: the base payload is a field whitelist, so a
    skill distilled from a failed iteration reached the curator as
    usage_count=0/success_count=0 and was deprecated at step 1/1.
    """
    lib = _lib(tmp_skills_path)
    lib.record_step_usage(["grasp_helper"], True, iteration=1, attempt=0, step_id="s0")
    lib.record_step_usage(["grasp_helper"], False, iteration=2, attempt=0, step_id="s1")
    entry = next(e for e in lib.get_learned_skills_for_curator() if e["name"] == "grasp_helper")
    assert entry["step_usage_count"] == 2
    assert entry["step_success_count"] == 1
    assert entry["step_success_rate"] == 0.5
    # the task counters the base class sends are still there, untouched
    assert entry["usage_count"] == 0 and entry["success_count"] == 0
    assert "code_preview" in entry


def test_curator_payload_marks_step_extracted_skills(tmp_skills_path):
    lib = _lib(tmp_skills_path)
    for s in lib._skills:
        if s.get("name") == "grasp_helper":
            s["credit_source"] = "step_oracle"
            s["strategy_tag"] = "topdown_obb_yaw"
    entry = next(e for e in lib.get_learned_skills_for_curator() if e["name"] == "grasp_helper")
    assert entry["credit_source"] == "step_oracle"
    assert entry["strategy_tag"] == "topdown_obb_yaw"
    assert entry["step_success_rate"] is None  # no step credit recorded yet


def test_curator_payload_unchanged_for_base_library(tmp_skills_path):
    """The default arm keeps the original payload: no step keys at all."""
    _lib(tmp_skills_path)
    plain = SkillLibrary(storage_path=str(tmp_skills_path))
    entry = next(e for e in plain.get_learned_skills_for_curator() if e["name"] == "grasp_helper")
    assert "step_usage_count" not in entry and "step_success_rate" not in entry
