"""Strategy bandit: forced exploration, preference, switching, persistence."""

from __future__ import annotations

import json

import pytest

from rats.step_growth.strategy_portfolio import (
    Family,
    Selection,
    StrategyBank,
    StrategyPortfolio,
    step_types_for_goal,
)

AVAIL = ["get_observation", "plan_grasp", "goto_pose", "close_gripper", "open_gripper",
         "segment_sam3_text_prompt", "get_oriented_bounding_box_from_3d_points"]
GOAL = [("on", "bowl_1", "plate_1")]


def _bank() -> StrategyBank:
    return StrategyBank(
        [
            Family("A", ("grasp",), "use A", ("plan_grasp",), (), ("plan_grasp",)),
            Family("B", ("grasp",), "use B", ("get_oriented_bounding_box_from_3d_points",), (), ("obb_yaw",)),
            Family("C", ("grasp",), "use C", ("goto_pose",), ("never do C-things",), ("other",)),
            Family("M", ("grasp",), "use Molmo", ("point_prompt_molmo",), (), ("molmo",)),
            Family("P", ("place",), "place carefully", ("open_gripper",), (), ("place",)),
        ],
        ["never hard-code a quaternion"],
        source="test-bank",
    )


def _portfolio(tmp_path, **kw) -> StrategyPortfolio:
    kw.setdefault("rng_seed", 0)
    kw.setdefault("min_pulls", 2)
    kw.setdefault("window_iters", 10)
    return StrategyPortfolio(_bank(), tmp_path / "strategy_state.json", **kw)


def test_step_types_come_from_the_goal_not_the_plan():
    assert step_types_for_goal(GOAL) == ["grasp", "place"]
    assert step_types_for_goal(["[open microwave_1]"]) == ["open_close"]
    assert step_types_for_goal([("turnon", "stove_1")]) == ["turn"]
    assert step_types_for_goal([("unknown_predicate", "x")]) == []


def test_requires_filters_unavailable_families(tmp_path):
    """M needs point_prompt_molmo, which the reduced API of this run lacks."""
    p = _portfolio(tmp_path)
    picked = set()
    for it in range(12):
        sel = p.select(goal_state=GOAL, available_functions=AVAIL, iteration=it)
        picked.add(sel.families["grasp"])
        p.update(sel, [])
    assert "M" not in picked
    assert picked == {"A", "B", "C"}


def test_forced_exploration_covers_every_family_first(tmp_path):
    p = _portfolio(tmp_path)
    for it in range(6):
        sel = p.select(goal_state=GOAL, available_functions=AVAIL, iteration=it)
        assert sel.reasons["grasp"] == "forced_exploration"
        p.update(sel, [])
    pulls = {k: v["pulls"] for k, v in p.stats_snapshot().items() if k in {"A", "B", "C"}}
    assert pulls == {"A": 2, "B": 2, "C": 2}
    # min_pulls satisfied inside the window -> the bandit takes over
    sel = p.select(goal_state=GOAL, available_functions=AVAIL, iteration=6)
    assert sel.reasons["grasp"] == "thompson"


def test_rewards_steer_the_choice(tmp_path):
    p = _portfolio(tmp_path)
    for _ in range(5):
        p.update(Selection(iteration=100, attempt=0, families={"grasp": "A"},
                           step_types=["grasp"]), ["grasped(bowl_1)", "lifted(bowl_1)"])
        for loser in ("B", "C"):
            p.update(Selection(iteration=100, attempt=0, families={"grasp": loser},
                               step_types=["grasp"]), ["grasped(bowl_1)"])
    chosen = [p.select(goal_state=GOAL, available_functions=AVAIL, iteration=100).families["grasp"]
              for _ in range(10)]
    assert chosen.count("A") >= 8, chosen


def test_reward_gating_and_attribution(tmp_path):
    """grasp is paid by `lifted`; place is not judged at all until it lifts."""
    p = _portfolio(tmp_path)
    sel = Selection(iteration=1, attempt=0, families={"grasp": "A", "place": "P"},
                    step_types=["grasp", "place"])
    assert p.rewards_for(sel, ["grasped(bowl_1)"]) == {"grasp": 0}
    assert p.rewards_for(sel, ["grasped(bowl_1)", "lifted(bowl_1)"]) == {"grasp": 1, "place": 0}
    assert p.rewards_for(
        sel, ["grasped(bowl_1)", "lifted(bowl_1)", "near(bowl_1, plate_1)", "placed(bowl_1, plate_1)"],
    ) == {"grasp": 1, "place": 1}


def test_retry_switches_only_on_a_repeated_physical_miss(tmp_path):
    p = _portfolio(tmp_path)
    prev = Selection(iteration=3, attempt=0, families={"grasp": "A", "place": "P"},
                     step_types=["grasp", "place"])
    common = dict(goal_state=GOAL, available_functions=AVAIL, iteration=3, attempt=1,
                  diagnosis=None, retry_switch_after=2)

    one_miss = p.select_for_retry(prev, misses={"grasp": 1},
                                  retry_feedback={"failure_mode": "grasp_failure"}, **common)
    assert one_miss.families["grasp"] == "A" and not one_miss.notes

    code_bug = p.select_for_retry(prev, misses={"grasp": 2},
                                  retry_feedback={"failure_mode": "code_bug",
                                                  "edit_scale": "rewrite_needed"}, **common)
    assert code_bug.families["grasp"] == "A" and not code_bug.notes

    switched = p.select_for_retry(prev, misses={"grasp": 2},
                                  retry_feedback={"failure_mode": "grasp_failure"}, **common)
    assert switched.families["grasp"] != "A"
    assert switched.reasons["grasp"] == "switched"
    assert switched.families["place"] == "P"          # untouched type keeps its family
    assert any("FORBIDDEN" in n and "'A'" in n for n in switched.notes)
    assert "A" in switched.banned["grasp"]
    assert "FORBIDDEN" in p.render(switched)

    # argument_level alone is also a physical signal (the diagnoser wants the
    # same code with different numbers -- and it has already failed twice).
    arg_level = p.select_for_retry(prev, misses={"grasp": 2},
                                   retry_feedback={"edit_scale": "argument_level"}, **common)
    assert arg_level.families["grasp"] != "A"


def test_state_round_trip(tmp_path):
    p = _portfolio(tmp_path)
    for it in range(4):
        sel = p.select(goal_state=GOAL, available_functions=AVAIL, iteration=it)
        p.update(sel, ["grasped(b)", "lifted(b)"])
    p.save()
    before = p.stats_snapshot()
    again = _portfolio(tmp_path)
    assert again.stats_snapshot() == before
    data = json.loads((tmp_path / "strategy_state.json").read_text())
    assert data["schema"] == "rats_strategy_portfolio_v1"


def test_collapse_override_excludes_the_degenerate_recipe(tmp_path):
    p = _portfolio(tmp_path, collapse_k=3)
    fp = {"family": "obb_yaw", "identity_quat": True, "uses_plan_grasp": False, "ast_hash": "h"}
    assert p.note_fingerprint(0, 0, fp) is None
    assert p.note_fingerprint(1, 0, fp) is None
    collapse = p.note_fingerprint(2, 0, fp)
    assert collapse and collapse["family"] == "obb_yaw"
    # non-first attempts never feed the rule
    assert p.note_fingerprint(3, 1, {"family": "graspnet", "identity_quat": False}) is collapse
    sel = p.select(goal_state=GOAL, available_functions=AVAIL, iteration=3)
    assert sel.families["grasp"] != "B"               # B is the obb_yaw family
    assert "collapse_override" in sel.reasons["grasp"]
    assert any(n.startswith("COLLAPSE:") for n in sel.notes)
    assert "COLLAPSE:" in p.render(sel)


def test_collapse_needs_one_family_and_a_hard_coded_quat(tmp_path):
    p = _portfolio(tmp_path, collapse_k=3)
    same_family_no_quat = {"family": "obb_yaw", "identity_quat": False}
    for it in range(3):
        assert p.note_fingerprint(it, 0, same_family_no_quat) is None
    mixed = [{"family": "obb_yaw", "identity_quat": True}, {"family": "graspnet", "identity_quat": True}]
    for it, fp in enumerate(mixed * 2):
        assert p.note_fingerprint(10 + it, 0, fp) is None


def test_render_is_empty_without_a_selection(tmp_path):
    p = _portfolio(tmp_path)
    assert p.render(Selection(iteration=0, attempt=0)) == ""
    sel = p.select(goal_state=GOAL, available_functions=AVAIL, iteration=0)
    text = p.render(sel)
    assert text.startswith("## STRATEGY DIRECTIVE (HARD")
    assert "GRASP —" in text and "PLACE —" in text
    assert "never hard-code a quaternion" in text        # common forbid
    assert "Evidence" not in text                        # oracle stats stay out by default


def test_evidence_line_is_opt_in(tmp_path):
    p = _portfolio(tmp_path, show_evidence=True)
    sel = p.select(goal_state=GOAL, available_functions=AVAIL, iteration=0)
    assert "Evidence (family successes/uses so far)" in p.render(sel)


def test_types_can_be_switched_off(tmp_path):
    p = _portfolio(tmp_path, per_type_families={"grasp": 1, "place": 0})
    sel = p.select(goal_state=GOAL, available_functions=AVAIL, iteration=0)
    assert sel.step_types == ["grasp"] and "place" not in sel.families


def test_missing_bank_file_is_not_fatal(tmp_path):
    bank = StrategyBank.load(tmp_path / "nope.yaml")
    assert bank.families == []
    p = StrategyPortfolio(bank, tmp_path / "s.json")
    sel = p.select(goal_state=GOAL, available_functions=AVAIL, iteration=0)
    assert sel.is_empty and p.render(sel) == ""


def test_shipped_bank_is_loadable_and_within_budget():
    bank = StrategyBank.load()
    assert {f.id for f in bank.families} >= {"graspnet_6dof", "obb_topdown_yaw", "hover_descend_release"}
    assert all(len(f.directive) <= 250 for f in bank.families)
    assert all(f.step_types for f in bank.families)
    ids = [f.id for f in bank.families]
    assert len(ids) == len(set(ids))
