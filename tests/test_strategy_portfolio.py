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


def test_exhausted_ban_list_recycles_instead_of_dropping_the_type(tmp_path):
    """Regression (smoke 5258527): every family banned -> the phase vanished.

    open_close has 3 families and turn has 2, so a few mid-iteration switches
    banned them all; `select` then returned no family for that type. The two
    milestones that run DID reach (open, turnon) landed on attempts with no
    assignment, so no family was ever paid and the bandit stayed at 0 rewards.
    """
    p = _portfolio(tmp_path)
    prev = Selection(iteration=1, attempt=4, families={"grasp": "C"}, step_types=["grasp"],
                     banned={"grasp": ["A", "B"]})
    sel = p.select_for_retry(prev, misses={"grasp": 2},
                             retry_feedback={"failure_mode": "grasp_failure"},
                             diagnosis=None, goal_state=GOAL, available_functions=AVAIL,
                             iteration=1, attempt=5, retry_switch_after=2)
    assert "grasp" in sel.families, sel.reasons        # the phase still gets a directive
    assert sel.families["grasp"] in {"A", "B"}         # C was just banned, so not C
    assert "recycled" in sel.reasons["grasp"]
    assert sel.banned["grasp"] == ["C"]                # ban list reset to the last one
    assert "GRASP —" in p.render(sel)


def test_recycling_keeps_the_reward_attributable(tmp_path):
    """After recycling, the achieving attempt still has a family to pay."""
    p = _portfolio(tmp_path)
    prev = Selection(iteration=1, attempt=4, families={"grasp": "C"}, step_types=["grasp"],
                     banned={"grasp": ["A", "B"]})
    sel = p.select_for_retry(prev, misses={"grasp": 2},
                             retry_feedback={"failure_mode": "grasp_failure"},
                             diagnosis=None, goal_state=GOAL, available_functions=AVAIL,
                             iteration=1, attempt=5, retry_switch_after=2)
    events = p.update(sel, ["grasped(bowl_1)", "lifted(bowl_1)"])
    paid = {e["step_type"]: e["reward"] for e in events}
    assert paid["grasp"] == 1 and paid["place"] == 0
    assert p.stats_snapshot()[sel.families["grasp"]]["rewards"] == 1


def test_single_family_type_is_never_dropped(tmp_path):
    """A type with one family: banning it must not silence the phase."""
    bank = StrategyBank([Family("ONLY", ("turn",), "turn it", (), (), ())], [], source="t")
    p = StrategyPortfolio(bank, tmp_path / "s.json", rng_seed=0)
    prev = Selection(iteration=1, attempt=0, families={"turn": "ONLY"}, step_types=["turn"])
    sel = p.select_for_retry(prev, misses={"turn": 3},
                             retry_feedback={"failure_mode": "wrong_affordance"},
                             diagnosis=None, goal_state=[("turnon", "stove_1")],
                             available_functions=AVAIL, iteration=1, attempt=1)
    assert sel.families["turn"] == "ONLY" and "recycled" in sel.reasons["turn"]


REDUCED_API = [
    "get_observation", "segment_sam3_text_prompt", "segment_sam3_point_prompt",
    "point_prompt_molmo", "plan_grasp", "plan_grasp_from_point_clouds",
    "get_oriented_bounding_box_from_3d_points", "solve_ik", "move_to_joints",
    "open_gripper", "close_gripper", "goto_pose", "goto_home_joint_position",
    "subsample_point_cloud", "filter_noise",
]


def test_shipped_bank_scopes_families_by_goal_predicate():
    """Regression (smoke 5258924): `(on milk plate)` was handed the
    put-it-inside-the-rim recipe, and `(open drawer)` the push-it-shut one."""
    bank = StrategyBank.load()
    place_on = {f.id for f in bank.for_type("place", REDUCED_API, {"on"})}
    place_in = {f.id for f in bank.for_type("place", REDUCED_API, {"in"})}
    assert "obb_inside_container" not in place_on and "hover_descend_release" in place_on
    assert "obb_inside_container" in place_in and "hover_descend_release" not in place_in

    open_fams = {f.id for f in bank.for_type("open_close", REDUCED_API, {"open"})}
    close_fams = {f.id for f in bank.for_type("open_close", REDUCED_API, {"close"})}
    assert "push_surface_closed_gripper" not in open_fams
    assert "push_surface_closed_gripper" in close_fams and "hook_edge_pull" not in close_fams
    # every type still has at least two families to choose between
    for fams in (place_on, place_in, open_fams, close_fams):
        assert len(fams) >= 2, fams


def test_yaml_boolean_predicates_are_read_back_as_predicates():
    """`predicates: [on]` is a BOOLEAN in YAML 1.1 — the loader maps it back."""
    bank = StrategyBank.load()
    hover = bank.get("hover_descend_release")
    assert hover is not None and hover.predicates == ("on",)
    assert hover.applies_to({"on"}) and not hover.applies_to({"in"})


def test_predicate_filter_never_empties_a_phase(tmp_path):
    bank = StrategyBank([Family("ONLY", ("place",), "put it down", (), (), (), ("in",))], [], source="t")
    p = StrategyPortfolio(bank, tmp_path / "s.json", rng_seed=0)
    sel = p.select(goal_state=[("on", "milk_1", "plate_1")], available_functions=AVAIL, iteration=0)
    assert sel.families["place"] == "ONLY"          # mismatched beats silent


def test_selection_records_the_predicates_it_used(tmp_path):
    p = _portfolio(tmp_path)
    sel = p.select(goal_state=[("in", "cookies_1", "cabinet_1")], available_functions=AVAIL, iteration=0)
    assert sel.as_dict()["predicates"] == {"grasp": ["in"], "place": ["in"]}


def test_directive_binds_the_strategy_but_defers_the_details(tmp_path):
    """The block used to claim it outranked the diagnoser too; that is the
    signal for repairing THIS attempt and it now wins on everything but the
    choice of strategy."""
    p = _portfolio(tmp_path)
    sel = p.select(goal_state=GOAL, available_functions=AVAIL, iteration=0)
    text = p.render(sel)
    assert text.startswith("## STRATEGY DIRECTIVE (HARD")     # launcher greps this
    assert "RETRY CONTEXT" in text and "Keep the strategy, fix the details." in text
    assert "overrides all other guidance" not in text
    assert "the diagnoser tells you to avoid" not in text


def test_directive_pins_the_program_shape(tmp_path):
    """Regression (smoke 5259311): 9 of 18 programs defined main() and never
    called it — they executed, reported success and moved nothing."""
    p = _portfolio(tmp_path)
    text = p.render(p.select(goal_state=GOAL, available_functions=AVAIL, iteration=0))
    assert "STRUCTURE:" in text
    assert "RESULT = main()" in text and "step_context" in text
