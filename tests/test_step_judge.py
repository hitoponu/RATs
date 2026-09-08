from __future__ import annotations

from conftest import boundary, make_record, snap

from rats.step_growth import milestones as ms
from rats.step_growth.config import StepGrowthConfig
from rats.step_growth.step_judge import judge, step_spans

BOWL, PLATE = "akita_black_bowl_1", "plate_1"
ON = f"[on {BOWL} {PLATE}]"
GOAL = [("on", BOWL, PLATE)]
CFG = StepGrowthConfig()

BASE = {BOWL: (0.0, 0.0, 0.90), PLATE: (0.30, 0.0, 0.88)}
LIFTED = {BOWL: (0.0, 0.0, 0.95), PLATE: (0.30, 0.0, 0.88)}
OVER = {BOWL: (0.30, 0.0, 0.95), PLATE: (0.30, 0.0, 0.88)}
ON_PLATE = {BOWL: (0.30, 0.0, 0.89), PLATE: (0.30, 0.0, 0.88)}
FELL = {BOWL: (0.30, 0.18, 0.88), PLATE: (0.30, 0.0, 0.88)}
PICK = [{"object": BOWL, "sim_step": 350, "z0": 0.90, "z": 0.94, "dz": 0.04}]
BASELINE = {BOWL: 0.90, PLATE: 0.88}
PLAN = [{"id": f"step-{i + 1}", "description": d} for i, d in enumerate(
    ["localize", "grasp", "lift", "move", "place", "retreat", "never runs"])]


def _verdict_map(v):
    return {s.step_index: s.verdict for s in v.steps}


def _run(boundaries, pick=PICK, plan=PLAN, final=None):
    rec = make_record(boundaries, pick_events=pick, baseline_z=BASELINE, final_sim_step=final)
    res = ms.evaluate(GOAL, rec, CFG)
    return judge(res, rec, plan, CFG)


def test_grasp_lift_then_drop_at_place():
    b = [
        boundary(0, "begin", 0, 0, snap(BASE, goal={ON: False})),
        boundary(1, "end", 0, 100, snap(BASE, goal={ON: False})),
        boundary(2, "begin", 1, 100, snap(BASE, goal={ON: False})),
        boundary(3, "end", 1, 300, snap(BASE, contact=[BOWL]), gripper=0.0),
        boundary(4, "begin", 2, 300, snap(BASE, contact=[BOWL]), gripper=0.0),
        boundary(5, "end", 2, 500, snap(LIFTED, contact=[BOWL]), gripper=0.0),
        boundary(6, "begin", 3, 500, snap(LIFTED, contact=[BOWL]), gripper=0.0),
        boundary(7, "end", 3, 800, snap(OVER, contact=[BOWL]), gripper=0.0),
        boundary(8, "begin", 4, 800, snap(OVER, contact=[BOWL]), gripper=0.0),
        boundary(9, "end", 4, 1000, snap(FELL, goal={ON: False}), gripper=1.0),
        boundary(10, "begin", 5, 1000, snap(FELL, goal={ON: False}), gripper=1.0),
        boundary(11, "end", 5, 1100, snap(FELL, goal={ON: False}), gripper=1.0),
    ]
    v = _run(b)
    assert v.t_star == 800 and v.s_star == 3
    assert v.fail_step == 4 and v.fail_reason == "failure_event:dropped"
    assert _verdict_map(v) == {0: "pass", 1: "pass", 2: "pass", 3: "pass", 4: "fail", 5: "unjudged", 6: "not_executed"}
    assert v.progress == 0.75 and v.pass_steps == [0, 1, 2, 3]


def test_drop_inside_s_star_fails_s_star():
    """Second turn re-enters the place step: near was reached in turn 0 (S*=3),
    the drop happens inside the same step in turn 1 -> S* itself fails."""
    b = [
        boundary(0, "begin", 0, 0, snap(BASE)),
        boundary(1, "end", 0, 100, snap(BASE)),
        boundary(2, "begin", 1, 100, snap(BASE)),
        boundary(3, "end", 1, 300, snap(BASE, contact=[BOWL]), gripper=0.0),
        boundary(4, "begin", 2, 300, snap(BASE, contact=[BOWL]), gripper=0.0),
        boundary(5, "end", 2, 500, snap(LIFTED, contact=[BOWL]), gripper=0.0),
        boundary(6, "begin", 3, 500, snap(LIFTED, contact=[BOWL]), gripper=0.0),
        boundary(7, "end", 3, 800, snap(OVER, contact=[BOWL]), gripper=0.0),
        # turn 1: step 3 runs again, ends with the bowl dropped off the plate
        boundary(8, "begin", 3, 800, snap(OVER, contact=[BOWL]), gripper=0.0, turn=1),
        boundary(9, "end", 3, 1200, snap(FELL, goal={ON: False}), gripper=1.0, turn=1),
        boundary(10, "begin", 4, 1200, snap(FELL), gripper=1.0, turn=1),
        boundary(11, "end", 4, 1300, snap(FELL), gripper=1.0, turn=1),
    ]
    v = _run(b)
    assert v.s_star == 3 and v.fail_step == 3 and v.fail_reason == "failure_event:dropped"
    assert _verdict_map(v)[3] == "fail" and _verdict_map(v)[4] == "unjudged"
    assert v.pass_steps == [0, 1, 2]


def test_verify_step_between_lift_and_place_passes():
    b = [
        boundary(0, "begin", 0, 0, snap(BASE)),
        boundary(1, "end", 0, 100, snap(BASE)),
        boundary(2, "begin", 1, 100, snap(BASE)),
        boundary(3, "end", 1, 300, snap(BASE, contact=[BOWL]), gripper=0.0),
        boundary(4, "begin", 2, 300, snap(BASE, contact=[BOWL]), gripper=0.0),
        boundary(5, "end", 2, 500, snap(LIFTED, contact=[BOWL]), gripper=0.0),
        boundary(6, "begin", 3, 500, snap(LIFTED, contact=[BOWL]), gripper=0.0),   # verify (no motion)
        boundary(7, "end", 3, 520, snap(LIFTED, contact=[BOWL]), gripper=0.0),
        boundary(8, "begin", 4, 520, snap(LIFTED, contact=[BOWL]), gripper=0.0),   # place: drops away
        boundary(9, "end", 4, 1000, snap(FELL, goal={ON: False}), gripper=1.0),
    ]
    v = _run(b)
    assert v.s_star == 2 and v.fail_step == 4 and v.fail_reason == "failure_event:dropped"
    assert _verdict_map(v)[3] == "pass"


def test_place_then_knock_over_in_retreat():
    b = [
        boundary(0, "begin", 1, 100, snap(BASE)),
        boundary(1, "end", 1, 300, snap(BASE, contact=[BOWL]), gripper=0.0),
        boundary(2, "begin", 2, 300, snap(BASE, contact=[BOWL]), gripper=0.0),
        boundary(3, "end", 2, 500, snap(LIFTED, contact=[BOWL]), gripper=0.0),
        boundary(4, "begin", 3, 500, snap(LIFTED, contact=[BOWL]), gripper=0.0),
        boundary(5, "end", 3, 800, snap(OVER, contact=[BOWL]), gripper=0.0),
        boundary(6, "begin", 4, 800, snap(OVER, contact=[BOWL]), gripper=0.0),
        boundary(7, "end", 4, 1000, snap(ON_PLATE, goal={ON: True}), gripper=1.0),
        boundary(8, "begin", 5, 1000, snap(ON_PLATE, goal={ON: True}), gripper=1.0),
        boundary(9, "end", 5, 1100, snap(FELL, goal={ON: False}), gripper=1.0),
    ]
    v = _run(b)
    assert v.progress == 1.0 and v.s_star == 4
    assert v.fail_step == 5 and v.fail_reason == "next_after_milestone"
    assert _verdict_map(v)[4] == "pass" and _verdict_map(v)[5] == "fail"


def test_wrong_object_fails_first_step():
    other = {**BASE, "akita_black_bowl_2": (0.1, 0.1, 0.90)}
    b = [
        boundary(0, "begin", 0, 0, snap(other)),
        boundary(1, "end", 0, 100, snap(other)),
        boundary(2, "begin", 1, 100, snap(other)),
        boundary(3, "end", 1, 300, snap(other, contact=["akita_black_bowl_2"]), gripper=0.0),
        boundary(4, "begin", 2, 300, snap(other, contact=["akita_black_bowl_2"]), gripper=0.0),
        boundary(5, "end", 2, 500, snap(other, contact=["akita_black_bowl_2"]), gripper=0.0),
    ]
    wrong_pick = [{"object": "akita_black_bowl_2", "sim_step": 350, "z0": 0.9, "z": 0.95, "dz": 0.05}]
    v = _run(b, pick=wrong_pick)
    assert v.progress == 0.0 and v.t_star is None
    assert v.fail_step == 0 and v.fail_reason == "no_milestone_first_step"
    assert _verdict_map(v) == {0: "fail", 1: "unjudged", 2: "unjudged", 3: "not_executed", 4: "not_executed", 5: "not_executed", 6: "not_executed"}


def test_no_failure_event_blames_next_step_and_open_span():
    """Crash inside step 3 (no end boundary): S*=2 (lift), next step fails."""
    b = [
        boundary(0, "begin", 1, 100, snap(BASE)),
        boundary(1, "end", 1, 300, snap(BASE, contact=[BOWL]), gripper=0.0),
        boundary(2, "begin", 2, 300, snap(BASE, contact=[BOWL]), gripper=0.0),
        boundary(3, "end", 2, 500, snap(LIFTED, contact=[BOWL]), gripper=0.0),
        boundary(4, "begin", 3, 500, snap(LIFTED, contact=[BOWL]), gripper=0.0),
        boundary(5, "end", 3, 700, snap(LIFTED, contact=[BOWL]), gripper=0.0, exc=True),
    ]
    v = _run(b, final=700)
    assert v.s_star == 2 and v.fail_step == 3 and v.fail_reason == "next_after_milestone"
    assert v.steps[2].exc_in_flight is True
    assert _verdict_map(v)[3] == "fail"


def test_last_step_reaches_milestone_nothing_fails():
    b = [
        boundary(0, "begin", 1, 100, snap(BASE)),
        boundary(1, "end", 1, 300, snap(BASE, contact=[BOWL]), gripper=0.0),
        boundary(2, "begin", 2, 300, snap(BASE, contact=[BOWL]), gripper=0.0),
        boundary(3, "end", 2, 500, snap(LIFTED, contact=[BOWL]), gripper=0.0),
    ]
    v = _run(b)
    assert v.s_star == 2 and v.fail_step is None
    assert v.pass_steps == [1, 2]


def test_no_markers_gives_no_step_verdicts():
    rec = make_record([], pick_events=PICK, baseline_z=BASELINE)
    res = ms.evaluate(GOAL, rec, CFG)
    v = judge(res, rec, PLAN, CFG)
    assert v.markers_seen is False and v.fail_reason == "no_markers"
    assert all(s.verdict == "not_executed" for s in v.steps)
    assert v.progress == 0.5  # grasped + lifted from pick_events alone


def test_step_spans_group_by_index_and_order():
    b = [
        boundary(0, "begin", 2, 0, None), boundary(1, "end", 2, 10, None),
        boundary(2, "begin", 0, 10, None), boundary(3, "end", 0, 20, None),
    ]
    spans = step_spans(make_record(b))
    assert [s.step_index for s in spans] == [2, 0]
    assert spans[0].begin_sim_step == 0 and spans[0].end_sim_step == 10
