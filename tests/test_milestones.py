from __future__ import annotations

from conftest import boundary, make_record, snap

from rats.step_growth import milestones as ms
from rats.step_growth.config import StepGrowthConfig

BOWL, PLATE = "akita_black_bowl_1", "plate_1"
ON = f"[on {BOWL} {PLATE}]"


def test_parse_goal_state_forms():
    assert ms.parse_goal_state([("On", BOWL, PLATE)]) == [("on", BOWL, PLATE)]
    assert ms.parse_goal_state(["[on akita_black_bowl_1 plate_1]"]) == [("on", BOWL, PLATE)]
    assert ms.parse_goal_state([{"predicate": "[open wooden_cabinet_1_top_region]", "satisfied": False}]) == [
        ("open", "wooden_cabinet_1_top_region")
    ]
    assert ms.parse_goal_state([["And", ["turnon", "flat_stove_1"], ["on", "pot_1", "r"]]]) == [
        ("turnon", "flat_stove_1"), ("on", "pot_1", "r"),
    ]


def test_derive_chain_for_on_and_open():
    chain = ms.derive_milestones([("on", BOWL, PLATE)])
    assert [m.kind for m in chain] == ["grasped", "lifted", "near", "placed"]
    assert chain[0].args == (BOWL,) and chain[2].args == (BOWL, PLATE)
    assert chain[3].predicate == ON
    single = ms.derive_milestones([("open", "wooden_cabinet_1_top_region")])
    assert [m.kind for m in single] == ["open"]
    assert ms.derive_milestones([("up", BOWL)]) == []  # unknown predicate skipped


def _pick_place_record(*, place_ok: bool):
    base = {BOWL: (0.0, 0.0, 0.90), PLATE: (0.30, 0.0, 0.88)}
    lifted = {BOWL: (0.0, 0.0, 0.95), PLATE: (0.30, 0.0, 0.88)}
    over = {BOWL: (0.30, 0.0, 0.95), PLATE: (0.30, 0.0, 0.88)}
    end_pos = {BOWL: (0.30, 0.0, 0.89), PLATE: (0.30, 0.0, 0.88)} if place_ok else {BOWL: (0.30, 0.18, 0.88), PLATE: (0.30, 0.0, 0.88)}
    b = [
        boundary(0, "begin", 0, 0, snap(base, goal={ON: False})),
        boundary(1, "end", 0, 100, snap(base, goal={ON: False})),
        boundary(2, "begin", 1, 100, snap(base, goal={ON: False})),
        boundary(3, "end", 1, 300, snap(base, contact=[BOWL], goal={ON: False}), gripper=0.0),
        boundary(4, "begin", 2, 300, snap(base, contact=[BOWL], goal={ON: False}), gripper=0.0),
        boundary(5, "end", 2, 500, snap(lifted, contact=[BOWL], goal={ON: False}), gripper=0.0),
        boundary(6, "begin", 3, 500, snap(lifted, contact=[BOWL], goal={ON: False}), gripper=0.0),
        boundary(7, "end", 3, 800, snap(over, contact=[BOWL], goal={ON: False}), gripper=0.0),
        boundary(8, "begin", 4, 800, snap(over, contact=[BOWL], goal={ON: False}), gripper=0.0),
        boundary(9, "end", 4, 1000, snap(end_pos, goal={ON: place_ok}, relations=[("on", BOWL, PLATE)] if place_ok else []), gripper=1.0),
    ]
    return make_record(
        b, pick_events=[{"object": BOWL, "sim_step": 350, "z0": 0.90, "z": 0.94, "dz": 0.04}],
        baseline_z={BOWL: 0.90, PLATE: 0.88},
    )


def test_first_true_times_success_chain():
    rec = _pick_place_record(place_ok=True)
    res = ms.evaluate([("on", BOWL, PLATE)], rec, StepGrowthConfig())
    times = {m.kind: res.times[m.key] for m in res.milestones}
    assert times == {"grasped": 350, "lifted": 350, "near": 800, "placed": 1000}
    assert res.progress == 1.0
    assert res.events == []


def test_dropped_event_when_place_fails():
    rec = _pick_place_record(place_ok=False)
    res = ms.evaluate([("on", BOWL, PLATE)], rec, StepGrowthConfig())
    times = {m.kind: res.times[m.key] for m in res.milestones}
    assert times["near"] == 800 and times["placed"] is None
    assert res.progress == 0.75
    assert res.events and res.events[0]["event"] == "dropped"
    assert res.events[0]["sim_step"] == 1000 and res.events[0]["step_index"] == 4


def test_near_cannot_precede_lift():
    """An object that starts within the near radius must not count as near before lift."""
    base = {BOWL: (0.32, 0.0, 0.90), PLATE: (0.30, 0.0, 0.88)}
    b = [
        boundary(0, "begin", 0, 0, snap(base, goal={ON: False})),
        boundary(1, "end", 0, 100, snap(base, goal={ON: False})),
    ]
    rec = make_record(b, baseline_z={BOWL: 0.90})
    res = ms.evaluate([("on", BOWL, PLATE)], rec, StepGrowthConfig())
    assert all(v is None for v in res.times.values())
    assert res.progress == 0.0


def test_straddle_is_not_a_grasp():
    base = {BOWL: (0.0, 0.0, 0.90)}
    b = [
        boundary(0, "begin", 1, 100, snap(base, goal={ON: False})),
        # contact but gripper open and no lift -> not grasped
        boundary(1, "end", 1, 300, snap(base, contact=[BOWL], goal={ON: False}), gripper=1.0),
    ]
    rec = make_record(b, baseline_z={BOWL: 0.90})
    res = ms.evaluate([("on", BOWL, PLATE)], rec, StepGrowthConfig())
    assert res.times[res.milestones[0].key] is None


def test_grasp_from_boundary_without_pick_event():
    base = {BOWL: (0.0, 0.0, 0.90)}
    up = {BOWL: (0.0, 0.0, 0.91)}
    b = [
        boundary(0, "begin", 1, 100, snap(base)),
        boundary(1, "end", 1, 300, snap(up, contact=[BOWL]), gripper=0.0),
    ]
    rec = make_record(b, baseline_z={BOWL: 0.90})
    res = ms.evaluate([("on", BOWL, PLATE)], rec, StepGrowthConfig())
    assert res.times[res.milestones[0].key] == 300   # grasped (5 mm + closed + contact)
    assert res.times[res.milestones[1].key] is None  # lifted needs 3 cm


def test_slipped_event():
    base = {BOWL: (0.0, 0.0, 0.90)}
    b = [
        boundary(0, "begin", 1, 100, snap(base)),
        boundary(1, "end", 1, 300, snap(base, contact=[BOWL]), gripper=0.0),  # closed on it, no lift
        boundary(2, "begin", 2, 300, snap(base, contact=[BOWL]), gripper=0.0),
        boundary(3, "end", 2, 500, snap(base), gripper=0.0),                   # contact lost
    ]
    rec = make_record(b, baseline_z={BOWL: 0.90})
    res = ms.evaluate([("on", BOWL, PLATE)], rec, StepGrowthConfig())
    assert res.events and res.events[0]["event"] == "slipped" and res.events[0]["step_index"] == 2


def test_open_goal_uses_goal_bit():
    pred = "[open wooden_cabinet_1_top_region]"
    b = [
        boundary(0, "begin", 0, 0, snap({}, goal={pred: False})),
        boundary(1, "end", 0, 200, snap({}, goal={pred: True})),
    ]
    rec = make_record(b)
    res = ms.evaluate([("open", "wooden_cabinet_1_top_region")], rec, StepGrowthConfig())
    assert res.progress == 1.0 and res.times[res.milestones[0].key] == 200


def test_goal_state_from_record_fallback():
    b = [boundary(0, "begin", 0, 0, snap({BOWL: (0, 0, 0.9)}, goal={ON: False}))]
    assert ms.goal_state_from_record(make_record(b)) == [("on", BOWL, PLATE)]
