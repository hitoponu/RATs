"""Milestone chains derived mechanically from the BDDL goal, and their
first-true times / failure events computed from an oracle record.

No plan text and no policy code is interpreted here. Inputs are:
  * ``goal_state`` — ``parsed_problem["goal_state"]`` (list of predicate
    tuples such as ``("on", "akita_black_bowl_1", "plate_1")``) or the
    ``[on a b]`` strings in a ``describe_object_state()`` snapshot;
  * the recorder's boundaries (``sim_step``, ``snapshot``, ``gripper_fraction``,
    ``eef_pos``) and ``pick_events`` (``{"object", "sim_step", "z0", "z", "dz"}``).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

# predicate name (lower-case) -> ordered milestone kinds
CHAIN_TEMPLATES: dict[str, list[str]] = {
    "on": ["grasped", "lifted", "near", "placed"],
    "in": ["grasped", "lifted", "near", "placed"],
    "open": ["open"],
    "close": ["closed"],
    "turnon": ["turnon"],
    "turnoff": ["turnoff"],
}

# milestone kinds whose first-true time comes from the goal predicate bit
_GOAL_BIT_KINDS = {"open", "closed", "turnon", "turnoff"}


@dataclass
class Milestone:
    key: str                 # unique, e.g. "grasped(akita_black_bowl_1)"
    kind: str                # grasped | lifted | near | placed | open | closed | turnon | turnoff
    args: tuple[str, ...]    # predicate args (object names)
    predicate: str           # goal predicate string as libero prints it: "[on a b]"
    chain: int               # chain index (one per goal predicate)
    pos: int                 # position inside the chain

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "kind": self.kind, "args": list(self.args),
            "predicate": self.predicate, "chain": self.chain, "pos": self.pos,
        }


@dataclass
class MilestoneResult:
    milestones: list[Milestone]
    times: dict[str, int | None]            # key -> first-true sim_step
    events: list[dict[str, Any]] = field(default_factory=list)   # failure events
    goal_state: list[tuple[str, ...]] = field(default_factory=list)
    # key -> {"sim_step", "boundary_seq", "step_index"} (step_index None when
    # the time came from a pick_event rather than a boundary)
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def achieved(self) -> list[Milestone]:
        return [m for m in self.milestones if self.times.get(m.key) is not None]

    @property
    def progress(self) -> float:
        if not self.milestones:
            return 0.0
        return len(self.achieved) / float(len(self.milestones))

    def as_dict(self) -> dict[str, Any]:
        return {
            "goal_state": [list(g) for g in self.goal_state],
            "milestones": [
                {**m.as_dict(), "first_true_sim_step": self.times.get(m.key),
                 "source": self.sources.get(m.key)}
                for m in self.milestones
            ],
            "achieved": [m.key for m in self.achieved],
            "progress": self.progress,
            "failure_events": list(self.events),
        }


# ---------------------------------------------------------------------------
# goal parsing
# ---------------------------------------------------------------------------
_PRED_STR_RE = re.compile(r"^\[?\s*([A-Za-z_]+)\s+([^\s\]]+)(?:\s+([^\s\]]+))?\s*\]?$")


def parse_goal_state(goal_state: Any) -> list[tuple[str, ...]]:
    """Normalise goal predicates to lower-case tuples.

    Accepts LIBERO's ``parsed_problem["goal_state"]`` (lists/tuples), the
    ``[on a b]`` strings from ``describe_object_state()["goal"]`` entries, or
    dicts with a ``predicate`` key.
    """
    out: list[tuple[str, ...]] = []
    for item in goal_state or []:
        if isinstance(item, dict):
            item = item.get("predicate")
        if isinstance(item, str):
            m = _PRED_STR_RE.match(item.strip())
            if not m:
                continue
            parts = [p for p in m.groups() if p]
            out.append(tuple([parts[0].lower(), *parts[1:]]))
        elif isinstance(item, (list, tuple)) and item:
            head = str(item[0]).lower()
            if head in ("and", "or", "not"):
                # nested logical form: flatten one level
                for sub in item[1:]:
                    out.extend(parse_goal_state([sub]))
                continue
            out.append(tuple([head, *[str(x) for x in item[1:]]]))
    return out


def derive_milestones(goal_state: Any) -> list[Milestone]:
    """Goal predicates -> ordered milestone chain(s). Unknown predicates are skipped."""
    milestones: list[Milestone] = []
    for chain_idx, pred in enumerate(parse_goal_state(goal_state)):
        name, args = pred[0], tuple(pred[1:])
        kinds = CHAIN_TEMPLATES.get(name)
        if not kinds:
            continue
        pred_str = "[" + " ".join([name, *args]) + "]"
        for pos, kind in enumerate(kinds):
            if kind in ("grasped", "lifted"):
                margs = args[:1]
            else:
                margs = args
            key = f"{kind}({', '.join(margs)})"
            milestones.append(Milestone(key, kind, margs, pred_str, chain_idx, pos))
    return milestones


# ---------------------------------------------------------------------------
# snapshot helpers
# ---------------------------------------------------------------------------
def _goal_bit(snapshot: dict[str, Any] | None, predicate: str) -> bool | None:
    if not snapshot:
        return None
    want = predicate.strip().lower()
    for g in snapshot.get("goal") or []:
        p = str(g.get("predicate", "")).strip().lower()
        if p == want:
            return bool(g.get("satisfied"))
    return None


def _obj_pos(snapshot: dict[str, Any] | None, name: str) -> list[float] | None:
    if not snapshot:
        return None
    obj = (snapshot.get("objects") or {}).get(name)
    if not isinstance(obj, dict):
        return None
    pos = obj.get("pos")
    if not isinstance(pos, (list, tuple)) or len(pos) < 3:
        return None
    try:
        return [float(pos[0]), float(pos[1]), float(pos[2])]
    except (TypeError, ValueError):
        return None


def _in_contact(snapshot: dict[str, Any] | None, name: str) -> bool:
    if not snapshot:
        return False
    return name in (snapshot.get("fingerpad_contact") or [])


def _relation_holds(snapshot: dict[str, Any] | None, rel: str, a: str, b: str) -> bool:
    if not snapshot:
        return False
    for r in snapshot.get("relations") or []:
        if str(r.get("rel", "")).lower() == rel and r.get("a") == a and r.get("b") == b:
            return True
    return False


def _sorted_boundaries(record: dict[str, Any]) -> list[dict[str, Any]]:
    bs = [b for b in (record.get("boundaries") or []) if isinstance(b, dict)]
    return sorted(bs, key=lambda b: (b.get("sim_step") if b.get("sim_step") is not None else -1, b.get("seq", 0)))


# ---------------------------------------------------------------------------
# first-true times
# ---------------------------------------------------------------------------
def first_true_times(
    milestones: list[Milestone],
    record: dict[str, Any],
    cfg: Any | None = None,
    sources: dict[str, dict[str, Any]] | None = None,
) -> dict[str, int | None]:
    """Chain-ordered first-true sim_step per milestone.

    Each milestone is searched only from its predecessor's time onward, so a
    goal that happens to hold at reset (object already near the target)
    cannot leak into the chain before the grasp happened.
    """
    lift_dz = float(getattr(cfg, "lift_dz_m", 0.03))
    grasp_dz = float(getattr(cfg, "grasp_end_dz_m", 0.005))
    closed_max = float(getattr(cfg, "gripper_closed_max_fraction", 0.6))
    near_r = float(getattr(cfg, "near_radius_m", 0.10))
    near_ztol = float(getattr(cfg, "near_z_tolerance_m", 0.02))

    boundaries = _sorted_boundaries(record)
    pick_events = [e for e in (record.get("pick_events") or []) if isinstance(e, dict)]
    baseline_z: dict[str, float] = {}
    for k, v in (record.get("baseline_z") or {}).items():
        try:
            baseline_z[str(k)] = float(v)
        except (TypeError, ValueError):
            pass

    def _dz(snapshot: dict[str, Any] | None, name: str) -> float | None:
        pos = _obj_pos(snapshot, name)
        z0 = baseline_z.get(name)
        if pos is None or z0 is None:
            return None
        return pos[2] - z0

    last_source: dict[str, Any] = {}

    def _first_boundary(pred, t_min: int | None) -> int | None:
        for b in boundaries:
            t = b.get("sim_step")
            if t is None:
                continue
            if t_min is not None and t < t_min:
                continue
            try:
                if pred(b):
                    last_source.clear()
                    last_source.update({
                        "sim_step": int(t), "boundary_seq": b.get("seq"),
                        "step_index": b.get("step_index"), "phase": b.get("phase"),
                    })
                    return int(t)
            except Exception:
                continue
        return None

    times: dict[str, int | None] = {}
    # chain -> time of the previous milestone in that chain
    prev_time: dict[int, int | None] = {}
    prev_missing: dict[int, bool] = {}

    for m in milestones:
        if prev_missing.get(m.chain):
            times[m.key] = None
            continue
        t_prev = prev_time.get(m.chain)
        a = m.args[0] if m.args else ""
        b = m.args[1] if len(m.args) > 1 else ""
        t: int | None = None

        last_source.clear()
        if m.kind == "grasped":
            ev = [int(e["sim_step"]) for e in pick_events if e.get("object") == a and e.get("sim_step") is not None]
            t = min(ev) if ev else None
            if t is not None:
                last_source.update({"sim_step": t, "boundary_seq": None, "step_index": None, "phase": "pick_event"})
            if t is None:
                t = _first_boundary(
                    lambda bd: _in_contact(bd.get("snapshot"), a)
                    and bd.get("gripper_fraction") is not None
                    and float(bd["gripper_fraction"]) <= closed_max
                    and (_dz(bd.get("snapshot"), a) or 0.0) >= grasp_dz,
                    t_prev,
                )
        elif m.kind == "lifted":
            ev = [int(e["sim_step"]) for e in pick_events if e.get("object") == a and e.get("sim_step") is not None]
            t = min(ev) if ev else None
            if t is not None:
                last_source.update({"sim_step": t, "boundary_seq": None, "step_index": None, "phase": "pick_event"})
            if t is None:
                t = _first_boundary(
                    lambda bd: _in_contact(bd.get("snapshot"), a)
                    and (_dz(bd.get("snapshot"), a) or 0.0) >= lift_dz,
                    t_prev,
                )
        elif m.kind == "near":
            def _near(bd: dict[str, Any]) -> bool:
                snap = bd.get("snapshot")
                pa, pb = _obj_pos(snap, a), _obj_pos(snap, b)
                if pa is None or pb is None:
                    return False
                d = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
                return d <= near_r and pa[2] >= pb[2] - near_ztol
            t = _first_boundary(_near, t_prev)
        elif m.kind == "placed":
            rel = m.predicate.strip("[]").split()[0].lower()

            def _placed(bd: dict[str, Any]) -> bool:
                snap = bd.get("snapshot")
                bit = _goal_bit(snap, m.predicate)
                holds = bool(bit) if bit is not None else _relation_holds(snap, rel, a, b)
                return holds and not _in_contact(snap, a)
            t = _first_boundary(_placed, t_prev)
        elif m.kind in _GOAL_BIT_KINDS:
            t = _first_boundary(lambda bd: bool(_goal_bit(bd.get("snapshot"), m.predicate)), t_prev)
        times[m.key] = t
        if t is not None and sources is not None:
            sources[m.key] = dict(last_source)
        prev_time[m.chain] = t
        if t is None:
            prev_missing[m.chain] = True
    return times


# ---------------------------------------------------------------------------
# failure events
# ---------------------------------------------------------------------------
def failure_events(
    milestones: list[Milestone],
    times: dict[str, int | None],
    record: dict[str, Any],
    cfg: Any | None = None,
) -> list[dict[str, Any]]:
    """Oracle-derived failure evidence with times: ``dropped(a)`` / ``slipped(a)``.

    dropped: after lifted(a), a boundary where ``a`` is out of contact and the
             chain's placed milestone is not (yet) true.
    slipped: a boundary where ``a`` was in contact with a closed gripper,
             followed by a boundary with no contact, before lifted(a).
    Times are boundary ``sim_step``s (the earliest boundary that shows the
    evidence). One event per kind per object.
    """
    closed_max = float(getattr(cfg, "gripper_closed_max_fraction", 0.6))
    boundaries = _sorted_boundaries(record)
    events: list[dict[str, Any]] = []
    seen_objects: set[str] = set()
    by_chain: dict[int, list[Milestone]] = {}
    for m in milestones:
        by_chain.setdefault(m.chain, []).append(m)

    for chain_ms in by_chain.values():
        kinds = {m.kind: m for m in chain_ms}
        if "grasped" not in kinds:
            continue
        a = kinds["grasped"].args[0]
        if a in seen_objects:
            continue
        seen_objects.add(a)
        t_lift = times.get(kinds["lifted"].key) if "lifted" in kinds else None
        t_placed = times.get(kinds["placed"].key) if "placed" in kinds else None

        if t_lift is not None:
            for bd in boundaries:
                t = bd.get("sim_step")
                if t is None or t <= t_lift:
                    continue
                if t_placed is not None and t >= t_placed:
                    break
                if not _in_contact(bd.get("snapshot"), a):
                    events.append({
                        "event": "dropped", "object": a, "sim_step": int(t),
                        "step_index": bd.get("step_index"), "boundary_seq": bd.get("seq"),
                    })
                    break
        else:
            held_seen = False
            for bd in boundaries:
                t = bd.get("sim_step")
                if t is None:
                    continue
                snap = bd.get("snapshot")
                gf = bd.get("gripper_fraction")
                closed = gf is not None and float(gf) <= closed_max
                if _in_contact(snap, a) and closed:
                    held_seen = True
                    continue
                if held_seen and not _in_contact(snap, a):
                    events.append({
                        "event": "slipped", "object": a, "sim_step": int(t),
                        "step_index": bd.get("step_index"), "boundary_seq": bd.get("seq"),
                    })
                    break
    events.sort(key=lambda e: e["sim_step"])
    return events


def evaluate(goal_state: Any, record: dict[str, Any], cfg: Any | None = None) -> MilestoneResult:
    """Convenience: derive chains, first-true times and failure events."""
    parsed = parse_goal_state(goal_state)
    ms = derive_milestones(parsed)
    sources: dict[str, dict[str, Any]] = {}
    times = first_true_times(ms, record, cfg, sources)
    events = failure_events(ms, times, record, cfg)
    return MilestoneResult(milestones=ms, times=times, events=events, goal_state=parsed, sources=sources)


def goal_state_from_record(record: dict[str, Any]) -> list[tuple[str, ...]]:
    """Fallback goal source: the ``goal`` list of the first snapshot in the record."""
    for bd in _sorted_boundaries(record):
        snap = bd.get("snapshot")
        if isinstance(snap, dict) and snap.get("goal"):
            return parse_goal_state(snap["goal"])
    for key in ("attempt_before", "attempt_after"):
        snap = record.get(key)
        if isinstance(snap, dict) and snap.get("goal"):
            return parse_goal_state(snap["goal"])
    return []
