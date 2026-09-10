"""Shared fixtures for the step-growth tests (no simulator, no LLM)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# synthetic oracle records
# ---------------------------------------------------------------------------
def snap(
    objects: dict[str, tuple[float, float, float]],
    *,
    contact: list[str] | None = None,
    goal: dict[str, bool] | None = None,
    relations: list[tuple[str, str, str]] | None = None,
    open_flags: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Build a ``describe_object_state()``-shaped snapshot."""
    objs = {
        name: {"pos": list(map(float, pos)), "quat_wxyz": [1.0, 0.0, 0.0, 0.0], "type": "object"}
        for name, pos in objects.items()
    }
    for name, flag in (open_flags or {}).items():
        objs.setdefault(name, {"pos": [0.0, 0.0, 0.0], "quat_wxyz": [1, 0, 0, 0], "type": "fixture"})
        objs[name]["open"] = bool(flag)
    return {
        "objects": objs,
        "relations": [{"rel": r, "a": a, "b": b, "b_is_site": False} for r, a, b in (relations or [])],
        "fingerpad_contact": list(contact or []),
        "goal": [{"predicate": p, "satisfied": bool(v)} for p, v in (goal or {}).items()],
        "success": all((goal or {}).values()) if goal else False,
        "picked": [], "picked_wrong": [], "goal_target": None, "goal_destination": None,
        "pick_only": False,
    }


def boundary(
    seq: int,
    phase: str,
    step_index: int,
    sim_step: int,
    snapshot: dict[str, Any] | None,
    *,
    gripper: float = 1.0,
    step_id: str | None = None,
    exc: bool = False,
    turn: int = 0,
) -> dict[str, Any]:
    return {
        "seq": seq, "phase": phase, "step_index": step_index,
        "step_id": step_id or f"step-{step_index + 1}", "step_goal": f"goal {step_index}",
        "marker_index": seq // 2, "frame": None, "exc_in_flight": exc,
        "turn_in_attempt": turn, "sim_step": sim_step, "eef_pos": [0.0, 0.0, 1.0],
        "gripper_fraction": gripper, "snapshot": snapshot, "snapshot_error": None,
    }


def make_record(
    boundaries: list[dict[str, Any]],
    *,
    pick_events: list[dict[str, Any]] | None = None,
    baseline_z: dict[str, float] | None = None,
    final_sim_step: int | None = None,
) -> dict[str, Any]:
    return {
        "schema": "rats_step_oracle_v1", "iteration": 1, "attempt": 0,
        "attempt_in_iter": 0, "turn_in_attempt": 0, "env_reset": True, "turns": [0],
        "markers_seen": bool(boundaries), "boundaries": boundaries,
        "pick_events": list(pick_events or []), "baseline_z": dict(baseline_z or {}),
        "attempt_before": None, "attempt_after": None,
        "final_sim_step": final_sim_step if final_sim_step is not None else (
            max((b["sim_step"] for b in boundaries), default=0)
        ),
        "snapshot_errors": 0,
    }


class _Sim:
    def __init__(self) -> None:
        self.data = type("D", (), {})()
        self.data.xpos = [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]]


class _Handle:
    def __init__(self) -> None:
        self.env = type("E", (), {})()
        self.env.sim = _Sim()


class FakeLowLevelEnv:
    """Minimal stand-in for FrankaLiberoEnv used by the recorder/controller.

    ``script`` is a list of snapshots returned in order by
    ``describe_object_state()``; the last one repeats once exhausted.
    """

    def __init__(self, script: list[dict[str, Any]], goal_state: list[tuple[str, ...]] | None = None) -> None:
        self._script = list(script)
        self._calls = 0
        self._sim_step_count = 0
        self._gripper_fraction = 1.0
        self._pick_events: list[dict[str, Any]] = []
        self._pick_baseline_z: dict[str, float] = {}
        self._pick_max_dz: dict[str, float] = {}
        self.gripper_link_idx = 1
        self.handle = _Handle()
        self._goal_state = goal_state or []

    def describe_object_state(self) -> dict[str, Any]:
        i = min(self._calls, len(self._script) - 1)
        self._calls += 1
        return self._script[i]

    def _predicate_env(self) -> Any:
        stub = type("P", (), {})()
        stub.parsed_problem = {"goal_state": [list(g) for g in self._goal_state]}
        stub._eval_predicate = lambda state: False
        return stub


@pytest.fixture
def fake_env_factory():
    return FakeLowLevelEnv


@pytest.fixture
def tmp_skills_path(tmp_path: Path) -> Path:
    return tmp_path / "skills.json"
