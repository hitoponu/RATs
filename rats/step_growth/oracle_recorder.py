"""Step-boundary oracle recorder.

Listens to plan-step markers (``with step_context(...)`` in generated policy
code, surfaced through :func:`rats.utils.execution_logger.register_policy_step_listener`)
and snapshots the LIBERO oracle at every begin/end boundary via
``FrankaLiberoEnv.describe_object_state()`` (rats/envs/simulators/libero.py).

The record is written to artifacts only. Nothing here is formatted into a
prompt (see docs/experiment-principles.md, 公平性の原則 2 and the
``grounded_state`` invariant in rats/executor/sandbox.py).
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Any

logger = logging.getLogger("rats.step_growth.oracle")

SCHEMA = "rats_step_oracle_v1"


def _json_safe(value: Any) -> Any:
    """numpy -> python scalars/lists; everything else passed through."""
    try:
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover - numpy always present in the loop
        np = None  # type: ignore
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if np is not None:
        if isinstance(value, np.ndarray):
            return [_json_safe(v) for v in value.tolist()]
        if isinstance(value, np.generic):
            return value.item()
    if isinstance(value, float):
        return float(value)
    return value


class StepOracleRecorder:
    """Collect oracle snapshots at plan-step boundaries for one attempt.

    Lifecycle (driven by :class:`StepGrowthController`):

    ``bind(low_level_env)`` -> ``begin_attempt(...)`` -> [step events arrive
    through ``on_step_event``] -> ``end_attempt(grounded_state)`` -> record.

    Events that arrive outside an open attempt window (e.g. the policy
    self-check dry run) are ignored, which is how the recorder stays out of
    the pre-execution repair loop without any loop-side bookkeeping.
    """

    def __init__(self, cfg: Any | None = None) -> None:
        self.cfg = cfg
        self._low: Any = None
        self._record: dict[str, Any] | None = None
        self._open = False
        self.snapshot_errors = 0

    # ------------------------------------------------------------------ setup
    def bind(self, low_level_env: Any) -> None:
        self._low = low_level_env

    @property
    def bound(self) -> bool:
        return self._low is not None

    def begin_attempt(
        self,
        *,
        iteration: int,
        attempt: int,
        attempt_in_iter: int,
        turn_in_attempt: int,
        env_reset: bool,
    ) -> None:
        """Open the recording window for one policy execution.

        ``env_reset=False`` (a later turn inside the same attempt) keeps the
        boundaries collected so far: the env was not reset, so milestone
        times keep counting on the same ``_sim_step_count`` clock.
        """
        if env_reset or self._record is None:
            self._record = {
                "schema": SCHEMA,
                "iteration": int(iteration),
                "attempt": int(attempt),
                "attempt_in_iter": int(attempt_in_iter),
                "turn_in_attempt": int(turn_in_attempt),
                "env_reset": bool(env_reset),
                "turns": [int(turn_in_attempt)],
                "markers_seen": False,
                "boundaries": [],
                "pick_events": [],
                "baseline_z": {},
                "attempt_before": None,
                "attempt_after": None,
                "started_at": time.time(),
            }
        else:
            self._record["attempt"] = int(attempt)
            self._record["turn_in_attempt"] = int(turn_in_attempt)
            self._record["turns"].append(int(turn_in_attempt))
        self._open = True

    # ---------------------------------------------------------------- events
    def on_step_event(self, event: dict[str, Any]) -> None:
        """Listener callback (see execution_logger.register_policy_step_listener)."""
        if not self._open or self._record is None or self._low is None:
            return
        boundary: dict[str, Any] = {
            "seq": len(self._record["boundaries"]),
            "phase": str(event.get("phase", "")),
            "step_id": event.get("step_id"),
            "step_index": event.get("step_index"),
            "step_goal": event.get("step_goal"),
            "marker_index": event.get("marker_index"),
            "frame": event.get("frame"),
            "exc_in_flight": bool(event.get("exc_in_flight", False)),
            "turn_in_attempt": self._record.get("turn_in_attempt"),
        }
        boundary.update(self._probe())
        self._record["boundaries"].append(boundary)
        self._record["markers_seen"] = True

    def _probe(self) -> dict[str, Any]:
        low = self._low
        out: dict[str, Any] = {
            "sim_step": None,
            "eef_pos": None,
            "gripper_fraction": None,
            "snapshot": None,
            "snapshot_error": None,
        }
        try:
            out["sim_step"] = int(getattr(low, "_sim_step_count", 0) or 0)
        except Exception:
            pass
        try:
            gf = getattr(low, "_gripper_fraction", None)
            out["gripper_fraction"] = float(gf) if gf is not None else None
        except Exception:
            pass
        try:
            idx = getattr(low, "gripper_link_idx", None)
            if idx is not None:
                xpos = low.handle.env.sim.data.xpos[idx]
                out["eef_pos"] = [float(x) for x in xpos]
        except Exception:
            pass
        try:
            snap = low.describe_object_state()
            snap = _json_safe(snap) if isinstance(snap, dict) else None
            if snap is not None:
                # pick_events are collected once at end_attempt; keep the
                # per-boundary snapshot small.
                snap.pop("pick_events", None)
            out["snapshot"] = snap
        except Exception as exc:
            self.snapshot_errors += 1
            out["snapshot_error"] = str(exc)[:200]
        return out

    # ----------------------------------------------------------------- close
    def end_attempt(self, grounded_state: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Close the window and return a deep copy of the record."""
        if self._record is None:
            return None
        self._open = False
        rec = self._record
        low = self._low
        try:
            rec["pick_events"] = _json_safe(list(getattr(low, "_pick_events", []) or []))
        except Exception:
            rec["pick_events"] = []
        try:
            rec["baseline_z"] = _json_safe(dict(getattr(low, "_pick_baseline_z", {}) or {}))
        except Exception:
            rec["baseline_z"] = {}
        try:
            rec["final_sim_step"] = int(getattr(low, "_sim_step_count", 0) or 0)
        except Exception:
            rec["final_sim_step"] = None
        if isinstance(grounded_state, dict):
            before = grounded_state.get("before") or {}
            after = grounded_state.get("after") or {}
            rec["attempt_before"] = _json_safe(before.get("object_state")) if isinstance(before, dict) else None
            rec["attempt_after"] = _json_safe(after.get("object_state")) if isinstance(after, dict) else None
        rec["ended_at"] = time.time()
        rec["snapshot_errors"] = int(self.snapshot_errors)
        return copy.deepcopy(rec)

    def discard(self) -> None:
        self._open = False
        self._record = None
