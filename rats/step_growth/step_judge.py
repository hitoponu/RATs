"""Step verdicts from milestone times + step markers.

Rule (decided 2026-09-08):
  * t*  = time of the highest milestone reached; S* = the step active at t*.
  * fail = the step containing the FIRST oracle failure event after t*
           (dropped / slipped), else the step executed right after S*.
  * pass = every executed step before the fail step (S* included);
    steps after the fail step are ``unjudged``; plan steps that never ran are
    ``not_executed``.
  * no milestone at all: fail = first failure event's step, else the first
    executed step; the rest unjudged.
No plan text or policy code is read here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rats.step_growth.milestones import MilestoneResult


@dataclass
class StepSpan:
    step_index: int
    step_id: str
    step_goal: str
    order: int                    # execution order (first begin)
    begin_sim_step: int | None
    end_sim_step: int | None      # None = never closed (crash/timeout)
    exc_in_flight: bool = False
    turns: list[int] = field(default_factory=list)


@dataclass
class StepVerdict:
    step_index: int
    step_id: str
    step_goal: str
    verdict: str                  # pass | fail | unjudged | not_executed
    reason: str = ""
    begin_sim_step: int | None = None
    end_sim_step: int | None = None
    exc_in_flight: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index, "step_id": self.step_id,
            "step_goal": self.step_goal, "verdict": self.verdict, "reason": self.reason,
            "begin_sim_step": self.begin_sim_step, "end_sim_step": self.end_sim_step,
            "exc_in_flight": self.exc_in_flight,
        }


@dataclass
class AttemptVerdicts:
    steps: list[StepVerdict]
    s_star: int | None
    t_star: int | None
    fail_step: int | None
    fail_reason: str | None
    progress: float
    achieved: list[str]
    failure_events: list[dict[str, Any]]
    markers_seen: bool
    multi_chain_approx: bool = False

    @property
    def pass_steps(self) -> list[int]:
        return [s.step_index for s in self.steps if s.verdict == "pass"]

    @property
    def fail_steps(self) -> list[int]:
        return [s.step_index for s in self.steps if s.verdict == "fail"]

    def as_dict(self) -> dict[str, Any]:
        return {
            "s_star": self.s_star, "t_star": self.t_star,
            "fail_step": self.fail_step, "fail_reason": self.fail_reason,
            "progress": self.progress, "achieved": list(self.achieved),
            "failure_events": list(self.failure_events),
            "markers_seen": self.markers_seen,
            "multi_chain_approx": self.multi_chain_approx,
            "steps": [s.as_dict() for s in self.steps],
        }


# ---------------------------------------------------------------------------
def step_spans(record: dict[str, Any]) -> list[StepSpan]:
    """Group boundaries by step_index into execution-ordered spans."""
    spans: dict[int, StepSpan] = {}
    order = 0
    bs = sorted(
        (b for b in (record.get("boundaries") or []) if isinstance(b, dict)),
        key=lambda b: b.get("seq", 0),
    )
    for b in bs:
        idx = b.get("step_index")
        if idx is None:
            # marker without step_index: derive from step_id digits (1-based)
            sid = str(b.get("step_id") or "")
            digits = "".join(ch for ch in sid if ch.isdigit())
            if not digits:
                continue
            idx = max(int(digits) - 1, 0)
        idx = int(idx)
        t = b.get("sim_step")
        if idx not in spans:
            spans[idx] = StepSpan(
                step_index=idx,
                step_id=str(b.get("step_id") or f"step-{idx + 1}"),
                step_goal=str(b.get("step_goal") or ""),
                order=order,
                begin_sim_step=int(t) if (t is not None and b.get("phase") == "begin") else None,
                end_sim_step=None,
            )
            order += 1
        sp = spans[idx]
        if b.get("phase") == "begin":
            if sp.begin_sim_step is None and t is not None:
                sp.begin_sim_step = int(t)
        elif b.get("phase") == "end":
            if t is not None:
                sp.end_sim_step = int(t) if sp.end_sim_step is None else max(sp.end_sim_step, int(t))
            if b.get("exc_in_flight"):
                sp.exc_in_flight = True
        turn = b.get("turn_in_attempt")
        if turn is not None and turn not in sp.turns:
            sp.turns.append(int(turn))
    return sorted(spans.values(), key=lambda s: s.order)


def step_at(spans: list[StepSpan], t: int | None, final_sim_step: int | None = None) -> StepSpan | None:
    """Innermost span whose [begin, end] contains t (open spans extend to the end).

    At an exact boundary (step k ended at t and step k+1 began at t) the
    state was produced by step k, so a span that *ends* at t wins over one
    that *begins* at t.
    """
    if t is None:
        return None
    cands: list[StepSpan] = []
    for sp in spans:
        b = sp.begin_sim_step
        if b is None or b > t:
            continue
        e = sp.end_sim_step if sp.end_sim_step is not None else final_sim_step
        if e is not None and e < t:
            continue
        cands.append(sp)
    if not cands:
        return None
    ending_here = [sp for sp in cands if sp.end_sim_step == t]
    if ending_here and len(cands) > 1:
        cands = ending_here
    return max(cands, key=lambda sp: (sp.begin_sim_step or -1, sp.order))


def judge(
    result: MilestoneResult,
    record: dict[str, Any],
    plan_steps: list[dict[str, Any]] | None = None,
    cfg: Any | None = None,
) -> AttemptVerdicts:
    spans = step_spans(record)
    final_t = record.get("final_sim_step")
    markers_seen = bool(spans)
    achieved = [m for m in result.milestones if result.times.get(m.key) is not None]
    achieved_keys = [m.key for m in achieved]
    events = list(result.events)

    chains = {m.chain for m in result.milestones}
    multi = len(chains) > 1

    # t*: highest reached milestone. With several chains, take the latest
    # per-chain top milestone time (multi_chain=max_time).
    t_star: int | None = None
    top_milestone = None
    if achieved:
        per_chain_top: dict[int, tuple[int, Any]] = {}
        for m in achieved:
            t = int(result.times[m.key])
            cur = per_chain_top.get(m.chain)
            if cur is None or t >= cur[0]:
                per_chain_top[m.chain] = (t, m)
        if per_chain_top:
            t_star, top_milestone = max(per_chain_top.values(), key=lambda x: x[0])

    if not markers_seen:
        return AttemptVerdicts(
            steps=[
                StepVerdict(i, str(s.get("id") or s.get("step_id") or f"step-{i + 1}"),
                            str(s.get("description") or ""), "not_executed", "no_markers")
                for i, s in enumerate(plan_steps or [])
            ],
            s_star=None, t_star=t_star, fail_step=None, fail_reason="no_markers",
            progress=result.progress, achieved=achieved_keys, failure_events=events,
            markers_seen=False, multi_chain_approx=multi,
        )

    s_star_span: StepSpan | None = None
    if t_star is not None:
        # Prefer the boundary that observed the milestone: its step_index is
        # exact, whereas a time lookup is ambiguous at shared boundary times.
        src = result.sources.get(top_milestone.key) if top_milestone is not None else None
        if src and src.get("step_index") is not None:
            s_star_span = next((sp for sp in spans if sp.step_index == int(src["step_index"])), None)
        if s_star_span is None:
            s_star_span = step_at(spans, t_star, final_t)
        if s_star_span is None:
            # t* outside any span (marker gap): last span that began before t*
            cands = [sp for sp in spans if sp.begin_sim_step is not None and sp.begin_sim_step <= t_star]
            s_star_span = cands[-1] if cands else None

    # failure step
    fail_span: StepSpan | None = None
    fail_reason: str | None = None
    later_events = [e for e in events if t_star is None or int(e.get("sim_step", -1)) > t_star]
    if later_events:
        ev = later_events[0]
        if ev.get("step_index") is not None:
            fail_span = next((sp for sp in spans if sp.step_index == int(ev["step_index"])), None)
        if fail_span is None:
            fail_span = step_at(spans, ev.get("sim_step"), final_t)
        if fail_span is not None:
            fail_reason = f"failure_event:{ev.get('event')}"
    if fail_span is None:
        if s_star_span is not None:
            after = [sp for sp in spans if sp.order > s_star_span.order]
            if after:
                fail_span = after[0]
                fail_reason = "next_after_milestone"
        elif t_star is None:
            fail_span = spans[0]
            fail_reason = "no_milestone_first_step"

    verdicts: list[StepVerdict] = []
    fail_order = fail_span.order if fail_span is not None else None
    for sp in spans:
        if fail_order is None:
            v, why = "pass", "before_or_at_milestone"
        elif sp.order < fail_order:
            v, why = "pass", "before_fail_step"
        elif sp.order == fail_order:
            v, why = "fail", fail_reason or "fail"
        else:
            v, why = "unjudged", "after_fail_step"
        verdicts.append(StepVerdict(
            sp.step_index, sp.step_id, sp.step_goal, v, why,
            sp.begin_sim_step, sp.end_sim_step, sp.exc_in_flight,
        ))
    executed = {sp.step_index for sp in spans}
    for i, s in enumerate(plan_steps or []):
        if i in executed:
            continue
        verdicts.append(StepVerdict(
            i, str(s.get("id") or s.get("step_id") or f"step-{i + 1}"),
            str(s.get("description") or ""), "not_executed", "never_ran",
        ))
    verdicts.sort(key=lambda v: (0 if v.verdict != "not_executed" else 1, v.step_index))

    return AttemptVerdicts(
        steps=verdicts,
        s_star=s_star_span.step_index if s_star_span else None,
        t_star=t_star,
        fail_step=fail_span.step_index if fail_span else None,
        fail_reason=fail_reason,
        progress=result.progress,
        achieved=achieved_keys,
        failure_events=events,
        markers_seen=True,
        multi_chain_approx=multi,
    )
