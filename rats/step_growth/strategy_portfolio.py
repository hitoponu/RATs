"""Which physical strategy family each step type gets this attempt.

Section B of the step-growth plan (2026-09-08). Inert unless the diversity
half of the arm is on; nothing here runs in the default step-growth arm.

The unit of diversity is a *physical strategy family* (GraspNet 6-DoF vs
OBB top-down yaw vs Molmo side pinch ...), not a code variant: run1 collapsed
onto one hand-built identity-quaternion recipe and kept failing the same way.
A family is chosen per step TYPE, and the step types come from the BDDL goal
predicates -- never from the plan prose, so the selection cannot be steered by
whatever the planner happened to write.

Selection is Thompson sampling over Beta(alpha, beta) per family with forced
exploration (a family with fewer than ``min_pulls`` pulls inside the last
``window_iters`` iterations is tried first). Reward is the milestone the
oracle recorded, so it is library bookkeeping, exactly like step credit:
nothing from the oracle is rendered into the directive unless
``show_evidence`` is turned on.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger("rats.step_growth.diversity")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BANK_PATH = _PROJECT_ROOT / "rats" / "config" / "strategy_bank.yaml"

STATE_SCHEMA = "rats_strategy_portfolio_v1"

# goal predicate -> step types it needs, in execution order.
PREDICATE_STEP_TYPES: dict[str, tuple[str, ...]] = {
    "on": ("grasp", "place"),
    "in": ("grasp", "place"),
    "open": ("open_close",),
    "close": ("open_close",),
    "turnon": ("turn",),
    "turnoff": ("turn",),
}

# step type -> (milestone kinds that pay it, milestone kinds that gate it).
# A gated type is not judged at all until the gate was reached: a place family
# cannot be blamed for an attempt that never lifted the object.
TYPE_REWARD_KINDS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "grasp": (frozenset({"lifted"}), frozenset()),
    "place": (frozenset({"placed"}), frozenset({"lifted"})),
    "open_close": (frozenset({"open", "closed"}), frozenset()),
    "turn": (frozenset({"turnon", "turnoff"}), frozenset()),
    "localize": (frozenset({"grasped"}), frozenset()),
}

# Failure signals that justify switching family mid-iteration (plan B2).
SWITCH_FAILURE_MODES = {"grasp_failure", "wrong_affordance", "collision"}
SWITCH_EDIT_SCALES = {"argument_level"}

# Tags preferred when a collapse override fires.
COLLAPSE_PREFERRED_TAGS = frozenset({"plan_grasp", "obb_yaw"})


# ---------------------------------------------------------------------------
# bank
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Family:
    id: str
    step_types: tuple[str, ...]
    directive: str
    requires: tuple[str, ...] = ()
    forbids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    def available(self, available_functions: Iterable[str]) -> bool:
        have = set(available_functions or [])
        return all(r in have for r in self.requires)


class StrategyBank:
    """The families in ``rats/config/strategy_bank.yaml``."""

    def __init__(self, families: list[Family], common_forbids: list[str], source: str = "") -> None:
        self.families = families
        self.common_forbids = common_forbids
        self.source = source
        self._by_id = {f.id: f for f in families}

    # -- loading -----------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None = None) -> "StrategyBank":
        p = Path(path or DEFAULT_BANK_PATH)
        if not p.is_absolute():
            root_rel = _PROJECT_ROOT / p
            p = root_rel if root_rel.exists() else p
        if not p.exists():
            logger.warning("strategy bank not found (%s) — diversity has no families", p)
            return cls([], [], source=str(p))
        try:
            import yaml  # type: ignore

            raw = yaml.safe_load(p.read_text()) or {}
        except Exception as exc:
            logger.warning("strategy bank unreadable (%s): %s", p, exc)
            return cls([], [], source=str(p))
        families: list[Family] = []
        for item in raw.get("families") or []:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            families.append(Family(
                id=str(item["id"]),
                step_types=tuple(str(t) for t in (item.get("step_types") or ())),
                directive=" ".join(str(item.get("directive") or "").split()),
                requires=tuple(str(r) for r in (item.get("requires") or ())),
                forbids=tuple(str(r) for r in (item.get("forbids") or ())),
                tags=tuple(str(t) for t in (item.get("tags") or ())),
            ))
        common = [str(x) for x in (raw.get("common_forbids") or [])]
        return cls(families, common, source=str(p))

    # -- queries -----------------------------------------------------------
    def get(self, family_id: str) -> Family | None:
        return self._by_id.get(family_id)

    def for_type(self, step_type: str, available_functions: Iterable[str] | None = None) -> list[Family]:
        out = [f for f in self.families if step_type in f.step_types]
        if available_functions is not None:
            out = [f for f in out if f.available(available_functions)]
        return out

    def step_types(self) -> list[str]:
        seen: list[str] = []
        for f in self.families:
            for t in f.step_types:
                if t not in seen:
                    seen.append(t)
        return seen


def step_types_for_goal(goal_state: Any) -> list[str]:
    """Goal predicates -> the step types this task needs, in order.

    Reads only the BDDL goal (same source as the milestone chain), never the
    plan text, so a planner that decides to "not use plan_grasp" cannot move
    the selection.
    """
    from rats.step_growth.milestones import parse_goal_state

    out: list[str] = []
    for pred in parse_goal_state(goal_state):
        for t in PREDICATE_STEP_TYPES.get(pred[0], ()):
            if t not in out:
                out.append(t)
    return out


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------
@dataclass
class Selection:
    iteration: int
    attempt: int
    families: dict[str, str] = field(default_factory=dict)          # step_type -> family id
    banned: dict[str, list[str]] = field(default_factory=dict)      # step_type -> banned ids
    reasons: dict[str, str] = field(default_factory=dict)           # step_type -> why
    notes: list[str] = field(default_factory=list)                  # SWITCH / COLLAPSE lines
    step_types: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration, "attempt": self.attempt,
            "families": dict(self.families), "banned": {k: list(v) for k, v in self.banned.items()},
            "reasons": dict(self.reasons), "notes": list(self.notes),
            "step_types": list(self.step_types),
        }

    @property
    def is_empty(self) -> bool:
        return not self.families


class StrategyPortfolio:
    """Thompson-sampling bandit over strategy families, with its own state file."""

    def __init__(
        self,
        bank: StrategyBank,
        state_path: str | Path,
        *,
        rng_seed: int | None = None,
        min_pulls: int = 2,
        window_iters: int = 10,
        collapse_k: int = 5,
        per_type_families: dict[str, int] | None = None,
        show_evidence: bool = False,
    ) -> None:
        self.bank = bank
        self.state_path = Path(state_path)
        self.min_pulls = max(0, int(min_pulls))
        self.window_iters = max(1, int(window_iters))
        self.collapse_k = max(0, int(collapse_k))
        self.per_type_families = dict(per_type_families or {})
        self.show_evidence = bool(show_evidence)
        self.rng = random.Random(rng_seed)
        self._stats: dict[str, dict[str, Any]] = {}
        self._fingerprints: list[dict[str, Any]] = []
        self._collapse: dict[str, Any] | None = None
        self._load()

    # -- state -------------------------------------------------------------
    def _load(self) -> None:
        try:
            if not self.state_path.exists():
                return
            data = json.loads(self.state_path.read_text())
        except Exception as exc:
            logger.warning("strategy state unreadable (%s): %s — starting fresh", self.state_path, exc)
            return
        if not isinstance(data, dict):
            return
        stats = data.get("families")
        if isinstance(stats, dict):
            self._stats = {str(k): dict(v) for k, v in stats.items() if isinstance(v, dict)}
        fps = data.get("attempt0_fingerprints")
        if isinstance(fps, list):
            self._fingerprints = [dict(f) for f in fps if isinstance(f, dict)]
        col = data.get("collapse")
        self._collapse = dict(col) if isinstance(col, dict) else None

    def save(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema": STATE_SCHEMA,
                "bank": self.bank.source,
                "families": self._stats,
                "attempt0_fingerprints": self._fingerprints[-200:],
                "collapse": self._collapse,
                "updated_at": time.time(),
            }
            self.state_path.write_text(json.dumps(payload, indent=2))
        except Exception as exc:
            logger.debug("strategy state save failed: %s", exc)

    def _stat(self, family_id: str) -> dict[str, Any]:
        st = self._stats.get(family_id)
        if st is None:
            st = {"alpha": 1.0, "beta": 1.0, "pulls": 0, "rewards": 0, "history": []}
            self._stats[family_id] = st
        return st

    def _window_pulls(self, family_id: str, iteration: int) -> int:
        st = self._stats.get(family_id)
        if not st:
            return 0
        lo = iteration - self.window_iters
        return sum(1 for h in st.get("history") or [] if isinstance(h, (list, tuple)) and h and int(h[0]) > lo)

    def stats_snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            fid: {
                "alpha": float(st.get("alpha", 1.0)), "beta": float(st.get("beta", 1.0)),
                "pulls": int(st.get("pulls", 0)), "rewards": int(st.get("rewards", 0)),
            }
            for fid, st in sorted(self._stats.items())
        }

    # -- selection ---------------------------------------------------------
    def _wanted_types(self, goal_state: Any) -> list[str]:
        types = step_types_for_goal(goal_state)
        return [t for t in types if int(self.per_type_families.get(t, 1)) > 0]

    def _candidates(
        self, step_type: str, available_functions: Iterable[str], banned: list[str] | None,
    ) -> tuple[list[Family], bool, bool]:
        """Candidates for this type; whether the collapse override narrowed them,
        and whether the ban list had to be recycled.

        Recycling matters: a type has as few as two families (turn), and each
        mid-iteration switch bans one for the rest of the iteration. Once they
        were all banned the type dropped out of the selection entirely — in the
        first smoke (job 5258527) that is exactly when the drawer finally
        opened and when the stove finally turned on, so the two milestones the
        run did reach paid no family at all. Rather than leave a phase with no
        directive we re-open the pool, keeping out only the family that was
        banned last.
        """
        banned = list(banned or ())
        pool = self.bank.for_type(step_type, available_functions)
        cands = [f for f in pool if f.id not in set(banned)]
        recycled = False
        if not cands and pool:
            last = {banned[-1]} if banned else set()
            cands = [f for f in pool if f.id not in last] or list(pool)
            recycled = True
        voted = str((self._collapse or {}).get("family") or "")
        if not voted or not cands:
            return cands, False, recycled
        not_collapsed = [f for f in cands if voted not in f.tags]
        if not not_collapsed:
            return cands, False, recycled
        preferred = [f for f in not_collapsed if COLLAPSE_PREFERRED_TAGS & set(f.tags)]
        narrowed = preferred or not_collapsed
        return narrowed, len(narrowed) < len(cands), recycled

    def _pick(self, cands: list[Family], iteration: int) -> tuple[Family, str]:
        under = [f for f in cands if self._window_pulls(f.id, iteration) < self.min_pulls]
        if under:
            fewest = min(self._window_pulls(f.id, iteration) for f in under)
            pool = [f for f in under if self._window_pulls(f.id, iteration) == fewest]
            return self.rng.choice(pool), "forced_exploration"
        scored = [(self.rng.betavariate(float(self._stat(f.id)["alpha"]), float(self._stat(f.id)["beta"])), f)
                  for f in cands]
        best = max(s for s, _ in scored)
        pool = [f for s, f in scored if s >= best - 1e-12]
        return self.rng.choice(pool), "thompson"

    def select(
        self,
        *,
        goal_state: Any,
        available_functions: Iterable[str],
        iteration: int,
        attempt: int = 0,
        keep: dict[str, str] | None = None,
        banned: dict[str, list[str]] | None = None,
        notes: list[str] | None = None,
    ) -> Selection:
        """Choose one family per step type. ``keep`` pins families already in force."""
        sel = Selection(iteration=iteration, attempt=attempt, notes=list(notes or []))
        sel.banned = {k: list(v) for k, v in (banned or {}).items()}
        sel.step_types = self._wanted_types(goal_state)
        collapse = self._collapse or {}
        collapse_fired = False
        for t in sel.step_types:
            pinned = (keep or {}).get(t)
            if pinned and pinned not in sel.banned.get(t, []):
                fam = self.bank.get(pinned)
                if fam is not None and fam.available(available_functions):
                    sel.families[t] = pinned
                    sel.reasons[t] = "kept"
                    continue
            cands, narrowed, recycled = self._candidates(
                t, available_functions, sel.banned.get(t, []),
            )
            if not cands:
                sel.reasons[t] = "no_candidate"
                continue
            fam, why = self._pick(cands, iteration)
            sel.families[t] = fam.id
            sel.reasons[t] = f"{why}+collapse_override" if narrowed else why
            collapse_fired = collapse_fired or narrowed
            if recycled:
                # The pool was re-opened: keep only the most recent exclusion so
                # the bans do not immediately empty it again.
                sel.banned[t] = sel.banned.get(t, [])[-1:]
                sel.reasons[t] = f"{sel.reasons[t]}+recycled"
        if collapse_fired:
            sel.notes.append(
                f"COLLAPSE: the last {int(collapse.get('k', self.collapse_k))} first attempts all "
                f"used the same '{collapse.get('family')}' recipe with a hard-coded orientation; "
                "that recipe is excluded here."
            )
        return sel

    def select_for_retry(
        self,
        prev: Selection,
        *,
        misses: dict[str, int],
        retry_feedback: dict[str, Any] | None,
        diagnosis: dict[str, Any] | None,
        goal_state: Any,
        available_functions: Iterable[str],
        iteration: int,
        attempt: int,
        retry_switch_after: int = 2,
    ) -> Selection:
        """Keep the attempt-0 families unless a type missed its milestone twice.

        The switch also needs a failure signal that is about the *physics*
        (``grasp_failure`` / ``wrong_affordance`` / ``collision``) or a
        diagnoser that only wants argument-level edits -- otherwise the retry
        is about a code bug and changing strategy would throw away a working
        approach.
        """
        fb = retry_feedback or {}
        diag = diagnosis or {}
        mode = str(fb.get("failure_mode") or diag.get("failure_mode") or "").strip().lower()
        edit_scale = str(fb.get("edit_scale") or diag.get("edit_scale") or "").strip().lower()
        signal_ok = mode in SWITCH_FAILURE_MODES or edit_scale in SWITCH_EDIT_SCALES
        keep = dict(prev.families)
        banned = {k: list(v) for k, v in prev.banned.items()}
        notes: list[str] = []
        switched: list[str] = []
        for t, streak in sorted(misses.items()):
            if t not in prev.families or streak < int(retry_switch_after) or not signal_ok:
                continue
            old = prev.families[t]
            banned.setdefault(t, [])
            if old not in banned[t]:
                banned[t].append(old)
            keep.pop(t, None)
            switched.append(t)
            # Deliberately says only WHAT is forbidden, not why: the "why" is
            # the oracle milestone, and the arm's rule is that oracle
            # judgements stay in the library bookkeeping and never enter a
            # prompt. The agent already has the diagnoser's own account of the
            # failure in its retry context.
            notes.append(
                f"SWITCH: family '{old}' is FORBIDDEN in this attempt. Commit to a physically "
                f"different approach for the {t} phase."
            )
        sel = self.select(
            goal_state=goal_state, available_functions=available_functions,
            iteration=iteration, attempt=attempt, keep=keep, banned=banned, notes=notes,
        )
        for t in switched:
            if t in sel.families:
                # keep the "+recycled" marker if the pool had to be re-opened
                suffix = "+recycled" if "recycled" in sel.reasons.get(t, "") else ""
                sel.reasons[t] = f"switched{suffix}"
        return sel

    # -- reward ------------------------------------------------------------
    @staticmethod
    def _kinds(achieved: Iterable[str]) -> set[str]:
        out: set[str] = set()
        for key in achieved or []:
            k = str(key).split("(", 1)[0].strip()
            if k:
                out.add(k)
        return out

    def rewards_for(self, selection: Selection, achieved: Iterable[str]) -> dict[str, int]:
        """``step_type -> 1|0`` for the types this attempt actually judged."""
        kinds = self._kinds(achieved)
        out: dict[str, int] = {}
        for t in selection.families:
            pay, gate = TYPE_REWARD_KINDS.get(t, (frozenset(), frozenset()))
            if gate and not (gate & kinds):
                continue  # not reached far enough to judge this type
            out[t] = 1 if (pay & kinds) else 0
        return out

    def update(self, selection: Selection, achieved: Iterable[str], *, iteration: int | None = None) -> list[dict[str, Any]]:
        it = int(selection.iteration if iteration is None else iteration)
        events: list[dict[str, Any]] = []
        for t, reward in self.rewards_for(selection, achieved).items():
            fid = selection.families.get(t)
            if not fid:
                continue
            st = self._stat(fid)
            st["pulls"] = int(st.get("pulls", 0)) + 1
            st["rewards"] = int(st.get("rewards", 0)) + int(reward)
            st["alpha"] = float(st.get("alpha", 1.0)) + (1.0 if reward else 0.0)
            st["beta"] = float(st.get("beta", 1.0)) + (0.0 if reward else 1.0)
            st.setdefault("history", []).append([it, int(reward)])
            st["history"] = st["history"][-200:]
            events.append({"step_type": t, "family": fid, "reward": int(reward), "iteration": it,
                           "attempt": selection.attempt})
        if events:
            self.save()
        return events

    # -- collapse ----------------------------------------------------------
    def note_fingerprint(self, iteration: int, attempt_in_iter: int, fp: dict[str, Any]) -> dict[str, Any] | None:
        """Record an attempt-0 fingerprint; return the collapse override if it fires.

        The rule (plan B3): the last ``collapse_k`` first-attempts all voted the
        same family AND all hard-coded an identity quaternion. That is the exact
        shape run1 degenerated into.
        """
        if attempt_in_iter != 0 or self.collapse_k <= 0:
            return self._collapse
        self._fingerprints.append({
            "iteration": int(iteration),
            "family": str(fp.get("family") or "other"),
            "identity_quat": bool(fp.get("identity_quat")),
            "uses_plan_grasp": bool(fp.get("uses_plan_grasp")),
            "ast_hash": str(fp.get("ast_hash") or ""),
        })
        recent = self._fingerprints[-self.collapse_k:]
        collapsed = (
            len(recent) >= self.collapse_k
            and len({r["family"] for r in recent}) == 1
            and all(r["identity_quat"] for r in recent)
        )
        if collapsed:
            fam = recent[-1]["family"]
            if not self._collapse or self._collapse.get("family") != fam:
                logger.info("  Strategy collapse detected (%s x%d) — excluding it next iteration", fam, self.collapse_k)
            self._collapse = {"family": fam, "since_iteration": int(iteration), "k": self.collapse_k}
        elif self._collapse and self._collapse.get("family") not in {r["family"] for r in recent}:
            self._collapse = None
        self.save()
        return self._collapse

    @property
    def collapse(self) -> dict[str, Any] | None:
        return self._collapse

    # -- rendering ---------------------------------------------------------
    def render(self, selection: Selection) -> str:
        """The priority-0 block for the policy writer. Empty when nothing selected."""
        if selection.is_empty:
            return ""
        lines = [
            "## STRATEGY DIRECTIVE (HARD — priority 0, overrides all other guidance below)",
            "",
            "Use the physical strategy assigned to each phase below. Other iterations are "
            "exploring different strategies; duplicating them wastes the search. If the plan "
            "notes, a distilled lesson or the diagnoser tells you to avoid a primitive this "
            "block names, THIS BLOCK WINS.",
            "",
        ]
        forbids: list[str] = list(self.bank.common_forbids)
        for t in selection.step_types:
            fid = selection.families.get(t)
            fam = self.bank.get(fid) if fid else None
            if fam is None:
                continue
            lines.append(f"  {t.upper()} — {fam.directive}")
            for f in fam.forbids:
                if f not in forbids:
                    forbids.append(f)
        lines.append("")
        if forbids:
            lines.append("MUST NOT:")
            lines.extend(f"  - {f}" for f in forbids)
            lines.append("")
        for note in selection.notes:
            lines.append(note)
        if selection.notes:
            lines.append("")
        if self.show_evidence:
            ev = ", ".join(
                f"{selection.families[t]}={int(self._stat(selection.families[t])['rewards'])}"
                f"/{int(self._stat(selection.families[t])['pulls'])}"
                for t in selection.step_types if t in selection.families
            )
            if ev:
                lines.append(f"Evidence (family successes/uses so far): {ev}")
                lines.append("")
        return "\n".join(lines).rstrip() + "\n"
