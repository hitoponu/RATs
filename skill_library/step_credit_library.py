"""SkillLibrary subclass that tracks STEP-level reliability.

Task-level counters (``usage_count`` / ``success_count``) keep being written
by the existing loop paths exactly as before. This subclass adds parallel
``step_usage_count`` / ``step_success_count`` counters fed by the step-growth
oracle judge, and (when ``tier_policy == "step"``) lets those counters drive
tier promotion and the planner's Wilson ranking.

Bit-identical to ``SkillLibrary`` when ``record_step_usage`` is never called.
"""

from __future__ import annotations

import logging
from typing import Any

from skill_library.library import SkillLibrary, wilson_lower_bound

logger = logging.getLogger("rats.skill_library.step_credit")

STEP_EVENT_CAP = 20


class StepCreditSkillLibrary(SkillLibrary):
    def __init__(
        self,
        storage_path: str = "skill_library/skills.json",
        *,
        tier_policy: str = "step",
        promote_min_step_uses: int = 3,
        promote_min_step_sr: float = 0.6,
        deprecate_min_step_uses: int = 8,
        deprecate_max_step_sr: float = 0.2,
    ) -> None:
        self.tier_policy = tier_policy
        self.promote_min_step_uses = int(promote_min_step_uses)
        self.promote_min_step_sr = float(promote_min_step_sr)
        self.deprecate_min_step_uses = int(deprecate_min_step_uses)
        self.deprecate_max_step_sr = float(deprecate_max_step_sr)
        super().__init__(storage_path=storage_path)

    # ------------------------------------------------------------ reliability
    def _step_counts(self, skill: dict[str, Any]) -> tuple[int, int]:
        return int(skill.get("step_success_count", 0) or 0), int(skill.get("step_usage_count", 0) or 0)

    def _wilson(self, skill: dict[str, Any]) -> float:  # type: ignore[override]
        if self.tier_policy == "step":
            sc, n = self._step_counts(skill)
            if n > 0:
                return wilson_lower_bound(sc, n)
        return wilson_lower_bound(
            int(skill.get("success_count", 0)), int(skill.get("usage_count", 0)),
        )

    def _update_tier(self, skill: dict[str, Any]) -> None:
        super()._update_tier(skill)
        if self.tier_policy != "step" or skill.get("is_primitive", False):
            return
        sc, n = self._step_counts(skill)
        if n <= 0:
            return
        sr = sc / n
        current = skill.get("tier", "experimental")
        if (
            current == "experimental"
            and n >= self.promote_min_step_uses
            and sr >= self.promote_min_step_sr
        ):
            skill["tier"] = "verified"
            skill["tier_source"] = "step_credit"
            logger.info(
                f"  Skill promoted verified (step credit): {skill['name']} "
                f"(step SR={sr:.2f}, n={n})"
            )
            return
        if (
            current == "experimental"
            and n >= self.deprecate_min_step_uses
            and sr <= self.deprecate_max_step_sr
        ):
            sname = skill.get("name")
            depended_on = sname and any(
                sname in (other.get("dependent_skills") or [])
                and other.get("tier") != "deprecated"
                and not other.get("is_primitive")
                for other in self._skills
            )
            if depended_on:
                return
            skill["tier"] = "deprecated"
            skill["tier_source"] = "step_credit"
            logger.info(
                f"  Skill deprecated (step credit): {skill['name']} (step SR={sr:.2f}, n={n})"
            )

    # ---------------------------------------------------------------- record
    def record_step_usage(
        self,
        skill_names: list[str],
        success: bool,
        *,
        iteration: int | None = None,
        attempt: int | None = None,
        step_id: str | None = None,
        effects: list[str] | None = None,
        source: str = "step_oracle",
    ) -> list[dict[str, Any]]:
        """Step-level counterpart of ``record_usage``; primitives ignored."""
        if not skill_names:
            return []
        seen: set[str] = set()
        changed = False
        events: list[dict[str, Any]] = []
        for name in skill_names:
            if name in seen:
                continue
            seen.add(name)
            for s in self._skills:
                if s.get("name") != name or s.get("is_primitive", False):
                    continue
                old_tier = s.get("tier", "experimental")
                s["step_usage_count"] = int(s.get("step_usage_count", 0) or 0) + 1
                if success:
                    s["step_success_count"] = int(s.get("step_success_count", 0) or 0) + 1
                else:
                    s.setdefault("step_success_count", 0)
                n = s["step_usage_count"]
                s["step_success_rate"] = s["step_success_count"] / n if n else 0.0
                ev_list = s.setdefault("step_events", [])
                ev_list.append({
                    "iteration": iteration, "attempt": attempt, "step_id": step_id,
                    "effects": list(effects or []), "ok": bool(success),
                })
                if len(ev_list) > STEP_EVENT_CAP:
                    del ev_list[:-STEP_EVENT_CAP]
                self._update_tier(s)
                new_tier = s.get("tier")
                if new_tier != old_tier:
                    if new_tier == "deprecated":
                        if iteration is not None:
                            s["deprecated_at_iteration"] = int(iteration)
                        s["deprecated_by"] = source
                        s["deprecated_reason"] = (
                            f"step_usage_count={n}, step_success_count={s['step_success_count']}"
                        )
                    events.append({
                        "iteration": iteration, "action": f"tier:{old_tier}->{new_tier}",
                        "skill": name, "source": source,
                        "reason": f"step {s['step_success_count']}/{n}",
                    })
                changed = True
                break
        if changed:
            self._save()
        return events

    # -------------------------------------------------------------- readers
    def get_full_skills_for_planner(self, include_deprecated: bool = False) -> list[dict[str, Any]]:
        out = super().get_full_skills_for_planner(include_deprecated=include_deprecated)
        by_name = {s.get("name"): s for s in self._skills}
        for entry in out:
            s = by_name.get(entry.get("name")) or {}
            sc, n = self._step_counts(s)
            entry["step_usage_count"] = n
            entry["step_success_count"] = sc
            entry["step_wilson"] = wilson_lower_bound(sc, n) if n else 0.0
            if s.get("strategy_tag"):
                entry["strategy_tag"] = s.get("strategy_tag")
        return out

    def step_stats(self) -> dict[str, dict[str, Any]]:
        stats: dict[str, dict[str, Any]] = {}
        for s in self._skills:
            if s.get("is_primitive"):
                continue
            sc, n = self._step_counts(s)
            if n == 0 and not s.get("credit_source"):
                continue
            stats[str(s.get("name"))] = {
                "step_usage_count": n, "step_success_count": sc,
                "tier": s.get("tier"), "credit_source": s.get("credit_source"),
            }
        return stats


def make_skill_library(storage_path: str, cfg: Any | None = None) -> SkillLibrary:
    """Pick the library class for this run: step-credit when the arm is on."""
    try:
        from rats.step_growth.config import load_config, step_growth_enabled
    except Exception:
        return SkillLibrary(storage_path=storage_path)
    if not step_growth_enabled():
        return SkillLibrary(storage_path=storage_path)
    cfg = cfg or load_config()
    return StepCreditSkillLibrary(
        storage_path=storage_path,
        tier_policy=getattr(cfg, "tier_policy", "step"),
        promote_min_step_uses=getattr(cfg, "promote_min_step_uses", 3),
        promote_min_step_sr=getattr(cfg, "promote_min_step_sr", 0.6),
        deprecate_min_step_uses=getattr(cfg, "deprecate_min_step_uses", 8),
        deprecate_max_step_sr=getattr(cfg, "deprecate_max_step_sr", 0.2),
    )
