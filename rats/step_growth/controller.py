"""The only object the lifelong loop talks to for the step-growth arm.

Hooks (all guarded by ``if self._step_growth is not None`` in the loop and
all exception-safe here):

  on_plan_ready        -> derive the milestone chain for this iteration
  on_attempt_start     -> (re)bind the low-level env, resolve the goal
  on_execution_start   -> open the oracle recording window
  on_attempt_executed  -> close window, judge steps, credit skills, persist
  on_iteration_end     -> step-level skill extraction on failed iterations
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable

from rats.step_growth import milestones as ms
from rats.step_growth.code_slices import (
    build_prefix,
    fallback_comment_blocks,
    reachable_learned_skills,
    step_code_blocks,
)
from rats.step_growth.config import StepGrowthConfig, load_config, step_growth_enabled
from rats.step_growth.oracle_recorder import StepOracleRecorder
from rats.step_growth.step_judge import AttemptVerdicts, judge

logger = logging.getLogger("rats.step_growth")


def _json_default(o: Any) -> Any:
    try:
        import numpy as np  # type: ignore

        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
    except Exception:
        pass
    if isinstance(o, Path):
        return str(o)
    return str(o)


class StepGrowthController:
    # ------------------------------------------------------------ creation
    @classmethod
    def maybe_create(
        cls,
        *,
        output_dir: str | Path,
        env_type: str,
        library_getter: Callable[[], Any],
    ) -> "StepGrowthController | None":
        if not step_growth_enabled():
            return None
        if env_type != "libero":
            logger.warning("RATS_STEP_GROWTH=1 but env_type=%s (LIBERO only) — arm disabled", env_type)
            return None
        cfg = load_config()
        ctrl = cls(cfg, output_dir=output_dir, library_getter=library_getter)
        logger.info(
            "Step-growth arm ENABLED (tier_policy=%s, extraction=%s, config=%s)",
            cfg.tier_policy, cfg.extraction_enabled, cfg.source_path or "<defaults>",
        )
        return ctrl

    def __init__(
        self,
        cfg: StepGrowthConfig,
        *,
        output_dir: str | Path,
        library_getter: Callable[[], Any],
        extractor: Any | None = None,
        register_listener: bool = True,
    ) -> None:
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self._library_getter = library_getter
        self.recorder = StepOracleRecorder(cfg)
        if register_listener:
            from rats.utils.execution_logger import register_policy_step_listener

            register_policy_step_listener(self.recorder.on_step_event)
        if extractor is None:
            from rats.agents.step_skill_extractor import StepSkillExtractor

            extractor = StepSkillExtractor(
                model=cfg.extraction_model,
                max_skill_lines=cfg.extraction_max_skill_lines,
                min_prefix_lines=cfg.extraction_min_prefix_lines,
                max_prefix_lines=cfg.extraction_max_prefix_lines,
            )
        self.extractor = extractor
        self._state_dir = self.output_dir / "step_growth"
        self._state_path = self._state_dir / "state.json"
        self._run_extractions = 0
        self._load_state()
        self._iter: dict[str, Any] = {}

    # -------------------------------------------------------------- state
    def _load_state(self) -> None:
        try:
            if self._state_path.exists():
                data = json.loads(self._state_path.read_text())
                self._run_extractions = int(data.get("run_extractions", 0) or 0)
        except Exception:
            self._run_extractions = 0

    def _save_state(self) -> None:
        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(json.dumps({
                "run_extractions": self._run_extractions,
                "updated_at": time.time(),
            }, indent=2))
        except Exception as exc:
            logger.debug("step-growth state save failed: %s", exc)

    @property
    def library(self) -> Any:
        return self._library_getter()

    # -------------------------------------------------------------- hooks
    def on_plan_ready(
        self,
        *,
        plan: dict[str, Any],
        task_proposal: dict[str, Any],
        scene_context: dict[str, Any],
        iteration_data: dict[str, Any],
    ) -> None:
        try:
            iteration = int(iteration_data.get("iteration", 0) or 0)
            self._iter = {
                "iteration": iteration,
                "plan_steps": list(plan.get("steps") or []),
                "bddl_path": scene_context.get("bddl_path"),
                "goal_state": None,
                "task_language": str(task_proposal.get("language") or ""),
                "activity_name": str(task_proposal.get("activity_name") or ""),
                "available_functions": list(scene_context.get("available_functions") or []),
                "attempts": {},
            }
            sg = iteration_data.setdefault("step_growth", {})
            sg.update({
                "enabled": True,
                "schema": "rats_step_growth_v1",
                "config": self.cfg.as_dict(),
                "bddl_path": str(self._iter["bddl_path"]) if self._iter["bddl_path"] else None,
                "attempts": {},
            })
        except Exception as exc:
            logger.warning("step-growth on_plan_ready failed (non-fatal): %s", exc)

    def on_attempt_start(
        self,
        low_level: Any,
        *,
        iteration: int,
        attempt: int,
        attempt_in_iter: int,
        turn_in_attempt: int,
        env_reset: bool,
    ) -> None:
        try:
            self.recorder.bind(low_level)
            if not self._iter:
                self._iter = {"iteration": iteration, "plan_steps": [], "attempts": {}, "goal_state": None}
            if self._iter.get("goal_state") is None:
                self._iter["goal_state"] = self._resolve_goal_state(low_level)
        except Exception as exc:
            logger.warning("step-growth on_attempt_start failed (non-fatal): %s", exc)

    def on_execution_start(
        self,
        *,
        iteration: int,
        attempt: int,
        attempt_in_iter: int,
        turn_in_attempt: int,
        env_reset: bool,
    ) -> None:
        try:
            self.recorder.begin_attempt(
                iteration=iteration, attempt=attempt, attempt_in_iter=attempt_in_iter,
                turn_in_attempt=turn_in_attempt, env_reset=env_reset,
            )
        except Exception as exc:
            logger.warning("step-growth on_execution_start failed (non-fatal): %s", exc)

    def on_attempt_executed(
        self,
        *,
        execution_result: dict[str, Any],
        plan: dict[str, Any],
        code: str,
        attempt: int,
        attempt_in_iter: int,
        turn_in_attempt: int,
        scene_context: dict[str, Any],
        task_proposal: dict[str, Any],
        iteration_data: dict[str, Any],
    ) -> dict[str, Any] | None:
        try:
            return self._on_attempt_executed(
                execution_result=execution_result, plan=plan, code=code, attempt=attempt,
                attempt_in_iter=attempt_in_iter, turn_in_attempt=turn_in_attempt,
                scene_context=scene_context, task_proposal=task_proposal,
                iteration_data=iteration_data,
            )
        except Exception as exc:
            logger.warning("step-growth on_attempt_executed failed (non-fatal): %s", exc, exc_info=True)
            self.recorder.discard()
            return None

    def on_iteration_end(
        self,
        *,
        success: bool,
        iteration_data: dict[str, Any],
        scene_context: dict[str, Any],
        task_proposal: dict[str, Any],
    ) -> dict[str, Any] | None:
        try:
            return self._on_iteration_end(
                success=success, iteration_data=iteration_data,
                scene_context=scene_context, task_proposal=task_proposal,
            )
        except Exception as exc:
            logger.warning("step-growth on_iteration_end failed (non-fatal): %s", exc, exc_info=True)
            return None
        finally:
            self._iter = {}
            self._save_state()

    # ---------------------------------------------------------- internals
    @staticmethod
    def _resolve_goal_state(low_level: Any) -> list[tuple[str, ...]] | None:
        pred_env_fn = getattr(low_level, "_predicate_env", None)
        pred_env = pred_env_fn() if callable(pred_env_fn) else None
        parsed = getattr(pred_env, "parsed_problem", None) if pred_env is not None else None
        if isinstance(parsed, dict) and parsed.get("goal_state"):
            gs = ms.parse_goal_state(parsed.get("goal_state"))
            return gs or None
        return None

    def _learned_names(self) -> set[str]:
        lib = self.library
        try:
            return {
                s["name"]
                for s in lib.get_full_skills_for_planner(include_deprecated=True)
                if not s.get("is_primitive", False)
            }
        except Exception:
            return set()

    def _credit(
        self,
        verdicts: AttemptVerdicts,
        blocks: dict[int, str],
        code: str,
        *,
        iteration: int,
        attempt: int,
        attempt_in_iter: int = 0,
    ) -> list[dict[str, Any]]:
        lib = self.library
        record_fn = getattr(lib, "record_step_usage", None)
        if not callable(record_fn):
            return []
        learned = self._learned_names()
        credited: list[dict[str, Any]] = []
        if not learned:
            return credited
        # Turns inside one attempt share the recorder window (env not reset),
        # so a step judged in turn 0 shows up again in turn 1's cumulative
        # record. Credit each (skill, step) once per attempt.
        done: set[tuple[str, int]] = self._iter.setdefault("credited_pairs", {}).setdefault(attempt_in_iter, set())
        for sv in verdicts.steps:
            if sv.verdict not in ("pass", "fail"):
                continue
            seg = blocks.get(sv.step_index, "")
            if not seg:
                continue
            names = [n for n in reachable_learned_skills(seg, code, learned) if (n, sv.step_index) not in done]
            if not names:
                continue
            done.update((n, sv.step_index) for n in names)
            events = record_fn(
                names, sv.verdict == "pass",
                iteration=iteration, attempt=attempt, step_id=sv.step_id,
                effects=list(verdicts.achieved), source="step_oracle",
            )
            for n in names:
                credited.append({"skill": n, "step_index": sv.step_index, "step_id": sv.step_id, "ok": sv.verdict == "pass"})
            for ev in events or []:
                credited.append({"lifecycle": ev})
        return credited

    def _sidecar_path(self, iteration: int, attempt: int) -> Path:
        return self.output_dir / f"iteration_{iteration:03d}" / f"attempt_{attempt:02d}" / "step_oracle.json"

    def _on_attempt_executed(
        self,
        *,
        execution_result: dict[str, Any],
        plan: dict[str, Any],
        code: str,
        attempt: int,
        attempt_in_iter: int,
        turn_in_attempt: int,
        scene_context: dict[str, Any],
        task_proposal: dict[str, Any],
        iteration_data: dict[str, Any],
    ) -> dict[str, Any] | None:
        artifacts = execution_result.get("artifacts") if isinstance(execution_result, dict) else None
        grounded = artifacts.get("grounded_state") if isinstance(artifacts, dict) else None
        record = self.recorder.end_attempt(grounded if isinstance(grounded, dict) else None)
        if record is None:
            return None
        iteration = int(self._iter.get("iteration") or iteration_data.get("iteration", 0) or 0)
        plan_steps = list(plan.get("steps") or []) if isinstance(plan, dict) else list(self._iter.get("plan_steps") or [])
        goal_state = self._iter.get("goal_state") or ms.goal_state_from_record(record)
        result = ms.evaluate(goal_state, record, self.cfg)
        verdicts = judge(result, record, plan_steps, self.cfg)
        blocks = step_code_blocks(code)
        slicing = "step_context"
        if not blocks:
            blocks = fallback_comment_blocks(code, len(plan_steps))
            slicing = "comment_headers" if blocks else "none"
        credited = self._credit(verdicts, blocks, code, iteration=iteration, attempt=attempt,
                                attempt_in_iter=attempt_in_iter)

        summary: dict[str, Any] = {
            "attempt": attempt,
            "attempt_in_iter": attempt_in_iter,
            "turn_in_attempt": turn_in_attempt,
            "markers_seen": bool(record.get("markers_seen")),
            "boundary_count": len(record.get("boundaries") or []),
            "snapshot_errors": int(record.get("snapshot_errors", 0) or 0),
            "goal_state": [list(g) for g in result.goal_state],
            "milestones": result.as_dict()["milestones"],
            "achieved": verdicts.achieved,
            "progress": verdicts.progress,
            "failure_events": verdicts.failure_events,
            "s_star": verdicts.s_star,
            "t_star": verdicts.t_star,
            "fail_step": verdicts.fail_step,
            "fail_reason": verdicts.fail_reason,
            "multi_chain_approx": verdicts.multi_chain_approx,
            "steps": [s.as_dict() for s in verdicts.steps],
            "code_slicing": slicing,
            "sliced_steps": sorted(blocks.keys()),
            "credited": credited,
            "picked_wrong": self._picked_wrong(record),
        }
        if self.cfg.persist_sidecar:
            path = self._sidecar_path(iteration, attempt)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w") as fh:
                    json.dump({"record": record, "evaluation": summary}, fh, indent=1, default=_json_default)
                summary["sidecar"] = str(path)
            except Exception as exc:
                logger.debug("step_oracle sidecar write failed: %s", exc)
        # Artifacts only: nothing downstream formats this into a prompt.
        if isinstance(artifacts, dict):
            artifacts["step_oracle"] = {
                "progress": verdicts.progress, "achieved": verdicts.achieved,
                "fail_step": verdicts.fail_step, "sidecar": summary.get("sidecar"),
            }
        sg = iteration_data.setdefault("step_growth", {"enabled": True, "attempts": {}})
        sg.setdefault("attempts", {})[str(attempt)] = summary
        self._iter.setdefault("attempts", {})[attempt] = {
            "code": code, "blocks": blocks, "verdicts": verdicts, "result": result,
        }
        logger.info(
            "  Step-oracle: progress %.2f (%s) S*=%s fail=%s(%s) credited=%d",
            verdicts.progress, ",".join(verdicts.achieved) or "-",
            verdicts.s_star, verdicts.fail_step, verdicts.fail_reason,
            len([c for c in credited if "skill" in c]),
        )
        return summary

    @staticmethod
    def _picked_wrong(record: dict[str, Any]) -> list[str]:
        for key in ("attempt_after",):
            snap = record.get(key)
            if isinstance(snap, dict) and snap.get("picked_wrong"):
                return list(snap["picked_wrong"])
        for bd in reversed(record.get("boundaries") or []):
            snap = bd.get("snapshot") if isinstance(bd, dict) else None
            if isinstance(snap, dict) and "picked_wrong" in snap:
                return list(snap.get("picked_wrong") or [])
        return []

    # ----------------------------------------------------------- extraction
    def _select_extraction_attempt(self) -> tuple[int, dict[str, Any]] | None:
        best: tuple[float, int, dict[str, Any]] | None = None
        for attempt, st in (self._iter.get("attempts") or {}).items():
            v: AttemptVerdicts = st["verdicts"]
            if not v.markers_seen or v.progress <= 0.0:
                continue
            passing = [i for i in v.pass_steps if st["blocks"].get(i)]
            if not passing:
                continue
            key = (v.progress, int(attempt))
            if best is None or key > (best[0], best[1]):
                best = (v.progress, int(attempt), st)
        if best is None:
            return None
        return best[1], best[2]

    def _on_iteration_end(
        self,
        *,
        success: bool,
        iteration_data: dict[str, Any],
        scene_context: dict[str, Any],
        task_proposal: dict[str, Any],
    ) -> dict[str, Any] | None:
        sg = iteration_data.setdefault("step_growth", {"enabled": True, "attempts": {}})
        iteration = int(self._iter.get("iteration") or iteration_data.get("iteration", 0) or 0)
        out: dict[str, Any] = {"attempted": False, "reason": None}
        sg["extraction"] = out
        if not self.cfg.extraction_enabled:
            out["reason"] = "disabled"
            return out
        if self.cfg.extraction_only_on_failed_iteration and success:
            out["reason"] = "iteration_succeeded"
            return out
        if self._run_extractions >= self.cfg.extraction_max_per_run:
            out["reason"] = "run_cap"
            return out
        sel = self._select_extraction_attempt()
        if sel is None:
            out["reason"] = "no_passing_step"
            return out
        attempt, st = sel
        v: AttemptVerdicts = st["verdicts"]
        pass_order = [s.step_index for s in v.steps if s.verdict == "pass"]
        prefix = build_prefix(st["blocks"], pass_order, full_code=st["code"])
        achieved = list(v.achieved)
        lib = self.library
        try:
            existing = lib.get_full_skills_for_planner()
        except Exception:
            existing = []
        available = list(scene_context.get("available_functions") or self._iter.get("available_functions") or [])
        task_language = str(task_proposal.get("language") or self._iter.get("task_language") or "")
        out.update({
            "attempted": True, "attempt": attempt, "pass_steps": pass_order,
            "achieved": achieved, "prefix_lines": len(prefix.splitlines()),
        })
        res = self.extractor.extract(
            prefix_code=prefix, achieved=achieved, task_language=task_language,
            existing_skills=existing, available_functions=available,
        )
        out["llm_called"] = bool(res.get("llm_called"))
        out["llm_reason"] = res.get("llm_reason")
        skill = res.get("skill")
        if not skill:
            out["rejected_reason"] = res.get("rejected_reason")
            out["candidate_name"] = res.get("candidate_name")
            return out
        candidate = self._to_candidate(skill, iteration=iteration, attempt=attempt, pass_steps=pass_order,
                                       achieved=achieved, task_proposal=task_proposal)
        out["proposed_name"] = candidate["name"]
        out["strategy_tag"] = candidate.get("strategy_tag")
        added = False
        try:
            added = bool(lib.add_skill(candidate, available_functions=available))
        except Exception as exc:
            out["rejected_reason"] = f"add_skill_error:{str(exc)[:120]}"
            return out
        if not added:
            out["rejected_reason"] = "duplicate_or_invalid"
            out["stored_name"] = None
            return out
        stored = candidate.get("name")
        out["stored_name"] = stored
        self._run_extractions += 1
        record_fn = getattr(lib, "record_step_usage", None)
        if callable(record_fn):
            try:
                record_fn([stored], True, iteration=iteration, attempt=attempt,
                          step_id="extraction", effects=achieved, source="step_skill_extractor")
            except Exception:
                pass
        iteration_data.setdefault("step_skills_learned", []).append({
            "name": stored, "strategy_tag": candidate.get("strategy_tag"),
            "achieved": achieved, "attempt": attempt, "learned_iteration": iteration,
        })
        logger.info("  Step-level skill added: %s (from attempt %d, %s)", stored, attempt, ",".join(achieved))
        return out

    def _to_candidate(
        self,
        skill: dict[str, Any],
        *,
        iteration: int,
        attempt: int,
        pass_steps: list[int],
        achieved: list[str],
        task_proposal: dict[str, Any],
    ) -> dict[str, Any]:
        name = str(skill.get("name") or "step_skill")
        code = str(skill.get("code") or "")
        tag = str(skill.get("strategy_tag") or "other")
        # Keep strategy variants distinct: same name, different tag -> suffix.
        try:
            existing = {s.get("name"): s for s in self.library.get_full_skills_for_planner(include_deprecated=True)}
        except Exception:
            existing = {}
        if name in existing and str(existing[name].get("strategy_tag") or "") not in ("", tag):
            new_name = f"{name}_{tag}"
            code = re.sub(rf"\bdef\s+{re.escape(name)}\s*\(", f"def {new_name}(", code, count=1)
            name = new_name
        desc = str(skill.get("description") or "")
        if tag and tag != "other" and not desc.startswith("[strategy:"):
            desc = f"[strategy: {tag}] {desc}"
        return {
            "name": name,
            "description": desc,
            "code": code,
            "api_primitives_used": list(skill.get("api_primitives_used") or []),
            "preconditions": list(skill.get("preconditions") or []),
            "effects": list(skill.get("effects") or []),
            "source_task": str(task_proposal.get("activity_name") or task_proposal.get("language") or ""),
            "learned_iteration": iteration,
            "params": list(skill.get("params") or []),
            "returns": skill.get("returns") or {"type": "", "shape": "", "description": ""},
            "usage_example": str(skill.get("usage_example") or ""),
            "extraction_rationale": str(skill.get("extraction_rationale") or ""),
            "strategy_tag": tag,
            "credit_source": "step_oracle",
            "evidence": {
                "iteration": iteration, "attempt": attempt,
                "pass_steps": list(pass_steps), "achieved": list(achieved),
            },
        }
