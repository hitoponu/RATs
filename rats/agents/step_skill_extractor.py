"""Step-level skill extractor: distil ONE helper from the passing prefix of a
FAILED attempt, gated by a boundary rule (object_name in, perception ->
planning -> execution inside).

The LLM is asked only with vocabulary labels of the achieved milestones
(``grasped(x)``, ``lifted(x)`` ...) — no raw oracle state is formatted into
the prompt.
"""

from __future__ import annotations

import ast
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("rats.step_skill_extractor")

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "rats" / "prompts" / "step_skill_extractor.txt"

# Primitives that constitute "the function localized/planned by itself".
PERCEPTION_PLANNING_PRIMITIVES = {
    "get_object_pose",
    "segment_sam3_text_prompt",
    "segment_sam3_point_prompt",
    "get_sam3_mask",
    "point_prompt_molmo",
    "get_object_3d_points_and_masks_from_language",
    "plan_grasp",
    "plan_grasp_from_point_clouds",
    "sample_grasp_pose",
    "get_oriented_bounding_box_from_3d_points",
    "inspect_at_wrist",
    "localize_object",
    "segment_object",
}
_PERCEPTION_PREFIXES = ("segment_", "localize_", "plan_grasp", "point_prompt", "get_object", "sample_grasp")

# Parameter names that mean "the caller computed the geometry".
_POSE_PARAM_TOKENS = {"pos", "quat", "pose", "grasp", "position", "orientation", "point", "points", "wxyz", "xyzw"}
_POSE_PARAM_ALLOW = ("offset", "height", "delta", "dz", "dx", "dy", "clearance", "margin", "radius", "tol")

# Milestone kinds that mean the skill claims a physical effect.
EFFECT_KINDS = {"grasped", "lifted", "near", "placed", "open", "closed", "turnon", "turnoff"}

_SNAKE_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


def _perception_set(available_functions: list[str] | set[str] | None) -> set[str]:
    avail = set(available_functions or [])
    if not avail:
        return set(PERCEPTION_PLANNING_PRIMITIVES)
    out = {f for f in avail if f in PERCEPTION_PLANNING_PRIMITIVES}
    out |= {f for f in avail if f.startswith(_PERCEPTION_PREFIXES)}
    return out


def _called_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


def lint_skill_code(
    code: str,
    *,
    available_functions: list[str] | set[str] | None,
    achieved_kinds: set[str] | None,
    max_lines: int = 80,
) -> str | None:
    """Return a rejection reason or None when the candidate passes the boundary rule."""
    if not code or not code.strip():
        return "empty_code"
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"syntax_error:{exc.msg}"
    defs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    others = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Import, ast.ImportFrom))]
    if len(defs) != 1:
        return f"expected_one_function:{len(defs)}"
    if others:
        return "top_level_statements_outside_function"
    fn = defs[0]
    if not _SNAKE_RE.match(fn.name):
        return f"name_not_snake_case:{fn.name}"
    n_lines = len([ln for ln in code.splitlines() if ln.strip()])
    if n_lines > max_lines:
        return f"too_long:{n_lines}>{max_lines}"
    params = [a.arg for a in fn.args.args] + [a.arg for a in fn.args.kwonlyargs]
    if not params:
        return "no_parameters"
    first = params[0]
    if not (first.endswith("_name") or first.endswith("_prompt") or first in ("object_name", "target", "label")):
        return f"first_param_not_object_name:{first}"
    called = _called_names(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.While):
            test = node.test
            if isinstance(test, ast.Constant) and test.value is True:
                return "while_true"
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in ("env", "low_level_env"):
            return "env_method_call"
    perception = _perception_set(available_functions)
    uses_perception = bool(called & perception) if perception else True
    for p in params[1:]:
        tokens = set(p.lower().split("_"))
        if tokens & _POSE_PARAM_TOKENS and not any(a in p.lower() for a in _POSE_PARAM_ALLOW):
            if not uses_perception:
                return f"pose_in_wrapper:{p}"
    if achieved_kinds and (achieved_kinds & EFFECT_KINDS) and perception and not uses_perception:
        return "effect_skill_without_perception"
    return None


class StepSkillExtractor:
    def __init__(
        self,
        *,
        model: str | None = None,
        prompt_path: str | os.PathLike[str] | None = None,
        llm_query: Callable[..., dict[str, Any]] | None = None,
        max_skill_lines: int = 80,
        min_prefix_lines: int = 3,
        max_prefix_lines: int = 200,
    ) -> None:
        self.model = model or os.getenv("RATS_STEP_SKILL_EXTRACTOR_MODEL") or None
        self.prompt_path = Path(prompt_path) if prompt_path else _PROMPT_PATH
        self._llm_query = llm_query
        self.max_skill_lines = int(max_skill_lines)
        self.min_prefix_lines = int(min_prefix_lines)
        self.max_prefix_lines = int(max_prefix_lines)

    # ------------------------------------------------------------ plumbing
    def _query(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if self._llm_query is not None:
            return self._llm_query(system_prompt, user_prompt, model=self.model)
        from rats.agents import base_agent  # lazy: keeps tests free of LLM deps

        kwargs: dict[str, Any] = {}
        if self.model:
            kwargs["model"] = self.model
        return base_agent.query_llm_json(system_prompt, user_prompt, **kwargs)

    def _template(self) -> str:
        return self.prompt_path.read_text()

    @staticmethod
    def _prefix_gate(prefix_code: str, min_lines: int, max_lines: int) -> str | None:
        lines = [ln for ln in (prefix_code or "").splitlines() if ln.strip() and not ln.strip().startswith("#")]
        if len(lines) < min_lines:
            return f"prefix_too_short:{len(lines)}"
        if len(lines) > max_lines:
            return f"prefix_too_long:{len(lines)}"
        try:
            ast.parse(prefix_code)
        except SyntaxError:
            return "prefix_not_parseable"
        return None

    # -------------------------------------------------------------- public
    def extract(
        self,
        *,
        prefix_code: str,
        achieved: list[str],
        task_language: str,
        existing_skills: list[dict[str, Any]] | None,
        available_functions: list[str] | set[str] | None,
    ) -> dict[str, Any]:
        """Returns ``{"skill": dict|None, "rejected_reason": str|None, "llm_reason": str, "raw": ...}``."""
        gate = self._prefix_gate(prefix_code, self.min_prefix_lines, self.max_prefix_lines)
        if gate:
            return {"skill": None, "rejected_reason": gate, "llm_reason": "", "raw": None, "llm_called": False}

        existing_block = "(none)"
        names: list[str] = []
        if existing_skills:
            entries = []
            for s in existing_skills[:40]:
                if s.get("is_primitive"):
                    continue
                name = s.get("name", "")
                if not name:
                    continue
                names.append(name)
                entries.append(f"- {name}: {(s.get('description') or '')[:160]}")
            if entries:
                existing_block = "\n".join(entries)
        avail_list = sorted(set(available_functions or []))
        prompt = (
            self._template()
            .replace("{max_lines}", str(self.max_skill_lines))
            .replace("{task_description}", str(task_language or ""))
            .replace("{achieved_effects}", ", ".join(achieved) if achieved else "(none)")
            .replace("{available_functions}", ", ".join(avail_list) if avail_list else "(see API documentation)")
            .replace("{existing_skills}", existing_block)
            .replace("{code}", prefix_code)
        )
        system_prompt = (
            "You extract reusable robot skills from partially successful code. "
            "Respond only in valid JSON."
        )
        try:
            raw = self._query(system_prompt, prompt)
        except Exception as exc:
            logger.warning("step skill extraction LLM call failed: %s", exc)
            return {"skill": None, "rejected_reason": f"llm_error:{str(exc)[:120]}", "llm_reason": "", "raw": None, "llm_called": True}

        skill = raw.get("skill") if isinstance(raw, dict) else None
        llm_reason = str(raw.get("reason", "")) if isinstance(raw, dict) else ""
        if not isinstance(skill, dict):
            return {"skill": None, "rejected_reason": "llm_returned_null", "llm_reason": llm_reason, "raw": raw, "llm_called": True}
        code = str(skill.get("code") or "")
        achieved_kinds = {a.split("(")[0] for a in achieved}
        reason = lint_skill_code(
            code,
            available_functions=available_functions,
            achieved_kinds=achieved_kinds,
            max_lines=self.max_skill_lines,
        )
        if reason:
            return {"skill": None, "rejected_reason": reason, "llm_reason": llm_reason, "raw": raw, "llm_called": True, "candidate_name": skill.get("name")}
        # Perception-verify anti-pattern gate shared with feedback_generator.
        try:
            from rats.agents.policy_quality_checker import PolicyQualityChecker

            issues = PolicyQualityChecker._check_perception_verify_consumed(code)
            if issues:
                return {"skill": None, "rejected_reason": f"perception_verify_unconsumed:{issues[0][:120]}", "llm_reason": llm_reason, "raw": raw, "llm_called": True, "candidate_name": skill.get("name")}
        except Exception:
            pass
        # Learned-skill calls in the candidate must be names we know.
        skill.setdefault("params", [])
        skill.setdefault("returns", {"type": "", "shape": "", "description": ""})
        skill.setdefault("api_primitives_used", [])
        skill.setdefault("preconditions", [])
        skill.setdefault("effects", [])
        skill.setdefault("usage_example", "")
        skill.setdefault("extraction_rationale", "")
        skill["strategy_tag"] = str(skill.get("strategy_tag") or "other")
        skill["credit_source"] = "step_oracle"
        return {"skill": skill, "rejected_reason": None, "llm_reason": llm_reason, "raw": raw, "llm_called": True}
