"""What an attempt's code actually *is*, structurally — for measuring collapse.

Section B3 of the step-growth plan. In play_qwen38_reduced/run1, 45 of 50
attempt-0 programs were the same recipe: a hard-coded identity quaternion plus
a mask-centroid position, with ``plan_grasp`` explicitly avoided. The run
looked like 50 experiments but was one experiment repeated, so the library
learned nothing from 47 of them.

A fingerprint is cheap, offline-reproducible, and reads only code the agent
itself wrote (no simulator state):

  api_calls          which env primitives the code calls (Counter)
  skill_calls        which learned skills it calls
  identity_quat      does it hard-code [1,0,0,0] / [0,0,0,1] as an orientation
  uses_plan_grasp    does it ask a grasp planner for the pose
  step_markers       how many `step_context` blocks it declares
  ast_hash           structure hash: variable names normalised away, constants
                     reduced to their type, known API/skill names KEPT — so a
                     rename is the same fingerprint but a different pipeline is not
  family             coarse strategy vote used by the collapse rule
"""

from __future__ import annotations

import ast
import hashlib
import math
import re
from collections import Counter
from typing import Any, Iterable

# [1, 0, 0, 0] or [0, 0, 0, 1] (and the np.array / tuple spellings), integer or
# float literals, any spacing. This is the "I did not ask perception for an
# orientation" tell.
_IDENTITY_QUAT_RE = re.compile(
    r"[\[\(]\s*"
    r"(?:1(?:\.0*)?\s*,\s*0(?:\.0*)?\s*,\s*0(?:\.0*)?\s*,\s*0(?:\.0*)?"
    r"|0(?:\.0*)?\s*,\s*0(?:\.0*)?\s*,\s*0(?:\.0*)?\s*,\s*1(?:\.0*)?)"
    r"\s*[\]\)]"
)
_STEP_MARKER_RE = re.compile(r"step_context\s*\(")

GRASP_PLANNERS = ("plan_grasp", "plan_grasp_from_point_clouds")

# Names that survive AST normalisation even when they are not in the env's
# available_functions (a fingerprint must not depend on the env to be readable).
_ALWAYS_KEEP = frozenset({
    "np", "numpy", "math", "step_context", "RESULT", "main", "env",
    "range", "len", "max", "min", "sorted", "abs", "float", "int", "list", "dict",
})


def _called_names(code: str) -> list[str]:
    """Every ``name(`` in the code (attribute calls excluded), in order."""
    return re.findall(r"(?<![\w.])([A-Za-z_][A-Za-z_0-9]*)\s*\(", code or "")


class _Normalizer(ast.NodeTransformer):
    """Erase identifiers and literal values, keep structure and known names."""

    def __init__(self, keep: set[str]) -> None:
        self.keep = keep

    def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802
        if node.id not in self.keep:
            node.id = "_"
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = "_"
        node.annotation = None
        return node

    def visit_keyword(self, node: ast.keyword) -> ast.AST:  # noqa: N802
        self.generic_visit(node)
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:  # noqa: N802
        node.value = type(node.value).__name__
        node.kind = None
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:  # noqa: N802
        if node.name not in self.keep:
            node.name = "_"
        node.decorator_list = []
        self.generic_visit(node)
        return node


def ast_hash(code: str, keep: Iterable[str] = ()) -> str:
    """Structure hash of ``code``; ``""`` on a syntax error."""
    keep_set = set(_ALWAYS_KEEP) | {k for k in keep if k}
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return ""
    try:
        tree = _Normalizer(keep_set).visit(tree)
        ast.fix_missing_locations(tree)
        dumped = ast.dump(tree, annotate_fields=False, include_attributes=False)
    except Exception:
        return ""
    return hashlib.sha1(dumped.encode("utf-8")).hexdigest()[:16]


def vote_family(
    *,
    api_calls: Counter,
    identity_quat: bool,
    uses_plan_grasp: bool,
    bank: Any | None = None,
) -> str:
    """Coarse strategy vote. Deliberately few buckets: this drives the
    collapse rule, which is about "everything looks the same", not about
    telling two GraspNet variants apart."""
    if uses_plan_grasp:
        return "graspnet"
    if api_calls.get("get_oriented_bounding_box_from_3d_points"):
        return "obb_yaw"
    if api_calls.get("point_prompt_molmo"):
        return "molmo"
    if identity_quat:
        return "handbuilt_identity"
    return "other"


def fingerprint(
    code: str,
    available_functions: Iterable[str] | None = None,
    learned_skill_names: Iterable[str] | None = None,
    bank: Any | None = None,
) -> dict[str, Any]:
    code = code or ""
    avail = set(available_functions or ())
    learned = set(learned_skill_names or ())
    called = _called_names(code)
    api_calls = Counter(n for n in called if n in avail)
    skill_calls = Counter(n for n in called if n in learned)
    identity_quat = bool(_IDENTITY_QUAT_RE.search(code))
    uses_plan_grasp = any(api_calls.get(p) for p in GRASP_PLANNERS) or any(
        p in called for p in GRASP_PLANNERS
    )
    fp: dict[str, Any] = {
        "api_calls": dict(sorted(api_calls.items())),
        "skill_calls": dict(sorted(skill_calls.items())),
        "identity_quat": identity_quat,
        "uses_plan_grasp": bool(uses_plan_grasp),
        "step_markers": len(_STEP_MARKER_RE.findall(code)),
        "lines": len([ln for ln in code.splitlines() if ln.strip()]),
        "ast_hash": ast_hash(code, keep=avail | learned),
    }
    fp["family"] = vote_family(
        api_calls=api_calls, identity_quat=identity_quat,
        uses_plan_grasp=fp["uses_plan_grasp"], bank=bank,
    )
    return fp


def summarize(fps: list[dict[str, Any]]) -> dict[str, Any]:
    """Run-level diversity metrics over a list of fingerprints."""
    n = len(fps)
    if not n:
        return {
            "n": 0, "unique_ast_ratio": 0.0, "family_entropy": 0.0, "families": {},
            "identity_quat_frac": 0.0, "plan_grasp_frac": 0.0, "marker_coverage": 0.0,
        }
    hashes = {f.get("ast_hash") or "" for f in fps}
    families = Counter(str(f.get("family") or "other") for f in fps)
    entropy = 0.0
    for count in families.values():
        p = count / n
        if p > 0:
            entropy -= p * math.log(p, 2)
    return {
        "n": n,
        "unique_ast_ratio": len(hashes) / float(n),
        "family_entropy": round(entropy, 4),
        "families": dict(sorted(families.items())),
        "identity_quat_frac": sum(1 for f in fps if f.get("identity_quat")) / float(n),
        "plan_grasp_frac": sum(1 for f in fps if f.get("uses_plan_grasp")) / float(n),
        "marker_coverage": sum(1 for f in fps if int(f.get("step_markers") or 0) > 0) / float(n),
    }


def detect_collapse(fps: list[dict[str, Any]], k: int = 5) -> str | None:
    """The family id if the last ``k`` fingerprints are one family AND all
    hard-code an orientation; otherwise ``None``."""
    if k <= 0 or len(fps) < k:
        return None
    recent = fps[-k:]
    if len({str(f.get("family") or "other") for f in recent}) != 1:
        return None
    if not all(f.get("identity_quat") for f in recent):
        return None
    return str(recent[-1].get("family") or "other")
