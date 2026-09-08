"""Slice policy code by ``with step_context(...)`` markers and find the
learned skills reachable from a slice.

Mirrors ``PerStepVerifier._step_code_blocks`` (rats/agents/per_step_verifier.py)
and the reachability helpers at the top of rats/loop/lifelong_loop.py,
duplicated here so this package does not import the whole loop.
"""

from __future__ import annotations

import ast
import re
from typing import Any

_MARKER_RE = re.compile(
    r"^(?P<indent>[ \t]*)with\s+step_context\(\s*(?P<args>[^)]*)\)\s*:\s*$",
    re.MULTILINE,
)
_ID_RE = re.compile(r"""['\"]([^'\"]+)['\"]""")
_INDEX_RE = re.compile(r"step_index\s*=\s*(\d+)")


def _dedent_body(body: str, with_indent: str) -> str:
    lines = body.splitlines()
    non_blank = [ln for ln in lines if ln.strip()]
    if not non_blank:
        return ""
    prefix_len = len(with_indent) + 4
    observed = min(len(ln) - len(ln.lstrip(" \t")) for ln in non_blank)
    prefix_len = min(prefix_len, observed) if observed > len(with_indent) else observed
    out: list[str] = []
    for ln in lines:
        if not ln.strip():
            out.append("")
            continue
        out.append(ln[prefix_len:] if len(ln) >= prefix_len and not ln[:prefix_len].strip() else ln.lstrip(" \t"))
    return "\n".join(out).strip("\n") + "\n"


def step_code_blocks(code: str) -> dict[int, str]:
    """``{step_index: dedented body}`` for every ``with step_context`` block.

    ``step_index=N`` wins; otherwise the digits of the quoted step id are
    read as a 1-based index. Blocks sharing an index are concatenated.
    """
    if not code:
        return {}
    matches = list(_MARKER_RE.finditer(code))
    blocks: dict[int, str] = {}
    for m in matches:
        args = m.group("args") or ""
        indent = m.group("indent") or ""
        step_index: int | None = None
        mi = _INDEX_RE.search(args)
        if mi:
            step_index = int(mi.group(1))
        else:
            mid = _ID_RE.search(args)
            if mid:
                digits = "".join(ch for ch in mid.group(1) if ch.isdigit())
                if digits:
                    step_index = max(int(digits) - 1, 0)
        if step_index is None:
            continue
        body_start = code.find("\n", m.end())
        if body_start < 0:
            continue
        body_start += 1
        with_indent_len = len(indent.expandtabs(4))
        i = body_start
        body_end = len(code)
        while i < len(code):
            nl = code.find("\n", i)
            line = code[i:nl] if nl >= 0 else code[i:]
            stripped = line.strip()
            if stripped:
                leading = len(line) - len(line.lstrip(" \t"))
                if len(line[:leading].expandtabs(4)) <= with_indent_len:
                    body_end = i
                    break
            if nl < 0:
                break
            i = nl + 1
        body = _dedent_body(code[body_start:body_end], indent)
        if body.strip():
            blocks[step_index] = (blocks[step_index].rstrip() + "\n\n" + body) if step_index in blocks else body
    return blocks


def local_function_defs(code: str) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    return {
        n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _call_names(statements: list[ast.stmt]) -> list[str]:
    calls: list[str] = []

    class _C(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            self.generic_visit(node)

        def visit_FunctionDef(self, node):  # noqa: N802
            return

        def visit_AsyncFunctionDef(self, node):  # noqa: N802
            return

        def visit_ClassDef(self, node):  # noqa: N802
            return

        def visit_Lambda(self, node):  # noqa: N802
            return

    c = _C()
    for st in statements:
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        c.visit(st)
    return calls


def _regex_calls(segment: str, names: set[str]) -> list[str]:
    found: list[str] = []
    for m in re.finditer(r"(?<![\w.])([A-Za-z_][A-Za-z_0-9]*)\s*\(", segment):
        n = m.group(1)
        if n in names and n not in found:
            found.append(n)
    return found


def reachable_learned_skills(segment: str, full_code: str, learned_names: set[str]) -> list[str]:
    """Learned skills called from ``segment`` directly or via local helpers of ``full_code``."""
    if not segment or not learned_names:
        return []
    try:
        tree = ast.parse(segment)
    except SyntaxError:
        return _regex_calls(segment, learned_names)
    helpers = local_function_defs(full_code)
    queue = _call_names(tree.body)
    visited: set[str] = set()
    found: list[str] = []
    while queue:
        n = queue.pop(0)
        if n in learned_names:
            if n not in found:
                found.append(n)
            continue
        h = helpers.get(n)
        if h is None or n in visited:
            continue
        visited.add(n)
        queue.extend(_call_names(h.body))
    return found


def all_called_names(code: str) -> set[str]:
    return set(re.findall(r"(?<![\w.])([A-Za-z_][A-Za-z_0-9]*)\s*\(", code or ""))


def build_prefix(blocks: dict[int, str], pass_indices: list[int], full_code: str = "") -> str:
    """Concatenate the passed steps' bodies (in the given order) plus the local
    helper definitions they rely on, so the prefix is self-describing."""
    parts: list[str] = []
    helpers = local_function_defs(full_code) if full_code else {}
    used: set[str] = set()
    for idx in pass_indices:
        body = blocks.get(idx)
        if not body:
            continue
        parts.append(f"# [step {idx + 1}]\n{body.rstrip()}\n")
        used |= all_called_names(body)
    helper_src: list[str] = []
    if helpers and used:
        # Include local helper defs reachable from the used names.
        queue = [n for n in used if n in helpers]
        seen: set[str] = set()
        while queue:
            n = queue.pop(0)
            if n in seen:
                continue
            seen.add(n)
            node = helpers[n]
            try:
                src = ast.get_source_segment(full_code, node)
            except Exception:
                src = None
            if src:
                helper_src.append(src.rstrip() + "\n")
                for callee in _call_names(node.body):
                    if callee in helpers and callee not in seen:
                        queue.append(callee)
    return "\n".join(helper_src + parts).strip() + "\n" if (helper_src or parts) else ""


def fallback_comment_blocks(code: str, step_count: int) -> dict[int, str]:
    """``# step-N:`` comment-header slicing when no ``step_context`` markers exist.

    Uses feedback_generator._extract_step_segment when importable; otherwise a
    minimal local matcher.
    """
    blocks: dict[int, str] = {}
    try:
        from rats.agents.feedback_generator import _extract_step_segment  # type: ignore
    except Exception:
        _extract_step_segment = None  # type: ignore
    for i in range(step_count):
        seg = ""
        if _extract_step_segment is not None:
            try:
                seg = _extract_step_segment(code, f"step-{i + 1}")
            except Exception:
                seg = ""
        if seg.strip():
            blocks[i] = seg.strip("\n") + "\n"
    return blocks
