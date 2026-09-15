"""PolicyWriter that can carry a sticky strategy directive.

Used only by the diversity half of the step-growth arm. The base
``PolicyWriter`` already has a HARD, priority-0 prompt slot
(``subagent_directive``, rendered above LESSONS FROM PAST FAILURES and above
the retry context) that the parallel sub-agents use to pin one worker to one
approach. That is exactly the slot a strategy directive needs, and reusing it
means ``rats/prompts/policy_writer.txt`` does not change.

The directive is *sticky*: the controller sets ``pending_directive`` once per
attempt and both the main writer call and the self-repair call inside
``_write_policy_with_quality_gate`` pick it up without either call site being
touched. An explicit ``subagent_directive=`` argument always wins, so the
parallel sub-agent path is unaffected.

With the arm (or its diversity half) off, ``make_policy_writer`` returns a
plain ``PolicyWriter`` and nothing in this file runs.
"""

from __future__ import annotations

import logging
from typing import Any

from rats.agents.policy_writer import PolicyWriter

logger = logging.getLogger("rats.step_growth.diversity")


class DirectivePolicyWriter(PolicyWriter):
    """``PolicyWriter`` + a ``pending_directive`` the loop can set per attempt."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.pending_directive: str = ""
        # Set by the controller for artifacts/audit; not read by the prompt.
        self.pending_directive_meta: dict[str, Any] = {}

    def write(
        self,
        plan: dict[str, Any],
        scene_context: dict[str, Any],
        *,
        retry_feedback: dict[str, Any] | None = None,
        failure_context: str = "",
        success_context: str = "",
        subagent_directive: str = "",
    ) -> str:
        if not subagent_directive and self.pending_directive:
            subagent_directive = self.pending_directive
        return super().write(
            plan,
            scene_context,
            retry_feedback=retry_feedback,
            failure_context=failure_context,
            success_context=success_context,
            subagent_directive=subagent_directive,
        )


def make_policy_writer(max_retries: int, ensemble_n: int) -> PolicyWriter:
    """The writer class for this run: directive-carrying only when diversity is on."""
    try:
        from rats.step_growth.config import diversity_enabled
    except Exception:
        return PolicyWriter(max_retries=max_retries, ensemble_n=ensemble_n)
    if not diversity_enabled():
        return PolicyWriter(max_retries=max_retries, ensemble_n=ensemble_n)
    logger.info("Step-growth diversity: PolicyWriter carries strategy directives")
    return DirectivePolicyWriter(max_retries=max_retries, ensemble_n=ensemble_n)
