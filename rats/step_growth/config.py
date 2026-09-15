"""Configuration for the step-growth arm.

Loaded by the package itself (not through the molmospaces config merge,
which LIBERO env YAMLs cannot reach). Defaults live in
``rats/config/step_growth.yaml``; ``RATS_STEP_GROWTH_CONFIG`` points at an
alternative file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "rats" / "config" / "step_growth.yaml"

_TRUTHY = {"1", "true", "yes", "on"}


_FALSY = {"0", "false", "no", "off"}


def step_growth_enabled() -> bool:
    """``RATS_STEP_GROWTH=1`` turns the whole arm on. Anything else: off."""
    return os.getenv("RATS_STEP_GROWTH", "").strip().lower() in _TRUTHY


def diversity_enabled(cfg: "StepGrowthConfig | None" = None) -> bool:
    """Is the DIVERSITY half (section B: strategy bank + bandit + fingerprints) on?

    Three-state on purpose. ``RATS_STEP_GROWTH_DIVERSITY`` wins in BOTH
    directions so a launcher can turn the half on (or force it off) without
    editing the YAML; unset falls back to ``diversity.enabled`` in the config
    file, which ships **false**. That default is what keeps the original
    step-growth arm reproducible: re-running
    ``run-rats-play-qwen-stepgrowth.sh`` gets the same loop it always had, and
    diversity is a separate arm with its own launcher and output tree.

    Always false when the arm itself is off — the bandit's reward is the
    oracle milestone, so there is no diversity-without-step-growth mode.
    """
    if not step_growth_enabled():
        return False
    raw = os.getenv("RATS_STEP_GROWTH_DIVERSITY", "").strip().lower()
    if raw in _TRUTHY:
        return True
    if raw in _FALSY:
        return False
    return bool((cfg if cfg is not None else load_config()).diversity_enabled)


@dataclass
class StepGrowthConfig:
    # --- oracle / milestone thresholds (metres) ---
    lift_dz_m: float = 0.03           # pick_event lift threshold mirrors libero.py _PICK_LIFT_M
    grasp_end_dz_m: float = 0.005     # boundary grasp check: contact + closed + this much lift
    gripper_closed_max_fraction: float = 0.6  # _gripper_fraction <= this counts as closed
    near_radius_m: float = 0.10       # transport milestone near(a, b): xy distance
    near_z_tolerance_m: float = 0.02  # near also needs z(a) >= z(b) - tol
    persist_sidecar: bool = True      # iteration_NNN/attempt_NN/step_oracle.json
    oracle_to_diagnoser: bool = False # C2 extension, off by default (not wired in v1)
    multi_chain: str = "max_time"     # how multi-predicate goals pick t*
    # --- step credit / tier policy ---
    tier_policy: str = "step"         # "step" | "task"
    promote_min_step_uses: int = 3
    promote_min_step_sr: float = 0.6
    deprecate_min_step_uses: int = 8
    deprecate_max_step_sr: float = 0.2
    # --- step-level extraction ---
    extraction_enabled: bool = True
    extraction_max_per_iteration: int = 1
    extraction_max_per_run: int = 200
    extraction_only_on_failed_iteration: bool = True
    extraction_model: str | None = None
    extraction_min_prefix_lines: int = 3
    extraction_max_prefix_lines: int = 200
    extraction_max_skill_lines: int = 80
    # --- curation ---
    # Prompt the MemoryCurator reads while this arm is on. The default file
    # describes only the task-level counters, so a skill living purely on step
    # credit reads to it as dead weight. Empty string = keep the curator's own
    # default (i.e. opt out of the fix).
    curator_prompt_path: str = "rats/prompts/skill_curator_step_growth.txt"
    # --- diversity (section B: strategy bank + bandit + fingerprints) ---
    # OFF by default: turning it on changes what the policy writer is told, so
    # it is its own arm (RATS_STEP_GROWTH_DIVERSITY=1 + its own output tree).
    diversity_enabled: bool = False
    diversity_bank_path: str = "rats/config/strategy_bank.yaml"
    diversity_seed: int | None = None
    diversity_min_pulls: int = 2          # forced exploration inside the window
    diversity_window_iters: int = 10
    diversity_collapse_k: int = 5         # identical attempt-0 fingerprints -> override
    diversity_show_evidence: bool = False # bandit stats in the prompt (oracle-derived: keep off)
    diversity_retry_switch_after: int = 2 # consecutive milestone misses before switching family
    diversity_per_type_families: dict[str, int] = field(
        default_factory=lambda: {"grasp": 1, "place": 1, "open_close": 1, "turn": 1, "localize": 0}
    )
    # informational
    source_path: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "raw"}
        return out


# yaml section -> dataclass field names
_SECTION_KEYS: dict[str, dict[str, str]] = {
    "oracle": {
        "lift_dz_m": "lift_dz_m",
        "grasp_end_dz_m": "grasp_end_dz_m",
        "gripper_closed_max_fraction": "gripper_closed_max_fraction",
        "persist_sidecar": "persist_sidecar",
        "oracle_to_diagnoser": "oracle_to_diagnoser",
    },
    "milestones": {
        "near_radius_m": "near_radius_m",
        "near_z_tolerance_m": "near_z_tolerance_m",
        "multi_chain": "multi_chain",
    },
    "credit": {
        "tier_policy": "tier_policy",
        "promote_min_step_uses": "promote_min_step_uses",
        "promote_min_step_sr": "promote_min_step_sr",
        "deprecate_min_step_uses": "deprecate_min_step_uses",
        "deprecate_max_step_sr": "deprecate_max_step_sr",
    },
    "curation": {
        "curator_prompt_path": "curator_prompt_path",
    },
    "diversity": {
        "enabled": "diversity_enabled",
        "bank_path": "diversity_bank_path",
        "seed": "diversity_seed",
        "min_pulls": "diversity_min_pulls",
        "window_iters": "diversity_window_iters",
        "collapse_k": "diversity_collapse_k",
        "show_evidence": "diversity_show_evidence",
        "retry_switch_after": "diversity_retry_switch_after",
        "per_type_families": "diversity_per_type_families",
    },
    "extraction": {
        "enabled": "extraction_enabled",
        "max_per_iteration": "extraction_max_per_iteration",
        "max_per_run": "extraction_max_per_run",
        "only_on_failed_iteration": "extraction_only_on_failed_iteration",
        "model": "extraction_model",
        "min_prefix_lines": "extraction_min_prefix_lines",
        "max_prefix_lines": "extraction_max_prefix_lines",
        "max_skill_lines": "extraction_max_skill_lines",
    },
}


def load_config(path: str | os.PathLike[str] | None = None) -> StepGrowthConfig:
    """Load ``step_growth.yaml`` (or ``RATS_STEP_GROWTH_CONFIG``) over the defaults.

    Missing file or missing keys silently keep the dataclass defaults so the
    arm can run from a bare checkout.
    """
    cfg = StepGrowthConfig()
    candidate = path or os.getenv("RATS_STEP_GROWTH_CONFIG") or DEFAULT_CONFIG_PATH
    candidate = Path(candidate)
    if not candidate.is_absolute():
        # Resolve relative to the project root first, then CWD.
        root_rel = _PROJECT_ROOT / candidate
        candidate = root_rel if root_rel.exists() else candidate
    if not candidate.exists():
        return cfg
    try:
        import yaml  # type: ignore

        with candidate.open() as fh:
            raw = yaml.safe_load(fh) or {}
    except Exception:
        return cfg
    if not isinstance(raw, dict):
        return cfg
    cfg.raw = raw
    cfg.source_path = str(candidate)
    for section, keymap in _SECTION_KEYS.items():
        sec = raw.get(section)
        if not isinstance(sec, dict):
            continue
        for yaml_key, attr in keymap.items():
            if yaml_key in sec and sec[yaml_key] is not None:
                current = getattr(cfg, attr)
                value = sec[yaml_key]
                try:
                    if isinstance(current, bool):
                        value = bool(value) if not isinstance(value, str) else value.lower() in _TRUTHY
                    elif isinstance(current, int):
                        value = int(value)
                    elif isinstance(current, float):
                        value = float(value)
                except (TypeError, ValueError):
                    continue
                setattr(cfg, attr, value)
    # per_type_families is the one non-scalar key: keep only str -> int pairs,
    # and keep the defaults for any type the file does not mention.
    if not isinstance(cfg.diversity_per_type_families, dict):
        cfg.diversity_per_type_families = StepGrowthConfig().diversity_per_type_families
    else:
        merged = StepGrowthConfig().diversity_per_type_families
        for k, v in cfg.diversity_per_type_families.items():
            try:
                merged[str(k)] = int(v)
            except (TypeError, ValueError):
                continue
        cfg.diversity_per_type_families = merged
    env_seed = os.getenv("RATS_STEP_GROWTH_DIVERSITY_SEED", "").strip()
    if env_seed:
        cfg.diversity_seed = env_seed
    if cfg.diversity_seed is not None:
        try:
            cfg.diversity_seed = int(cfg.diversity_seed)
        except (TypeError, ValueError):
            cfg.diversity_seed = None
    # Model override via env, same convention as the other agents.
    env_model = os.getenv("RATS_STEP_SKILL_EXTRACTOR_MODEL")
    if env_model:
        cfg.extraction_model = env_model
    return cfg
