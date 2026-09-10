#!/usr/bin/env python3
"""Replay a past play attempt with the step-oracle recorder on, and compare the
oracle step verdicts against the VLM per-step verdicts stored in the run.

This is the "問題がなければ導入" check for the step-growth arm: it re-executes
``code_attempt_A`` from ``iteration_NNN.json`` in a freshly built LIBERO env
(same BDDL, ``reset(seed=N)`` exactly like the loop's ``_reset_env``), records
``describe_object_state()`` at every ``step_context`` boundary, judges the
steps with the milestone rule, and prints a per-step table next to
``per_step_verification_attempt_A.steps[i].success``.

Must run where the perception servers are reachable (same env vars the
launcher exports: SAM3_SERVICE_URL / GRASPNET_SERVICE_URL / PYROKI_SERVICE_URL
/ MOLMO_BASE_URL, MUJOCO_GL=egl). Physics is seeded; perception may drift, so
steps whose API-call trace diverges are flagged, not silently trusted.

Usage:
  python scripts/replay_step_oracle.py --run-dir R --iteration 12 [--attempt 0]
      [--config env_configs/libero/rats_libero_play_reduced.yaml] [--bddl path]
      [--timeout 600] [--out replay.json]
  python scripts/replay_step_oracle.py --run-dir R --iterations 1-10   # batch + summary
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _slug(language: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (language or "").lower()).strip("_")[:60]


def _find_bddl(run_dir: Path, iteration_data: dict[str, Any], override: str | None) -> Path:
    if override:
        return Path(override)
    sg = iteration_data.get("step_growth") or {}
    if sg.get("bddl_path") and Path(sg["bddl_path"]).exists():
        return Path(sg["bddl_path"])
    tp = iteration_data.get("task_proposal") or {}
    language = tp.get("language") or ""
    slug = _slug(language)
    cands = sorted(
        (Path(p) for p in glob.glob(str(run_dir / "**" / f"{slug}_*.bddl"), recursive=True)),
        key=lambda p: p.stat().st_mtime,
    )
    if not cands:
        cands = sorted(
            (Path(p) for p in glob.glob(str(run_dir / "**" / "*.bddl"), recursive=True)),
            key=lambda p: p.stat().st_mtime,
        )
        raise SystemExit(
            f"no BDDL matching slug '{slug}' under {run_dir}; {len(cands)} .bddl files exist — pass --bddl"
        )
    if len(cands) > 1:
        print(f"[warn] {len(cands)} BDDL candidates for '{slug}', using newest: {cands[-1]}", file=sys.stderr)
    return cands[-1]


def _api_names(config_path: str | None) -> list[str]:
    default = ["FrankaLiberoApiReduced"]
    if not config_path:
        return default
    try:
        import yaml  # type: ignore

        cfg = yaml.safe_load(Path(config_path).read_text()) or {}
    except Exception:
        return default

    def _walk(node: Any) -> list[str] | None:
        if isinstance(node, dict):
            if isinstance(node.get("apis"), list):
                return [str(x) for x in node["apis"]]
            for v in node.values():
                r = _walk(v)
                if r:
                    return r
        return None

    return _walk(cfg) or default


def _skill_preamble(run_dir: Path, iteration: int) -> tuple[str, set[str]]:
    from skill_library.library import SkillLibrary

    path = run_dir / "skills.json"
    if not path.exists():
        return "", set()
    lib = SkillLibrary(storage_path=str(path))
    defs: list[str] = []
    names: set[str] = set()
    for s in lib.get_full_skills_for_planner(include_deprecated=True):
        if s.get("is_primitive") or not s.get("code"):
            continue
        li = s.get("learned_iteration")
        try:
            if li is not None and int(li) >= iteration:
                continue  # learned later than the replayed iteration
        except (TypeError, ValueError):
            pass
        names.add(s["name"])
        defs.append(s["code"])
    return "\n\n".join(defs), names


def replay_one(
    run_dir: Path, iteration: int, attempt: int | None, *, config: str | None,
    bddl: str | None, timeout: int, cfg_step_growth: Any,
) -> dict[str, Any]:
    from rats.agents.environment_creator import build_full_libero_env_from_bddl
    from rats.executor.sandbox import Executor
    from rats.loop.libero_utils import extract_libero_scene_context
    from rats.step_growth import milestones as ms
    from rats.step_growth.code_slices import step_code_blocks
    from rats.step_growth.oracle_recorder import StepOracleRecorder
    from rats.step_growth.step_judge import judge
    from rats.utils.execution_logger import register_policy_step_listener, unregister_policy_step_listener

    it_path = run_dir / f"iteration_{iteration:03d}.json"
    data = json.loads(it_path.read_text())
    attempts = sorted(int(k.split("_")[-1]) for k in data if k.startswith("code_attempt_") and k.count("_") == 2)
    if not attempts:
        raise SystemExit(f"{it_path}: no code_attempt_* keys")
    if attempt is None:
        attempt = attempts[-1]
    code = data.get(f"code_attempt_{attempt}") or ""
    plan = data.get(f"plan_refined_attempt_{attempt}") or data.get("plan") or {}
    vlm_steps = ((data.get(f"per_step_verification_attempt_{attempt}") or {}).get("steps")) or []
    bddl_path = _find_bddl(run_dir, data, bddl)

    env = build_full_libero_env_from_bddl(str(bddl_path), api_names=_api_names(config), privileged=False)
    try:
        env.reset(seed=iteration)
    except TypeError:
        env.reset()
    low = getattr(env, "low_level_env", env)
    scene_context = extract_libero_scene_context(env)
    preamble, learned = _skill_preamble(run_dir, iteration)
    exec_code = f"{preamble}\n\n{code}" if preamble else code

    recorder = StepOracleRecorder(cfg_step_growth)
    recorder.bind(low)
    register_policy_step_listener(recorder.on_step_event)
    try:
        recorder.begin_attempt(iteration=iteration, attempt=attempt, attempt_in_iter=attempt, turn_in_attempt=0, env_reset=True)
        result = Executor(timeout_seconds=timeout).execute(exec_code, env, scene_context)
    finally:
        unregister_policy_step_listener(recorder.on_step_event)
    record = recorder.end_attempt((result.get("artifacts") or {}).get("grounded_state"))

    pred_env = low._predicate_env() if hasattr(low, "_predicate_env") else None
    goal_state = (getattr(pred_env, "parsed_problem", None) or {}).get("goal_state") if pred_env else None
    goal_state = ms.parse_goal_state(goal_state) if goal_state else ms.goal_state_from_record(record)
    mres = ms.evaluate(goal_state, record, cfg_step_growth)
    verdicts = judge(mres, record, list(plan.get("steps") or []), cfg_step_growth)
    blocks = step_code_blocks(code)

    vlm_by_index: dict[int, dict[str, Any]] = {}
    for i, st in enumerate(vlm_steps):
        vlm_by_index[i] = st
    rows: list[dict[str, Any]] = []
    agree = disagree = compared = 0
    for sv in verdicts.steps:
        v = vlm_by_index.get(sv.step_index) or {}
        vlm_ok = v.get("success")
        row = {
            "step_index": sv.step_index, "step_id": sv.step_id, "oracle": sv.verdict, "reason": sv.reason,
            "vlm_success": vlm_ok, "vlm_status": v.get("status"), "vlm_confidence": v.get("confidence"),
            "has_code_slice": sv.step_index in blocks,
        }
        if sv.verdict in ("pass", "fail") and isinstance(vlm_ok, bool):
            compared += 1
            same = (sv.verdict == "pass") == vlm_ok
            agree += int(same)
            disagree += int(not same)
            row["agree"] = same
        rows.append(row)
    stored_verif = data.get(f"verification_attempt_{attempt}") or {}
    # Motion diagnostics. Without these an attempt that never moved the robot
    # and an attempt that manipulated the scene and failed both print
    # progress=0.00, and the run cannot tell an inert replay from a real one.
    bnds = record.get("boundaries") or []
    sims = [b.get("sim_step") for b in bnds if isinstance(b.get("sim_step"), int)]
    contact_bnds = sum(1 for b in bnds
                       if ((b.get("snapshot") or {}).get("fingerpad_contact") or []))
    max_dz = {k: round(float(v), 4) for k, v in (record.get("max_lift_dz") or {}).items()}
    out = {
        "run_dir": str(run_dir), "iteration": iteration, "attempt": attempt, "bddl": str(bddl_path),
        "exec_success": bool(result.get("success")), "replay_task_completed": result.get("task_completed"),
        "stored_task_completed": stored_verif.get("task_completed"),
        "goal_state": [list(g) for g in goal_state],
        "milestones": mres.as_dict(), "progress": verdicts.progress, "s_star": verdicts.s_star,
        "fail_step": verdicts.fail_step, "fail_reason": verdicts.fail_reason,
        "markers_seen": verdicts.markers_seen, "boundaries": len(bnds),
        "sim_step_first": min(sims) if sims else None, "sim_step_last": max(sims) if sims else None,
        "pick_bodies": len(record.get("baseline_z") or {}),
        "pick_events": record.get("pick_events") or [],
        "max_lift_dz": max_dz,
        "best_lift_dz": max(max_dz.values()) if max_dz else 0.0,
        "contact_boundaries": contact_bnds,
        "learned_skills_in_scope": sorted(learned),
        "rows": rows, "compared": compared, "agree": agree, "disagree": disagree,
    }
    try:
        env.close()
    except Exception:
        pass
    return out


def _print(out: dict[str, Any]) -> None:
    print(f"\n=== iteration {out['iteration']} attempt {out['attempt']}  progress={out['progress']:.2f} "
          f"S*={out['s_star']} fail={out['fail_step']}({out['fail_reason']}) markers={out['markers_seen']} "
          f"exec={out['exec_success']} task_completed replay={out['replay_task_completed']} stored={out['stored_task_completed']}")
    print(f"  goal={out['goal_state']}  achieved={out['milestones']['achieved']}  events={[e['event'] for e in out['milestones']['failure_events']]}")
    print(f"  sim={out['sim_step_first']}..{out['sim_step_last']} tracked_bodies={out['pick_bodies']} "
          f"contact_boundaries={out['contact_boundaries']} best_lift_dz={out['best_lift_dz']:.3f}m "
          f"picks={[e['object'] for e in out['pick_events']]}")
    print(f"  {'idx':>3} {'step_id':<10} {'oracle':<12} {'reason':<24} {'vlm':<6} {'conf':<5} agree")
    for r in out["rows"]:
        conf = r.get("vlm_confidence")
        print(f"  {r['step_index']:>3} {str(r['step_id'])[:10]:<10} {r['oracle']:<12} {str(r['reason'])[:24]:<24} "
              f"{str(r['vlm_success']):<6} {('%.2f' % conf) if isinstance(conf, (int, float)) else '-':<5} {r.get('agree', '-')}")
    print(f"  compared={out['compared']} agree={out['agree']} disagree={out['disagree']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--iteration", type=int)
    ap.add_argument("--iterations", help="range like 1-10 or list 1,4,7")
    ap.add_argument("--attempt", type=int, default=None, help="default: last attempt with code")
    ap.add_argument("--attempts", default=None,
                    help="'all' replays every attempt that has code (overrides --attempt)")
    ap.add_argument("--config", default="env_configs/libero/rats_libero_play_reduced.yaml")
    ap.add_argument("--bddl", default=None)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", default=None, help="write JSON results here")
    a = ap.parse_args()

    from rats.step_growth.config import load_config

    cfg = load_config()
    run_dir = Path(a.run_dir)
    iters: list[int] = []
    if a.iterations:
        for part in a.iterations.split(","):
            if "-" in part:
                lo, hi = part.split("-")
                iters.extend(range(int(lo), int(hi) + 1))
            else:
                iters.append(int(part))
    elif a.iteration is not None:
        iters = [a.iteration]
    else:
        ap.error("--iteration or --iterations required")

    def _attempts_for(it: int) -> list[int | None]:
        if a.attempts != "all":
            return [a.attempt]
        try:
            data = json.loads((run_dir / f"iteration_{it:03d}.json").read_text())
        except Exception:
            return [a.attempt]
        return sorted(int(k.split("_")[-1]) for k in data
                      if k.startswith("code_attempt_") and k.count("_") == 2) or [None]

    results = []
    for it in iters:
        for att in _attempts_for(it):
            try:
                out = replay_one(run_dir, it, att, config=a.config, bddl=a.bddl, timeout=a.timeout, cfg_step_growth=cfg)
            except SystemExit as e:
                print(f"[skip] iteration {it} attempt {att}: {e}", file=sys.stderr)
                continue
            except Exception as e:  # keep the batch going
                print(f"[error] iteration {it} attempt {att}: {e!r}", file=sys.stderr)
                continue
            _print(out)
            results.append(out)
            if a.out:  # checkpoint after every attempt: a long batch must survive a kill
                Path(a.out).write_text(json.dumps(results, indent=1, default=str))
    if results:
        compared = sum(r["compared"] for r in results)
        agree = sum(r["agree"] for r in results)
        n = len(results)
        lifted = sum(1 for r in results if r["pick_events"])
        touched = sum(1 for r in results if r["contact_boundaries"])
        moved = sum(1 for r in results if r["best_lift_dz"] > 0.005)
        inert = sum(1 for r in results if not r["sim_step_last"])
        prog = sum(1 for r in results if r["progress"] > 0)
        print(f"\nTOTAL attempts={n} inert(no sim)={inert} "
              f"moved>5mm={moved} pad_contact={touched} lifted>3cm={lifted} progress>0={prog}")
        print(f"      compared_steps={compared} agree={agree} "
              + (f"rate={(agree / compared):.2f}" if compared else "rate=n/a"))
        if a.out:
            Path(a.out).write_text(json.dumps(results, indent=1, default=str))
            print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
