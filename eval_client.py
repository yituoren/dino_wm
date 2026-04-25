"""DINO-WM eval client: iterates curated seeds under a goal bank.

Drives the random-shooting MPC planner from ``robotwin_policy.DINOWMRoboTwinPolicy``
(same directory) against a ``RoboTwin/script/eval_env_server.py`` running
in a separate process. Connects via the length-prefixed JSON protocol in
the vendored ``eval_protocol.py``.

Unlike the LeWM client, DINO-WM's checkpoint is *not* self-contained:
rehydration needs both the ``.pth`` ckpt (submodules pickled directly) and
the run's saved ``hydra.yaml``. Default layout from ``train.py`` is::

    <run_dir>/hydra.yaml
    <run_dir>/checkpoints/model_<epoch>.pth

so when ``--hydra-cfg`` is omitted we auto-locate ``hydra.yaml`` in the
ckpt's grandparent directory (``<ckpt>.parent.parent``).

Goal bank layout (one dir per task, produced by a collection script):

    <goals_root>/<task>/
        seeds.json      -- {task_name, task_config, camera_key, success_seeds, ...}
        seed_<N>.png    -- final frame for seed N

DINO-WM's ``get_action`` hard-fails without a goal, so seeds missing a
matching ``seed_<N>.png`` are always skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import eval_protocol as proto  # noqa: E402
from robotwin_policy import DINOWMRoboTwinPolicy  # noqa: E402


# ------------------------------------------------------------------- args

def _split_csv(values: list[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for v in values:
        out.extend(x.strip() for x in v.split(",") if x.strip())
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9765)
    p.add_argument("--goals-root", required=True,
                   help="Root dir holding <task>/seeds.json and seed_<N>.png files.")
    p.add_argument("--task", action="append", default=None,
                   help="Filter to these tasks (comma-sep or repeated). "
                        "Default: every subdir of --goals-root with a seeds.json.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt",
                   help="Single DINO-WM .pth ckpt; all tasks share it.")
    g.add_argument("--ckpt-dir",
                   help="Per-task root: resolves to <dir>/<task>/<pattern> via --ckpt-glob.")
    p.add_argument("--ckpt-glob", default="checkpoints/model_latest.pth",
                   help="Glob inside <ckpt-dir>/<task>/; picks newest mtime match. "
                        "Default: checkpoints/model_latest.pth")
    p.add_argument("--hydra-cfg", default=None,
                   help="Optional override for hydra.yaml. Default: <ckpt>.parent.parent/hydra.yaml.")
    p.add_argument("--data-path", default=None,
                   help="Training data dir (for recomputing action mean/std if "
                        "the ckpt's hydra cfg has normalize_action=true). "
                        "If --ckpt-dir is used this is interpreted per-task as "
                        "<data-path>/<task>.")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Cap episodes per task (head of success_seeds list).")
    p.add_argument("--instruction-type", default="seen")
    p.add_argument("--action-type", default="qpos")
    p.add_argument("--device", default="cuda")
    p.add_argument("--clear-cache-freq", type=int, default=10)
    p.add_argument("--shutdown-server", action="store_true")

    # MPC knobs
    p.add_argument("--planning-horizon", type=int, default=5)
    p.add_argument("--planning-iters", type=int, default=100)
    p.add_argument("--action-scale", type=float, default=1.0)
    return p.parse_args()


# ------------------------------------------------------------ goal bank

def _discover_tasks(goals_root: Path, filt: list[str] | None) -> list[Path]:
    if not goals_root.exists():
        raise FileNotFoundError(f"goals-root not found: {goals_root}")
    dirs = sorted([p for p in goals_root.iterdir()
                   if p.is_dir() and (p / "seeds.json").is_file()])
    if filt:
        names = set(filt)
        dirs = [p for p in dirs if p.name in names]
        missing = names - {p.name for p in dirs}
        if missing:
            raise SystemExit(f"no goal bank for task(s): {sorted(missing)}")
    return dirs


def _load_bank(task_dir: Path) -> dict:
    with (task_dir / "seeds.json").open() as f:
        bank = json.load(f)
    bank["_dir"] = task_dir
    bank["_task_dir_name"] = task_dir.name
    bank.setdefault("task_name", task_dir.name)
    return bank


# ------------------------------------------------------------ per-episode

def _run_episode(
    client: proto.RpcClient,
    policy: DINOWMRoboTwinPolicy,
    camera_key: str,
    action_type: str,
) -> bool:
    policy.reset_obs()
    while True:
        obs = client.call("get_obs", cameras=[camera_key])
        actions = policy.get_action(obs)
        for a in actions:
            status = client.call(
                "step",
                action=np.asarray(a, dtype=np.float32),
                action_type=action_type,
            )
            if status["done"]:
                return bool(status["eval_success"])


# --------------------------------------------------------------- per-task

def _resolve_ckpt(args: argparse.Namespace, task_name: str) -> Path:
    if args.ckpt:
        return Path(args.ckpt).expanduser()
    task_dir = Path(args.ckpt_dir).expanduser() / task_name
    if not task_dir.is_dir():
        raise SystemExit(f"no dinowm ckpt dir for task {task_name!r}: {task_dir}")
    matches = sorted(task_dir.glob(args.ckpt_glob), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise SystemExit(f"no ckpt in {task_dir} matching {args.ckpt_glob!r}")
    return matches[-1]


def _resolve_hydra_cfg(args: argparse.Namespace, ckpt_path: Path) -> Path:
    if args.hydra_cfg:
        return Path(args.hydra_cfg).expanduser()
    cand = ckpt_path.parent.parent / "hydra.yaml"
    if not cand.is_file():
        raise SystemExit(
            f"no hydra.yaml alongside ckpt {ckpt_path}; pass --hydra-cfg explicitly."
        )
    return cand


def _resolve_data_path(args: argparse.Namespace, task_name: str) -> Path | None:
    if args.data_path is None:
        return None
    root = Path(args.data_path).expanduser()
    per_task = root / task_name
    if per_task.is_dir():
        return per_task
    return root


def _run_task(
    client: proto.RpcClient,
    bank: dict,
    args: argparse.Namespace,
) -> tuple[int, int]:
    task_name = bank["task_name"]
    task_config = bank.get("task_config")
    if not task_config:
        raise SystemExit(
            f"seeds.json for {task_name!r} is missing 'task_config'"
        )
    camera_key = bank.get("camera_key", "head_camera")
    seeds: list[int] = list(bank.get("success_seeds") or [])
    if not seeds:
        print(f"[client] {task_name}: no success_seeds in bank; skipping.", flush=True)
        return 0, 0
    if args.max_episodes is not None:
        seeds = seeds[: args.max_episodes]

    ckpt_path = _resolve_ckpt(args, task_name)
    hydra_cfg_path = _resolve_hydra_cfg(args, ckpt_path)
    data_path = _resolve_data_path(args, task_name)

    info = client.call(
        "init_task",
        task_config=task_config,
        task_name=task_name,
    )
    print(f"[client] ==== task={task_name} config={task_config} "
          f"seeds={len(seeds)} "
          f"cam={info['head_camera_w']}x{info['head_camera_h']} ====",
          flush=True)
    print(f"[client] dinowm_ckpt={ckpt_path}", flush=True)
    print(f"[client] dinowm_hydra={hydra_cfg_path}", flush=True)
    if data_path is not None:
        print(f"[client] data_path={data_path}", flush=True)
    print(f"[client] mpc: horizon={args.planning_horizon} iters={args.planning_iters} "
          f"act_scale={args.action_scale}", flush=True)
    if info.get("video_root"):
        print(f"[client] video_root={info['video_root']}", flush=True)

    policy = DINOWMRoboTwinPolicy(
        ckpt_path=str(ckpt_path),
        hydra_cfg_path=str(hydra_cfg_path),
        data_path=str(data_path) if data_path is not None else None,
        goal_bank_root=args.goals_root,
        task_name=task_name,
        camera_key=camera_key,
        device=args.device,
        planning_horizon=args.planning_horizon,
        planning_iters=args.planning_iters,
        action_scale=args.action_scale,
    )

    success = 0
    run = 0
    for idx, seed in enumerate(seeds):
        goal_loaded = policy.load_goal_for_seed(int(seed))
        if not goal_loaded:
            print(f"[client]   seed={seed}: no goal image; skipped.",
                  flush=True)
            continue

        res = client.call(
            "prepare_seed",
            seed=int(seed),
            now_ep_num=idx,
            instruction_type=args.instruction_type,
            skip_expert_check=True,
        )
        if not res["ok"]:
            print(f"[client]   seed={seed}: prepare_seed failed "
                  f"({res.get('skip_reason')!r}); skipped.", flush=True)
            continue

        ep_success = False
        try:
            ep_success = _run_episode(client, policy,
                                      camera_key=camera_key,
                                      action_type=args.action_type)
        except Exception as e:
            print(f"[client]   seed={seed}: episode error: {e}", flush=True)
            traceback.print_exc()

        run += 1
        clear_cache = (run % args.clear_cache_freq) == 0
        client.call("close_episode", clear_cache=clear_cache)

        if ep_success:
            success += 1
            tag = "\033[92mSUCCESS\033[0m"
        else:
            tag = "\033[91mFAIL\033[0m"
        print(f"[client]   seed={seed} -> {tag} "
              f"({success}/{run} = {success / max(1, run) * 100:.1f}%)",
              flush=True)

    return success, run


# ------------------------------------------------------------------- main

def main():
    args = _parse_args()
    goals_root = Path(args.goals_root).expanduser().resolve()
    task_filter = _split_csv(args.task)

    tasks = _discover_tasks(goals_root, task_filter or None)
    if not tasks:
        raise SystemExit(f"no tasks under {goals_root} "
                         f"(filter={task_filter or 'none'})")

    print(f"[client] goals_root={goals_root}", flush=True)
    print(f"[client] tasks: {[t.name for t in tasks]}", flush=True)

    client = proto.RpcClient(args.host, args.port, timeout=None)
    results: list[tuple[str, int, int]] = []
    t0 = time.time()
    try:
        for task_dir in tasks:
            bank = _load_bank(task_dir)
            try:
                succ, run = _run_task(client, bank, args)
            except Exception as e:
                print(f"[client] task {task_dir.name!r} crashed: {e}",
                      flush=True)
                traceback.print_exc()
                succ, run = 0, 0
                try:
                    client.call("close_episode", clear_cache=True)
                except Exception:
                    pass
            results.append((task_dir.name, succ, run))

        if args.shutdown_server:
            try:
                client.call("shutdown")
            except Exception:
                pass
    finally:
        client.close()

    dt = time.time() - t0
    print("", flush=True)
    print("=" * 70, flush=True)
    print("[client] per-task summary", flush=True)
    total_s = total_r = 0
    for name, s, r in results:
        rate = s / r * 100 if r else 0.0
        print(f"  {name:<40} {s}/{r} = {rate:5.1f}%", flush=True)
        total_s += s
        total_r += r
    if total_r:
        print(f"  {'OVERALL':<40} {total_s}/{total_r} "
              f"= {total_s / total_r * 100:5.1f}%", flush=True)
    print(f"[client] elapsed: {dt:.1f}s", flush=True)


if __name__ == "__main__":
    main()
