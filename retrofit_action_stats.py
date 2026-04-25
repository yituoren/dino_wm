"""Backfill per-step action_mean/action_std into pre-bake DINO-WM ckpts.

train.py:save_ckpt now bundles action stats into the ckpt dict so
eval-only machines don't need the finetune dataset on disk. Older ckpts
(saved before that change) are missing the keys; this script computes
them from disk and rewrites each ckpt in-place.

Usage:
    python retrofit_action_stats.py \\
        --ckpt-root  $CKPT_BASE/outputs \\
        --data-root  $FINETUNE_ROOT \\
        [--ckpt-glob 'checkpoints/model_*.pth'] \\
        [--subdir-template 'ft_{task}_{N}ep'] \\
        [--tasks click_bell beat_block_hammer ...] \\
        [--splits 1 3 10 50] \\
        [--dry-run] [--force]

Layout assumed (matches run_finetune.sh / run_eval.sh):
    <ckpt-root>/ft_<task>_<N>ep/checkpoints/model_*.pth
    <data-root>/<task>/<N>_episodes/<ep>/actions_raw.npy

For each (task, N), stats are computed once from the data dir and then
written into every ckpt under that subdir. Existing action_mean/
action_std keys are left alone unless --force is passed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def compute_stats(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Mirror RoboTwinPreprocessedDataset._compute_action_stats."""
    chunks = []
    for ep in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        raw_path = ep / "actions_raw.npy"
        if not raw_path.is_file():
            continue
        raw = np.load(raw_path, mmap_mode="r")  # (T, fs, A)
        chunks.append(np.asarray(raw[:, -1], dtype=np.float32))
    if not chunks:
        raise RuntimeError(f"no actions_raw.npy under {data_dir}")
    cat = np.concatenate(chunks, axis=0)
    mean = cat.mean(axis=0)
    std = np.clip(cat.std(axis=0), 1e-6, None)
    return mean.astype(np.float32), std.astype(np.float32)


def retrofit_ckpt(
    ckpt_path: Path,
    mean: np.ndarray,
    std: np.ndarray,
    force: bool,
    dry_run: bool,
) -> str:
    # Submodules pickled in the ckpt may reference custom classes; load them
    # on CPU and trust the pickle (these are our own ckpts).
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    has_mean = "action_mean" in ckpt and ckpt["action_mean"] is not None
    has_std = "action_std" in ckpt and ckpt["action_std"] is not None
    if has_mean and has_std and not force:
        return "skip-existing"
    ckpt["action_mean"] = torch.from_numpy(mean).float()
    ckpt["action_std"] = torch.from_numpy(std).float()
    if dry_run:
        return "would-write"
    torch.save(ckpt, str(ckpt_path))
    return "wrote" if not (has_mean and has_std) else "overwrote"


def resolve_subdir(template: str, task: str, n: str) -> str:
    return template.replace("{task}", task).replace("{N}", n)


DEFAULT_TASKS = (
    "click_bell",
    "beat_block_hammer",
    "handover_block",
    "blocks_ranking_size",
)
DEFAULT_SPLITS = ("1", "3", "10", "50")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", required=True, type=Path,
                    help="dir containing ft_<task>_<N>ep/checkpoints/")
    ap.add_argument("--data-root", required=True, type=Path,
                    help="dir containing <task>/<N>_episodes/<ep>/actions_raw.npy")
    ap.add_argument("--ckpt-glob", default="checkpoints/model_*.pth",
                    help="glob (relative to ft_<task>_<N>ep/) for ckpt files")
    ap.add_argument("--subdir-template", default="ft_{task}_{N}ep")
    ap.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    ap.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing action_mean/action_std")
    args = ap.parse_args()

    if not args.ckpt_root.is_dir():
        ap.error(f"ckpt-root not a dir: {args.ckpt_root}")
    if not args.data_root.is_dir():
        ap.error(f"data-root not a dir: {args.data_root}")

    total = 0
    for task in args.tasks:
        for n in args.splits:
            sub = resolve_subdir(args.subdir_template, task, n)
            ckpt_dir = args.ckpt_root / sub
            data_dir = args.data_root / task / f"{n}_episodes"
            if not ckpt_dir.is_dir():
                print(f"[skip] {sub}: ckpt subdir missing")
                continue
            if not data_dir.is_dir():
                print(f"[skip] {sub}: data dir missing ({data_dir})")
                continue
            ckpts = sorted(ckpt_dir.glob(args.ckpt_glob))
            if not ckpts:
                print(f"[skip] {sub}: no ckpt matching '{args.ckpt_glob}'")
                continue

            try:
                mean, std = compute_stats(data_dir)
            except RuntimeError as e:
                print(f"[skip] {sub}: {e}")
                continue
            print(f"[stats] {sub}: A={len(mean)}, |mean|={np.linalg.norm(mean):.3f}, "
                  f"|std|={np.linalg.norm(std):.3f}")

            for ck in ckpts:
                action = retrofit_ckpt(ck, mean, std, args.force, args.dry_run)
                print(f"  {action}: {ck.relative_to(args.ckpt_root)}")
                total += 1

    print(f"\n[retrofit] processed {total} ckpt file(s){' (dry-run)' if args.dry_run else ''}")


if __name__ == "__main__":
    main()
