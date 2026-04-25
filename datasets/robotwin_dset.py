"""DINO-WM adapter for the preprocessed RoboTwin format.

Reads the per-episode dump produced by ``sdlwm.data.preprocess`` (see
``lwm/sdlwm/data/preprocess.py``):

    <data_path>/<episode_name>/
        frames.npy        (T_ds, 3, H, W) uint8   -- already resized & CHW
        actions_raw.npy   (T_ds, fs, A)  float32  -- repeat-last-pad setpoints
        meta.json

We expose the *downsampled* stream directly (one dataset step == one
preprocessed frame), so the DINO-WM config should run with ``frameskip=1``
even though the underlying HDF5 used a real skip of 5 -- that skip has
already been baked into the .npy files. ``TrajSlicerDataset`` then does
the standard ``(num_frames)`` windowing on top.

Per-step action is ``actions_raw[:, -1]`` (last-of-skip -- the legal
drive_target setpoint that advances to the next sampled frame), matching
what SD-LWM's ``FinetuneEpisodeDataset`` uses.

Proprio / state are set to the same setpoint vector: we don't save real
qpos during preprocess, and the drive_target is an adequate proxy for
a JEPA-style predictor that treats proprio as an auxiliary token.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from einops import rearrange

from .traj_dset import (
    TrajDataset,
    TrajSlicerDataset,
    TrajSubset,
    get_train_val_sliced,
)


class RoboTwinPreprocessedDataset(TrajDataset):
    def __init__(
        self,
        data_path: str | Path,
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
        n_rollout: Optional[int] = None,
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalize_action = normalize_action

        episodes: list[Path] = []
        seq_lengths: list[int] = []
        action_dim: int | None = None
        for ep in sorted(p for p in self.data_path.iterdir() if p.is_dir()):
            meta_path = ep / "meta.json"
            raw_path = ep / "actions_raw.npy"
            if not (meta_path.is_file() and raw_path.is_file()):
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                continue
            if not meta.get("has_actions_raw", False):
                continue
            T = int(meta.get("ds_length", 0))
            if T < 2:
                continue
            if action_dim is None:
                action_dim = int(meta.get("action_dim", 0)) or None
            episodes.append(ep)
            seq_lengths.append(T)

        if n_rollout is not None:
            episodes = episodes[:n_rollout]
            seq_lengths = seq_lengths[:n_rollout]

        if not episodes:
            raise RuntimeError(
                f"RoboTwinPreprocessedDataset: no usable episodes under {self.data_path}"
            )
        if action_dim is None:
            # meta.json predates action_dim field -- sniff from the first file
            action_dim = int(np.load(episodes[0] / "actions_raw.npy", mmap_mode="r").shape[-1])

        self._episodes = episodes
        self._seq_lengths = seq_lengths

        self.action_dim = int(action_dim)
        self.proprio_dim = int(action_dim)
        self.state_dim = int(action_dim)

        self.action_mean, self.action_std = self._compute_action_stats()
        self.proprio_mean = self.action_mean.clone()
        self.proprio_std = self.action_std.clone()
        self.state_mean = self.action_mean.clone()
        self.state_std = self.action_std.clone()

        print(f"[RoboTwinPreprocessedDataset] loaded {len(self)} episodes "
              f"from {self.data_path} (action_dim={self.action_dim}, "
              f"normalize_action={self.normalize_action})")

    def _compute_action_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.normalize_action:
            return torch.zeros(self.action_dim), torch.ones(self.action_dim)
        chunks = []
        for ep in self._episodes:
            raw = np.load(ep / "actions_raw.npy", mmap_mode="r")  # (T, fs, A)
            chunks.append(np.asarray(raw[:, -1], dtype=np.float32))
        cat = np.concatenate(chunks, axis=0)
        mean = torch.from_numpy(cat.mean(axis=0)).float()
        std = torch.from_numpy(np.clip(cat.std(axis=0), 1e-6, None)).float()
        return mean, std

    def get_seq_length(self, idx: int) -> int:
        return self._seq_lengths[idx]

    def __len__(self) -> int:
        return len(self._episodes)

    def __getitem__(self, idx: int):
        ep = self._episodes[idx]
        frames = np.load(ep / "frames.npy")                          # (T, 3, H, W) uint8
        actions_raw = np.load(ep / "actions_raw.npy")                # (T, fs, A)
        actions = np.ascontiguousarray(actions_raw[:, -1])           # (T, A) last-of-skip

        image = torch.from_numpy(np.ascontiguousarray(frames).astype(np.float32) / 255.0)
        if self.transform is not None:
            image = self.transform(image)

        act = torch.from_numpy(actions.astype(np.float32))
        act = (act - self.action_mean) / self.action_std
        proprio = act.clone()
        state = act.clone()

        obs = {"visual": image, "proprio": proprio}
        return obs, act, state, {}

    def preprocess_imgs(self, imgs):
        """Matches the surface used by point_maze_dset for offline eval tools."""
        if isinstance(imgs, np.ndarray):
            imgs = torch.from_numpy(imgs)
        if imgs.dtype == torch.uint8:
            imgs = imgs.float() / 255.0
        if imgs.ndim == 4 and imgs.shape[-1] == 3:
            imgs = rearrange(imgs, "b h w c -> b c h w")
        return imgs


def load_robotwin_slice_train_val(
    transform,
    data_path,
    normalize_action: bool = False,
    split_ratio: float = 0.9,
    n_rollout: Optional[int] = None,
    num_hist: int = 0,
    num_pred: int = 0,
    frameskip: int = 1,
):
    """Entry point referenced by ``conf/env/robotwin.yaml`` via ``_target_``.

    ``frameskip`` is the DINO-WM window stride over the *preprocessed*
    episode -- the raw HDF5 skip is already baked in, so set this to 1 in
    the config unless you actually want to sub-sample the .npy files
    further (rarely useful).
    """
    dset = RoboTwinPreprocessedDataset(
        data_path=data_path,
        transform=transform,
        normalize_action=normalize_action,
        n_rollout=n_rollout,
    )

    N = len(dset)
    num_frames = num_hist + num_pred

    if N == 1:
        # Tiny finetune split: share the single episode between train and
        # val. The default split_ratio=0.9 would otherwise put all episodes
        # in val (int(0.9*1)=0), leaving train empty -> empty dataloader
        # -> None batches in train(). No real holdout here -- val loss
        # measures train-set fit, not generalization.
        idx = list(range(N))
        dset_train = TrajSubset(dset, idx)
        dset_val = TrajSubset(dset, idx)
        train_slices = TrajSlicerDataset(dset_train, num_frames, frameskip)
        val_slices = TrajSlicerDataset(dset_val, num_frames, frameskip)
    else:
        # Clamp so neither half is empty regardless of split_ratio. With
        # the default ratio=0.9 and N>=2 this is a no-op; the clamp matters
        # only at extreme ratios (~0 or ~1) or pathological N.
        n_train = max(1, min(N - 1, int(split_ratio * N)))
        eff_ratio = n_train / N
        dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
            traj_dataset=dset,
            train_fraction=eff_ratio,
            num_frames=num_frames,
            frameskip=frameskip,
        )

    datasets = {"train": train_slices, "valid": val_slices}
    traj_dset = {"train": dset_train, "valid": dset_val}
    return datasets, traj_dset
