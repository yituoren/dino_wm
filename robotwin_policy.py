"""DINO-WM inference wrapper for RoboTwin closed-loop eval.

Rehydrates the VWorldModel from a training checkpoint + its saved hydra
config (``<saved_folder>/hydra.yaml``), then drives it with random-shooting
MPC the same way ``reference/oracle/lewm_policy.py`` drives the JEPA
baseline. Exposes the ``reset_obs / update_obs / get_action`` contract that
``RoboTwin/policy/DINOWM/deploy_policy.py`` forwards to.

Key shape contract matches the dataset at ``datasets/robotwin_dset.py``:
  * Visuals: (T, 3, img_size, img_size) float, Normalize(0.5, 0.5) -> [-1, 1]
  * Actions: (T, action_dim), normalized to zero-mean/unit-std if the
    training config used normalize_action=True
  * Proprio: same as action (setpoint proxy; see dataset docstring)

Action_dim in the checkpoint already includes the DINO-WM fs-concat factor
(``dataset.action_dim * frameskip``), so a config trained with frameskip=1
yields action_dim=14 for aloha-agilex. Higher frameskip would mean each
MPC step commits ``frameskip`` qpos setpoints; we currently assume =1.
"""
from __future__ import annotations

import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision import transforms


_MODEL_KEYS = ("encoder", "predictor", "decoder", "proprio_encoder", "action_encoder")


def _preprocess_rgb(rgb: np.ndarray, img_size: int) -> torch.Tensor:
    """HWC uint8 -> (3, S, S) float, Normalize(0.5, 0.5) -> [-1, 1].

    Matches ``datasets.img_transforms.default_transform`` so inference
    pixels land in the same distribution the model was trained on.
    """
    from PIL import Image

    img = Image.fromarray(rgb).resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr.transpose(2, 0, 1))  # (3, S, S)


def _compute_action_stats(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Recompute dataset action mean/std from ``actions_raw[:, -1]``."""
    chunks = []
    for ep in sorted(p for p in data_path.iterdir() if p.is_dir()):
        raw_path = ep / "actions_raw.npy"
        if not raw_path.is_file():
            continue
        raw = np.load(raw_path, mmap_mode="r")  # (T, fs, A)
        chunks.append(np.asarray(raw[:, -1], dtype=np.float32))
    if not chunks:
        raise RuntimeError(f"no actions_raw.npy files under {data_path}")
    cat = np.concatenate(chunks, axis=0)
    return cat.mean(axis=0), np.clip(cat.std(axis=0), 1e-6, None)


def _load_model_from_ckpt(
    ckpt_path: Path,
    hydra_cfg_path: Path,
    device: torch.device,
):
    """Rebuild the VWorldModel from a train-time ckpt + its hydra.yaml.

    DINO-WM's ``save_ckpt`` dumps the submodules directly (not state_dicts)
    via ``torch.save(ckpt_dict, ...)`` with ckpt[key] being the actual
    module, so rehydration is just torch.load + hydra.utils.instantiate on
    the VWorldModel wrapper.
    """
    import hydra
    from hydra.core.global_hydra import GlobalHydra

    cfg = OmegaConf.load(str(hydra_cfg_path))

    ckpt = torch.load(str(ckpt_path), map_location=device)
    modules = {k: ckpt[k].to(device) for k in _MODEL_KEYS if k in ckpt and ckpt[k] is not None}
    # train.py:save_ckpt bundles per-step action_mean/action_std (1-D tensors,
    # length = dataset.action_dim_raw) so eval-only machines don't need the
    # finetune dataset on disk just to recompute normalization stats.
    bundled_mean = ckpt.get("action_mean")
    bundled_std = ckpt.get("action_std")

    if not GlobalHydra.instance().is_initialized():
        # allow hydra.utils.instantiate to resolve relative _target_ strings
        # (e.g. "datasets.img_transforms.default_transform") from the
        # baseline package dir.
        pass

    model = hydra.utils.instantiate(
        cfg.model,
        encoder=modules["encoder"],
        proprio_encoder=modules["proprio_encoder"],
        action_encoder=modules["action_encoder"],
        predictor=modules["predictor"],
        decoder=modules.get("decoder"),
        proprio_dim=cfg.proprio_emb_dim,
        action_dim=cfg.action_emb_dim,
        concat_dim=cfg.concat_dim,
        num_action_repeat=cfg.num_action_repeat,
        num_proprio_repeat=cfg.num_proprio_repeat,
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg, bundled_mean, bundled_std


class DINOWMRoboTwinPolicy:
    """Random-shooting MPC driver for a trained DINO-WM VWorldModel."""

    def __init__(
        self,
        ckpt_path: str | Path,
        hydra_cfg_path: str | Path,
        data_path: str | Path | None = None,
        goal_image_path: str | Path | None = None,
        goal_bank_root: str | Path | None = None,
        task_name: str | None = None,
        camera_key: str = "head_camera",
        device: str | torch.device = "cuda",
        planning_horizon: int = 5,
        planning_iters: int = 100,
        action_scale: float = 1.0,
    ):
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.camera_key = camera_key
        self.goal_bank_root = Path(goal_bank_root) if goal_bank_root else None
        self._task_name_hint = task_name
        self._seeds_cache: dict[Path, set[int]] = {}

        # Make sure ``import datasets`` / ``import models`` used by the
        # hydra targets inside the ckpt resolve to the baseline package.
        baseline_root = Path(__file__).resolve().parent
        if str(baseline_root) not in sys.path:
            sys.path.insert(0, str(baseline_root))

        self.model, self.cfg, bundled_mean, bundled_std = _load_model_from_ckpt(
            Path(ckpt_path), Path(hydra_cfg_path), self.device
        )
        self.img_size = int(self.cfg.img_size)
        self.num_hist = int(self.cfg.num_hist)
        self.frameskip = int(self.cfg.frameskip)
        # action_dim stored in the model already accounts for fs-concat.
        self.action_dim_per_step = int(self.model.action_encoder.in_chans)
        self.action_dim_raw = self.action_dim_per_step // self.frameskip

        self.planning_horizon = int(planning_horizon)
        self.planning_iters = int(planning_iters)
        self.action_scale = float(action_scale)

        # Action normalization stats (match the dataset's normalize_action).
        # Prefer ckpt-bundled stats (saved by train.py:save_ckpt), fall back
        # to recomputing from data_path. The latter only matters for old
        # ckpts saved before stat-baking; retrofit_action_stats.py can
        # backfill them in-place if needed.
        self.normalize_action = bool(self.cfg.get("normalize_action", False))
        if self.normalize_action:
            if bundled_mean is not None and bundled_std is not None:
                self.action_mean = bundled_mean.float().to(self.device)
                self.action_std = bundled_std.float().to(self.device)
            elif data_path is not None:
                mean, std = _compute_action_stats(Path(data_path))
                self.action_mean = torch.from_numpy(mean).float().to(self.device)
                self.action_std = torch.from_numpy(std).float().to(self.device)
            else:
                raise RuntimeError(
                    "Checkpoint was trained with normalize_action=true but "
                    "has no bundled action_mean/action_std, and no data_path "
                    "was provided to recompute them. Either pass data_path "
                    "or run retrofit_action_stats.py on the ckpt."
                )
        else:
            self.action_mean = torch.zeros(self.action_dim_raw, device=self.device)
            self.action_std = torch.ones(self.action_dim_raw, device=self.device)

        self.z_goal_visual: torch.Tensor | None = None
        if goal_image_path is not None:
            self.load_goal_image(goal_image_path)

        self.frame_history: deque = deque(maxlen=self.num_hist)
        self.action_history: deque = deque(maxlen=self.num_hist)

    # --------------------------------------------------------- goal bank
    @torch.no_grad()
    def _set_goal_from_array(self, rgb: np.ndarray) -> None:
        pix = _preprocess_rgb(rgb, self.img_size).to(self.device)  # (3, S, S)
        obs_g = {
            "visual": pix.unsqueeze(0).unsqueeze(0),               # (1, 1, 3, S, S)
            "proprio": torch.zeros(1, 1, self.action_dim_raw, device=self.device),
        }
        z = self.model.encode_obs(obs_g)
        self.z_goal_visual = z["visual"]                            # (1, 1, P, D)

    def load_goal_image(self, path: str | Path) -> None:
        from PIL import Image

        rgb = np.asarray(Image.open(str(path)).convert("RGB"))
        self._set_goal_from_array(rgb)

    def load_goal_for_seed(self, seed: int) -> bool:
        if self.goal_bank_root is None:
            return False
        task = self._task_name_hint or ""
        candidates = []
        if task:
            candidates.append(self.goal_bank_root / task / f"seed_{seed}.png")
        candidates.append(self.goal_bank_root / f"seed_{seed}.png")
        for p in candidates:
            if p.exists():
                self.load_goal_image(p)
                return True
        print(
            f"[DINOWMRoboTwinPolicy] no goal frame for seed={seed} under "
            f"{self.goal_bank_root} (task={task!r}); z_goal unchanged."
        )
        return False

    def available_seeds(self, task_name: str | None = None) -> set[int]:
        if self.goal_bank_root is None:
            return set()
        task = task_name or self._task_name_hint or ""
        path = (self.goal_bank_root / task / "seeds.json") if task else (
            self.goal_bank_root / "seeds.json"
        )
        if path in self._seeds_cache:
            return self._seeds_cache[path]
        if not path.exists():
            self._seeds_cache[path] = set()
            return set()
        data = json.loads(path.read_text())
        seeds = set(int(s) for s in data.get("success_seeds", []))
        self._seeds_cache[path] = seeds
        return seeds

    # --------------------------------------------------------- helpers
    def _pad_history(self, buf: deque, pad_value: torch.Tensor) -> list[torch.Tensor]:
        items = list(buf)
        if len(items) < self.num_hist:
            items = [pad_value] * (self.num_hist - len(items)) + items
        return items

    def _current_qpos(self, obs: Mapping) -> np.ndarray:
        """Pull 14-d qpos from the RoboTwin observation dict."""
        if "joint_action" in obs and "vector" in obs["joint_action"]:
            return np.asarray(obs["joint_action"]["vector"], dtype=np.float32)
        # Fall back: some task wrappers hand us a flat action vector already.
        if "qpos" in obs:
            return np.asarray(obs["qpos"], dtype=np.float32)
        raise KeyError(
            "DINO-WM policy needs obs['joint_action']['vector'] or obs['qpos']"
        )

    # ------------------------------------------------------- deploy API
    def reset_obs(self) -> None:
        self.frame_history.clear()
        self.action_history.clear()

    def update_obs(self, obs: Mapping) -> None:
        """Sample-level hook -- the MPC re-plans every step, so no buffering."""
        return

    @torch.no_grad()
    def get_action(self, obs: Mapping) -> list[np.ndarray]:
        if self.z_goal_visual is None:
            raise RuntimeError(
                "get_action called before a goal was loaded. Use "
                "load_goal_image / load_goal_for_seed first."
            )

        # --- assemble history of observed (visual, proprio) ----------
        rgb = obs["observation"][self.camera_key]["rgb"]
        current_pix = _preprocess_rgb(np.asarray(rgb), self.img_size).to(self.device)
        self.frame_history.append(current_pix)

        qpos = torch.from_numpy(self._current_qpos(obs)).to(self.device)
        qpos_norm = (qpos - self.action_mean) / self.action_std
        # proprio is the same setpoint proxy used at training time
        self.action_history.append(qpos_norm)

        pixels_hist = torch.stack(self._pad_history(self.frame_history, current_pix), dim=0)
        proprio_hist = torch.stack(self._pad_history(self.action_history, qpos_norm), dim=0)

        K = self.planning_iters
        H = self.num_hist
        T_plan = self.planning_horizon

        # (K, H, 3, S, S) + (K, H, A)
        pixels_batch = pixels_hist.unsqueeze(0).expand(K, -1, -1, -1, -1).contiguous()
        proprio_batch = proprio_hist.unsqueeze(0).expand(K, -1, -1).contiguous()
        obs_0 = {"visual": pixels_batch, "proprio": proprio_batch}

        # Past actions for the first num_hist steps come from history too;
        # candidate actions are sampled Gaussian in the *normalized* space
        # so action_scale is in std units. Denormalize before dispatch.
        past_act = torch.stack(self._pad_history(self.action_history, qpos_norm), dim=0)
        past_act = past_act.unsqueeze(0).expand(K, -1, -1).contiguous()      # (K, H, A)

        cand = torch.randn(K, T_plan, self.action_dim_raw, device=self.device) * self.action_scale
        act_batch = torch.cat([past_act, cand], dim=1)                       # (K, H + T_plan, A)

        z_obses, _ = self.model.rollout(obs_0, act_batch)
        # z_obses['visual']: (K, H + T_plan + 1, P, D)
        z_final = z_obses["visual"][:, -1:]                                  # (K, 1, P, D)
        z_goal = self.z_goal_visual.expand(K, -1, -1, -1)                    # (K, 1, P, D)
        cost = ((z_final - z_goal) ** 2).mean(dim=(1, 2, 3))                 # (K,)

        best = int(cost.argmin().item())
        first_norm = cand[best, 0]                                           # (A,)
        first = first_norm * self.action_std + self.action_mean              # denormalize
        action_np = first.detach().cpu().numpy().astype(np.float32)

        # Log the *normalized* action we just committed into history so the
        # next step's proprio matches training-time convention.
        self.action_history[-1] = first_norm.detach()
        return [action_np]
