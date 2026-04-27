"""DINO-WM inference wrapper for RoboTwin closed-loop eval.

Rehydrates the VWorldModel from a training checkpoint + its saved hydra
config (``<saved_folder>/hydra.yaml``), then drives it with the *original*
DINO-WM planner from ``planning/`` (CEMPlanner or GDPlanner) instead of an
ad-hoc random-shooting loop. Exposes the ``reset_obs / update_obs /
get_action`` contract that ``RoboTwin/policy/DINOWM/deploy_policy.py``
forwards to.

The original DINO-WM planners assume ``obs_0['visual'].shape[1] == 1`` (a
single first observation per traj) -- their offline plan.py prepares
targets that way. RoboTwin's online MPC instead carries num_hist past
obs/proprio frames, so we wrap the world model in ``_RolloutAdapter``
that prepends the stored past actions inside ``rollout(obs_0, act)``
before delegating to the real ``VWorldModel.rollout``. The planner is
unmodified.

Key shape contract matches the dataset at ``datasets/robotwin_dset.py``:
  * Visuals: (T, 3, img_size, img_size) float, Normalize(0.5, 0.5) -> [-1, 1]
  * Actions: (T, action_dim), normalized to zero-mean/unit-std if the
    training config used normalize_action=True
  * Proprio: same as action (setpoint proxy; see dataset docstring)

Action_dim in the checkpoint already includes the DINO-WM fs-concat factor
(``dataset.action_dim * frameskip``), so a config trained with frameskip=1
yields action_dim=14 for aloha-agilex.
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

    cfg = OmegaConf.load(str(hydra_cfg_path))

    ckpt = torch.load(str(ckpt_path), map_location=device)
    modules = {k: ckpt[k].to(device) for k in _MODEL_KEYS if k in ckpt and ckpt[k] is not None}
    bundled_mean = ckpt.get("action_mean")
    bundled_std = ckpt.get("action_std")

    encoder = modules.get("encoder")
    if encoder is None:
        encoder = hydra.utils.instantiate(cfg.encoder).to(device)

    model = hydra.utils.instantiate(
        cfg.model,
        encoder=encoder,
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


# --------------------------------------------------------- planner stubs

class _NullWandb:
    """Minimal stub matching the wandb_run.log contract used by the planners."""
    def log(self, *args, **kwargs):
        return


class _IdentityPreprocessor:
    """Pass-through preprocessor.

    The wrapper already converts RoboTwin observations into the model's
    input space (pixels in [-1, 1], proprio normalized) before they reach
    the planner, so the planner-side preprocessor is just an identity over
    obs and a no-op over actions (only used by GDPlanner's sample_type=
    'zero' branch, which we don't trigger).
    """
    def transform_obs(self, obs):
        return obs

    def normalize_actions(self, actions):
        return actions

    def denormalize_actions(self, actions):
        return actions


class _RolloutAdapter:
    """Wraps the DINO-WM world model so the original CEM/GD planners can
    drive online MPC over a num_hist-long observation window.

    The original ``VWorldModel.rollout(obs_0, act)`` reads
    ``num_obs_init = obs_0['visual'].shape[1]`` and treats
    ``act[:, :num_obs_init]`` as the past actions paired with each frame.
    Our planners only sample the *future* action sequence, so this adapter
    holds the past actions externally and prepends them inside ``rollout``
    before delegating. ``encode_obs`` is a passthrough.

    Set ``past_act`` (shape ``(B, num_hist, action_dim)``) before each
    planner call; the adapter expands it to match the candidate batch.
    """

    def __init__(self, wm: torch.nn.Module):
        self.wm = wm
        self.past_act: torch.Tensor | None = None  # (1, num_hist, A_chunk)

    @property
    def num_hist(self) -> int:
        return int(self.wm.num_hist)

    def parameters(self):
        return self.wm.parameters()

    def encode_obs(self, obs):
        return self.wm.encode_obs(obs)

    def rollout(self, obs_0, act):
        if self.past_act is None:
            raise RuntimeError("_RolloutAdapter.past_act must be set before rollout")
        B = act.shape[0]
        past = self.past_act.expand(B, -1, -1).to(act.dtype)
        full = torch.cat([past, act], dim=1)
        return self.wm.rollout(obs_0, full)


# --------------------------------------------------------------- policy

class DINOWMRoboTwinPolicy:
    """CEM/Adam-MPC driver for a trained DINO-WM VWorldModel."""

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
        solver: str = "cem",
        planning_horizon: int = 5,
        # CEM knobs (mirror conf/planner/cem.yaml defaults from the original repo).
        cem_num_samples: int = 300,
        cem_topk: int = 30,
        cem_var_scale: float = 1.0,
        cem_opt_steps: int = 30,
        # GD knobs (mirror conf/planner/gd.yaml).
        gd_lr: float = 0.1,
        gd_action_noise: float = 0.0,
        gd_opt_steps: int = 30,
        gd_sample_type: str = "randn",
        objective_alpha: float = 1.0,
    ):
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.camera_key = camera_key
        self.goal_bank_root = Path(goal_bank_root) if goal_bank_root else None
        self._task_name_hint = task_name
        self._seeds_cache: dict[Path, set[int]] = {}

        # Make sure ``import datasets`` / ``import models`` / ``import planning``
        # used by the hydra targets and the planner imports resolve to the
        # baseline package.
        baseline_root = Path(__file__).resolve().parent
        if str(baseline_root) not in sys.path:
            sys.path.insert(0, str(baseline_root))

        self.model, self.cfg, bundled_mean, bundled_std = _load_model_from_ckpt(
            Path(ckpt_path), Path(hydra_cfg_path), self.device
        )
        self.img_size = int(self.cfg.img_size)
        self.num_hist = int(self.cfg.num_hist)
        self.frameskip = int(self.cfg.frameskip)
        self.action_dim_per_step = int(self.model.action_encoder.in_chans)
        self.action_dim_raw = self.action_dim_per_step // self.frameskip

        self.planning_horizon = int(planning_horizon)

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
                    "was provided to recompute them."
                )
        else:
            self.action_mean = torch.zeros(self.action_dim_raw, device=self.device)
            self.action_std = torch.ones(self.action_dim_raw, device=self.device)

        # --- planner setup (mirrors planning/cem.py + planning/gd.py) ----
        from planning.cem import CEMPlanner
        from planning.gd import GDPlanner
        from planning.objectives import create_objective_fn

        self._wm_adapter = _RolloutAdapter(self.model)
        self._objective_fn = create_objective_fn(
            alpha=float(objective_alpha), base=1.0, mode="last"
        )
        self._wandb = _NullWandb()
        self._preprocessor = _IdentityPreprocessor()

        solver = solver.lower()
        if solver == "cem":
            self._planner = CEMPlanner(
                horizon=self.planning_horizon,
                topk=int(cem_topk),
                num_samples=int(cem_num_samples),
                var_scale=float(cem_var_scale),
                opt_steps=int(cem_opt_steps),
                eval_every=10**9,  # never trigger the evaluator branch
                wm=self._wm_adapter,
                action_dim=self.action_dim_per_step,
                objective_fn=self._objective_fn,
                preprocessor=self._preprocessor,
                evaluator=None,
                wandb_run=self._wandb,
                logging_prefix="robotwin_cem",
                log_filename=None,
            )
        elif solver in ("gd", "adam", "sgd"):
            self._planner = GDPlanner(
                horizon=self.planning_horizon,
                action_noise=float(gd_action_noise),
                sample_type=gd_sample_type,
                lr=float(gd_lr),
                opt_steps=int(gd_opt_steps),
                eval_every=10**9,
                wm=self._wm_adapter,
                action_dim=self.action_dim_per_step,
                objective_fn=self._objective_fn,
                preprocessor=self._preprocessor,
                evaluator=None,
                wandb_run=self._wandb,
                logging_prefix="robotwin_gd",
                log_filename=None,
            )
        else:
            raise ValueError(f"unknown solver={solver!r}; expected 'cem' or 'gd'")
        self.solver = solver

        self.z_goal_visual: torch.Tensor | None = None
        self._obs_g: dict | None = None
        if goal_image_path is not None:
            self.load_goal_image(goal_image_path)

        self.frame_history: deque = deque(maxlen=self.num_hist)
        self.proprio_history: deque = deque(maxlen=self.num_hist)
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
        self._obs_g = obs_g

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
        if "joint_action" in obs and "vector" in obs["joint_action"]:
            return np.asarray(obs["joint_action"]["vector"], dtype=np.float32)
        if "qpos" in obs:
            return np.asarray(obs["qpos"], dtype=np.float32)
        raise KeyError(
            "DINO-WM policy needs obs['joint_action']['vector'] or obs['qpos']"
        )

    # ------------------------------------------------------- deploy API
    def reset_obs(self) -> None:
        self.frame_history.clear()
        self.proprio_history.clear()
        self.action_history.clear()

    def update_obs(self, obs: Mapping) -> None:
        return

    def get_action(self, obs: Mapping) -> list[np.ndarray]:
        if self.z_goal_visual is None or self._obs_g is None:
            raise RuntimeError(
                "get_action called before a goal was loaded. Use "
                "load_goal_image / load_goal_for_seed first."
            )

        # --- assemble history ---------------------------------------------
        rgb = obs["observation"][self.camera_key]["rgb"]
        current_pix = _preprocess_rgb(np.asarray(rgb), self.img_size).to(self.device)
        self.frame_history.append(current_pix)

        qpos = torch.from_numpy(self._current_qpos(obs)).to(self.device)
        qpos_norm = (qpos - self.action_mean) / self.action_std
        self.proprio_history.append(qpos_norm)

        pixels_hist = torch.stack(
            self._pad_history(self.frame_history, current_pix), dim=0
        )  # (H, 3, S, S)
        proprio_hist = torch.stack(
            self._pad_history(self.proprio_history, qpos_norm), dim=0
        )  # (H, A_raw)

        zero_chunk = torch.zeros(self.action_dim_per_step, device=self.device)
        past_chunks = self._pad_history(self.action_history, zero_chunk)
        past_act = torch.stack(past_chunks, dim=0).unsqueeze(0).contiguous()  # (1, H, A_chunk)

        # --- planner inputs (B=1) ----------------------------------------
        obs_0 = {
            "visual": pixels_hist.unsqueeze(0),                     # (1, H, 3, S, S)
            "proprio": proprio_hist.unsqueeze(0),                   # (1, H, A_raw)
        }
        # The planner reads obs_g['visual'].shape only to derive z_obs_g via
        # ``wm.encode_obs``; the rest of obs_g is ignored after encoding.
        # We supply a (1, 1, ...) goal frame, matching the original eval shape.
        obs_g = self._obs_g

        # Stash past actions so the rollout adapter can reconstruct
        # ``act_full = [past_act, sampled_future]`` inside ``wm.rollout``.
        self._wm_adapter.past_act = past_act

        try:
            actions, _ = self._planner.plan(obs_0=obs_0, obs_g=obs_g, actions=None)
        finally:
            self._wm_adapter.past_act = None

        # CEMPlanner returns the running mean (B, T_plan, A_chunk); GDPlanner
        # returns the optimized actions (n_evals, T_plan, A_chunk). For
        # n_evals=1 we pull the first traj's first chunk and execute it.
        best_chunk_norm = actions[0, 0].detach()                       # (A_chunk,)
        self.action_history.append(best_chunk_norm.clone())

        per_step_norm = best_chunk_norm.view(self.frameskip, self.action_dim_raw)
        per_step = per_step_norm * self.action_std + self.action_mean
        return [per_step[i].cpu().numpy().astype(np.float32)
                for i in range(self.frameskip)]
