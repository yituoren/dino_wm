#!/usr/bin/env bash
# Finetune DINO-WM per (task, split) from a pretrained ckpt.
#
# Mirrors ../oracle/run_finetune.sh for the DINO-WM baseline. Writes a
# deterministic per-(task, split) run dir at
#
#   ${CKPT_BASE}/outputs/ft_<task>_<N>ep/
#       hydra.yaml
#       checkpoints/model_<E>.pth
#
# so eval_client can auto-locate hydra.yaml via <ckpt>.parent.parent.
#
# Dataset layout (produced by sdlwm/scripts/preprocess_data.sh):
#   $FINETUNE_ROOT/<task>/<N>_episodes/<ep>/{frames,actions_raw}.npy
#
# Envs:
#   FINETUNE_FROM   required. Pretrained DINO-WM .pth (saved by train.py's
#                   save_ckpt, i.e. <run>/checkpoints/model_<E>.pth).
#   FINETUNE_ROOT   required. Root containing <task>/<N>_episodes/.
#   CKPT_BASE       where finetune runs land. Defaults to this dir.
#   TASKS           space-sep list; default = 4 reps from preprocess_data.sh.
#   SPLITS          space-sep list of N; default: 1 3 10 50.
#   EPOCHS_<N>      per-split epoch budget. Defaults: 1=60, 3=45, 10=30, 50=30.
#   EPOCHS_DEFAULT  fallback for splits not listed above. Default: 30.
#   DRY_RUN=1       print commands without running.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

export FINETUNE_FROM=~/lwm/reference/baseline/outputs/2026-04-21/22-36-56/checkpoints/model_latest.pth
export FINETUNE_ROOT=/cephfs/zhaorui/data/robotwin/dataset/robotwin_easy_finetune

: "${FINETUNE_FROM:?set FINETUNE_FROM to a pretrained checkpoints/model_*.pth}"
: "${FINETUNE_ROOT:?set FINETUNE_ROOT to the per-task dataset root}"
CKPT_BASE="${CKPT_BASE:-$HERE}"

if [ ! -f "$FINETUNE_FROM" ]; then
    echo "FINETUNE_FROM not found: $FINETUNE_FROM"; exit 1
fi
if [ ! -d "$FINETUNE_ROOT" ]; then
    echo "FINETUNE_ROOT not found: $FINETUNE_ROOT"; exit 1
fi

DEFAULT_TASKS=(
    click_bell
    beat_block_hammer
    handover_block
    blocks_ranking_size
)
read -r -a TASKS <<< "${TASKS:-${DEFAULT_TASKS[@]}}"
read -r -a SPLITS <<< "${SPLITS:-1 3 10 50}"

# Per-split epoch budget: tiny splits need more passes because each epoch
# only has a handful of batches. Override via EPOCHS_<N>, or EPOCHS_DEFAULT
# for any split not explicitly listed.
EPOCHS_1="${EPOCHS_1:-60}"
EPOCHS_3="${EPOCHS_3:-45}"
EPOCHS_10="${EPOCHS_10:-30}"
EPOCHS_50="${EPOCHS_50:-30}"
EPOCHS_DEFAULT="${EPOCHS_DEFAULT:-30}"

epochs_for_split() {
    local n="$1"
    local var="EPOCHS_${n}"
    if [ -n "${!var:-}" ]; then
        echo "${!var}"
    else
        echo "$EPOCHS_DEFAULT"
    fi
}

export FINETUNE_FROM FINETUNE_ROOT

echo "[finetune] pretrained=$FINETUNE_FROM"
echo "[finetune] dataset_root=$FINETUNE_ROOT"
echo "[finetune] ckpt_base=$CKPT_BASE"
echo "[finetune] tasks=${TASKS[*]}"
echo "[finetune] splits=${SPLITS[*]}"

for TASK in "${TASKS[@]}"; do
    for N in "${SPLITS[@]}"; do
        SPLIT_DIR="$FINETUNE_ROOT/$TASK/${N}_episodes"
        if [ ! -d "$SPLIT_DIR" ]; then
            echo "  [skip] $TASK/${N}_episodes: $SPLIT_DIR not found"
            continue
        fi

        export FT_TASK="$TASK" FT_SPLIT="$N"
        SUBDIR="ft_${TASK}_${N}ep"
        EPOCHS="$(epochs_for_split "$N")"
        echo
        echo "================================================================"
        echo "[finetune] task=$TASK split=$N epochs=$EPOCHS  ->  $SUBDIR"
        echo "================================================================"

        CMD=(
            python train.py
            --config-name=train_finetune
            subdir="$SUBDIR"
            training.epochs="$EPOCHS"
            ckpt_base_path="$CKPT_BASE"
        )
        if [ "${DRY_RUN:-0}" = "1" ]; then
            echo "DRY_RUN: ${CMD[*]}"
        else
            "${CMD[@]}"
        fi
    done
done

echo
echo "[finetune] done."
