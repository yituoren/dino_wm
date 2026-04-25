#!/usr/bin/env bash
# Sweep eval_client.py across all (split, task) DINO-WM finetune runs.
#
# Expects run_finetune.sh's layout:
#   $CKPT_ROOT/ft_<task>_<N>ep/hydra.yaml
#   $CKPT_ROOT/ft_<task>_<N>ep/checkpoints/model_*.pth
#
# Newer ckpts (post action-stat baking in train.py:save_ckpt) carry their
# action_mean/action_std internally, so eval_client only needs the ckpt
# dir. When FINETUNE_ROOT is set we still build a per-task data symlink
# tree so robotwin_policy can recompute stats for older ckpts.
#
#     $tmp_ckpt/<task>  ->  $CKPT_ROOT/ft_<task>_<N>ep/
#     $tmp_data/<task>  ->  $FINETUNE_ROOT/<task>/<N>_episodes/   (optional)
#
# and pass `--ckpt-dir $tmp_ckpt --ckpt-glob <pat>` (+ `--data-path $tmp_data`
# when FINETUNE_ROOT is set).
#
# Server must already be running at $HOST:$PORT -- this script only drives
# the client side. Each split spawns one eval_client run (which internally
# iterates all tasks), logs to $LOG_DIR/split_<N>.log, and the per-task
# summary lines are parsed at the end into a split x task matrix.
#
# Envs:
#   GOALS_ROOT      required. --goals-root for eval_client.
#   CKPT_ROOT       required. Dir containing ft_<task>_<N>ep/ subdirs
#                   (default layout: $CKPT_BASE/outputs/).
#   FINETUNE_ROOT   optional. Dataset root containing <task>/<N>_episodes/.
#                   Only needed for old ckpts that don't bundle action
#                   stats; new ckpts carry mean/std and don't need this.
#   TASKS           space-sep list; default = 4 reps.
#   SPLITS          space-sep list; default: 1 3 10 50.
#   CKPT_GLOB       picks which ckpt inside each ft_<task>_<N>ep/. Default:
#                   'checkpoints/model_latest.pth'.
#   SUBDIR_TEMPLATE override ckpt subdir name; default ft_{task}_{N}ep.
#   LOG_DIR         where per-split client logs land. Default: ./eval_logs.
#   HOST / PORT     server address; default 127.0.0.1 / 9765.
#   DEVICE          --device; default cuda.
#   EXTRA_ARGS      appended verbatim to each eval_client call.
#   DRY_RUN=1       print commands without running.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

export GOALS_ROOT=~/lwm/goals
export CKPT_ROOT=~/lwm/checkpoints/dino

: "${GOALS_ROOT:?set GOALS_ROOT to the goal bank root (--goals-root)}"
: "${CKPT_ROOT:?set CKPT_ROOT to the dir containing ft_<task>_<N>ep/}"
# FINETUNE_ROOT is optional: train.py:save_ckpt now bakes per-step
# action_mean/action_std into the ckpt, so eval-only machines without the
# finetune dataset on disk can run by leaving this unset. We still wire it
# up if provided, so old (pre-bake) ckpts keep working.
FINETUNE_ROOT="${FINETUNE_ROOT:-}"

[ -d "$CKPT_ROOT" ] || { echo "CKPT_ROOT not a dir: $CKPT_ROOT"; exit 1; }
[ -d "$GOALS_ROOT" ] || { echo "GOALS_ROOT not a dir: $GOALS_ROOT"; exit 1; }
if [ -n "$FINETUNE_ROOT" ] && [ ! -d "$FINETUNE_ROOT" ]; then
    echo "FINETUNE_ROOT set but not a dir: $FINETUNE_ROOT"; exit 1
fi

DEFAULT_TASKS=(
    click_bell
    beat_block_hammer
    handover_block
    blocks_ranking_size
)
read -r -a TASKS <<< "${TASKS:-${DEFAULT_TASKS[@]}}"
read -r -a SPLITS <<< "${SPLITS:-1 3 10 50}"
CKPT_GLOB="${CKPT_GLOB:-checkpoints/model_latest.pth}"
# See ../oracle/compute_action_stats.sh for why this isn't done via ${VAR:-...}.
if [ -z "${SUBDIR_TEMPLATE:-}" ]; then
    SUBDIR_TEMPLATE='ft_{task}_{N}ep'
fi
LOG_DIR="${LOG_DIR:-$HERE/eval_logs}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-9765}"
DEVICE="${DEVICE:-cuda}"
mkdir -p "$LOG_DIR"

read -r -a EXTRA_ARGS_ARR <<< "${EXTRA_ARGS:-}"

resolve_subdir() {
    local task="$1" n="$2"
    printf '%s\n' "$SUBDIR_TEMPLATE" | sed -e "s/{task}/$task/g" -e "s/{N}/$n/g"
}

run_split() {
    local N="$1"
    local tmp_ckpt tmp_data
    tmp_ckpt="$(mktemp -d)"
    if [ -n "$FINETUNE_ROOT" ]; then
        tmp_data="$(mktemp -d)"
        trap 'rm -rf "$tmp_ckpt" "$tmp_data"' RETURN
    else
        tmp_data=""
        trap 'rm -rf "$tmp_ckpt"' RETURN
    fi
    local log="$LOG_DIR/split_${N}.log"

    local any_run=0
    local present_tasks=()
    for TASK in "${TASKS[@]}"; do
        local sub; sub="$(resolve_subdir "$TASK" "$N")"
        local ckpt_src="$CKPT_ROOT/$sub"
        local data_src=""
        if [ -n "$FINETUNE_ROOT" ]; then
            data_src="$FINETUNE_ROOT/$TASK/${N}_episodes"
        fi
        if [ ! -d "$ckpt_src" ]; then
            continue
        fi
        # eval_client needs at least one ckpt matching CKPT_GLOB inside ckpt_src.
        shopt -s nullglob
        local matches=("$ckpt_src"/$CKPT_GLOB)
        shopt -u nullglob
        if [ "${#matches[@]}" -eq 0 ]; then
            echo "  [skip] $sub: no ckpt matching '$CKPT_GLOB'"
            continue
        fi
        # hydra.yaml is resolved by eval_client as <ckpt>.parent.parent/hydra.yaml,
        # which through the symlink means $ckpt_src/hydra.yaml.
        if [ ! -f "$ckpt_src/hydra.yaml" ]; then
            echo "  [skip] $sub: no hydra.yaml (training run wasn't saved?)"
            continue
        fi
        # When FINETUNE_ROOT is set, require the per-task data dir; otherwise
        # rely on the ckpt-bundled action stats and skip the data symlink.
        if [ -n "$FINETUNE_ROOT" ]; then
            if [ ! -d "$data_src" ]; then
                echo "  [skip] $sub: data_src missing ($data_src); policy can't recompute stats"
                continue
            fi
            ln -sfn "$data_src" "$tmp_data/$TASK"
        fi
        ln -sfn "$ckpt_src" "$tmp_ckpt/$TASK"
        present_tasks+=("$TASK")
        any_run=1
    done
    if [ "$any_run" -eq 0 ]; then
        echo "[eval] split=$N: no runnable (task, ckpt) pairs, skipping"
        return
    fi

    local task_args=()
    for TASK in "${present_tasks[@]}"; do
        task_args+=(--task "$TASK")
    done

    local CMD=(
        python eval_client.py
        --host "$HOST" --port "$PORT"
        --goals-root "$GOALS_ROOT"
        --ckpt-dir "$tmp_ckpt"
        --ckpt-glob "$CKPT_GLOB"
        --device "$DEVICE"
        "${task_args[@]}"
    )
    if [ -n "$tmp_data" ]; then
        CMD+=(--data-path "$tmp_data")
    fi
    if [ "${#EXTRA_ARGS_ARR[@]}" -gt 0 ]; then
        CMD+=("${EXTRA_ARGS_ARR[@]}")
    fi

    echo
    echo "================================================================"
    echo "[eval] split=$N tasks=${present_tasks[*]}"
    echo "       log -> $log"
    echo "================================================================"
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "DRY_RUN: ${CMD[*]}"
    else
        "${CMD[@]}" 2>&1 | tee "$log"
    fi
}

for N in "${SPLITS[@]}"; do
    run_split "$N"
done

# ------------------------------------------------------------ summary
# eval_client prints per-task lines like `  <task_padded>  S/R = XX.X%`
# in the per-task summary block. Grep them out into a matrix.
if [ "${DRY_RUN:-0}" = "1" ]; then
    exit 0
fi

echo
echo "================================================================"
echo "[eval] split x task summary (success / run = rate)"
echo "================================================================"

printf "%-28s" "task"
for N in "${SPLITS[@]}"; do
    printf " %12s" "${N}ep"
done
echo

for TASK in "${TASKS[@]}"; do
    printf "%-28s" "$TASK"
    for N in "${SPLITS[@]}"; do
        LOG="$LOG_DIR/split_${N}.log"
        if [ ! -f "$LOG" ]; then
            printf " %12s" "-"
            continue
        fi
        LINE="$(awk -v t="$TASK" '
            $0 ~ ("^  " t "[[:space:]]+[0-9]+/[0-9]+[[:space:]]*=") { print; exit }
        ' "$LOG")"
        if [ -z "$LINE" ]; then
            printf " %12s" "-"
        else
            SR="$(echo "$LINE" | awk '{for (i=1;i<=NF;i++) if ($i ~ /^[0-9]+\/[0-9]+$/) {print $i; exit}}')"
            RATE="$(echo "$LINE" | awk '{for (i=1;i<=NF;i++) if ($i ~ /%$/) {print $i; exit}}')"
            printf " %12s" "${SR:-?}(${RATE:-?})"
        fi
    done
    echo
done

echo
echo "[eval] done. per-split logs in $LOG_DIR/"
