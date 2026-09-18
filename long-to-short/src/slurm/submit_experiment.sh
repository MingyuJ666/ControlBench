#!/usr/bin/env bash
# Submit one execution track. Outputs use configurable scratch; logs use src/slurm/out.
set -euo pipefail

project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
scratch=${L2S_SCRATCH_ROOT:-$project/.cache/l2s}
track=${L2S_TRACK:-}
eval_sets=${L2S_EVAL_SETS:-math500}
max_new_tokens=${L2S_MAX_NEW_TOKENS:-3072}
zero_check=${L2S_ZERO_CHECK:-false}
through=grid
dry_run=false
while (( $# )); do
  case "$1" in
    --track) track=${2:?--track requires iti or non-iti}; shift 2 ;;
    --dry-run) dry_run=true; shift ;;
    --zero-check) zero_check=true; shift ;;
    --through) through=${2:?--through requires prepare, grid or report}; shift 2 ;;
    --include-amc12) eval_sets="math500 amc12"; shift ;;
    --max-new-tokens) max_new_tokens=${2:?--max-new-tokens requires an integer}; shift 2 ;;
    *) echo "Usage: $0 --track {iti|non-iti} [--include-amc12] [--max-new-tokens N] [--zero-check] [--through {prepare|grid|report}] [--dry-run]" >&2; exit 2 ;;
  esac
done
case "$track" in
  iti|non-iti) ;;
  *) echo "Select --track iti or --track non-iti (or use the dedicated submit wrapper)." >&2; exit 2 ;;
esac
case "$eval_sets" in
  math500|"math500 amc12") ;;
  *) echo 'L2S_EVAL_SETS must be "math500" or "math500 amc12".' >&2; exit 2 ;;
esac
if [[ ! "$max_new_tokens" =~ ^[0-9]+$ ]] || [[ "$max_new_tokens" =~ ^0+$ ]]; then
  echo "--max-new-tokens must be a positive integer." >&2; exit 2
fi
# Normalize leading zeroes and cap before arithmetic to avoid integer overflow.
while [[ ${#max_new_tokens} -gt 1 && "$max_new_tokens" == 0* ]]; do max_new_tokens=${max_new_tokens#0}; done
if [[ ${#max_new_tokens} -gt 4 ]] || (( 10#$max_new_tokens > 3072 )); then max_new_tokens=3072; fi
case "$through" in prepare|grid|report) ;; *) echo "Invalid --through: $through" >&2; exit 2 ;; esac
case "$zero_check" in true|false) ;; *) echo "L2S_ZERO_CHECK must be true or false" >&2; exit 2 ;; esac
run_root=${L2S_RUN_ROOT:-$scratch/l2s-$track}
cd "$project"
if ! "$dry_run" && [[ "${L2S_SUBMIT_CONFIRM:-}" != "I_UNDERSTAND_FULL_GRID" ]]; then
  echo "Set L2S_SUBMIT_CONFIRM=I_UNDERSTAND_FULL_GRID after reviewing the $track matrix." >&2
  exit 2
fi
subject_gpu_args=(--gres=gpu:1)
judge_gpu_args=(--gres=gpu:1)
if ! "$dry_run"; then
  mkdir -p "$project/src/slurm/out" "$run_root" "$scratch/tmp"
fi

batch() {
  if "$dry_run"; then
    printf 'sbatch' >&2
    printf ' %q' "$@" >&2
    printf '\n' >&2
    echo DRY_RUN
  else
    sbatch --parsable "$@"
  fi
}

submit_stage() {
  local dependency=$1 stage=$2 model=$3
  shift 3
  # Clear inherited shard filters for fit, baselines and zero-check jobs.
  local methods= layers= strengths= shard= value
  for value in "$@"; do
    case "$value" in
      L2S_METHODS=*) methods=${value#*=} ;;
      L2S_LAYERS=*) layers=${value#*=} ;;
      L2S_STRENGTHS=*) strengths=${value#*=} ;;
      L2S_SHARD=*) shard=${value#*=} ;;
      *) echo "Unknown stage setting: $value" >&2; return 2 ;;
    esac
  done
  local exports="ALL,L2S_SCRATCH_ROOT=$scratch,L2S_RUN_ROOT=$run_root,L2S_TRACK=$track,L2S_EVAL_SETS=$eval_sets,L2S_MAX_NEW_TOKENS=$max_new_tokens,L2S_ZERO_CHECK=$zero_check,L2S_STAGE=$stage,L2S_MODEL=$model,L2S_METHODS=$methods,L2S_LAYERS=$layers,L2S_STRENGTHS=$strengths,L2S_SHARD=$shard"
  batch "${subject_gpu_args[@]}" --dependency="afterok:${dependency}" \
    --job-name="l2s-$track-$model-$stage${shard:+-$shard}" \
    --export="$exports" src/slurm/run_stage.sbatch
}

submit_final() {
  local stage=$1 dependency=$2
  local -a gpu_args=()
  if [[ "$stage" == "judge" ]]; then gpu_args=("${judge_gpu_args[@]}"); fi
  batch "${gpu_args[@]}" --dependency="afterok:${dependency}" --job-name="l2s-$track-$stage" \
    --export="ALL,L2S_SCRATCH_ROOT=$scratch,L2S_RUN_ROOT=$run_root,L2S_TRACK=$track,L2S_EVAL_SETS=$eval_sets,L2S_MAX_NEW_TOKENS=$max_new_tokens" "src/slurm/$stage.sbatch"
}

echo "track=$track root=$run_root eval_sets=$eval_sets max_new_tokens=$max_new_tokens zero_check=$zero_check through=$through dry_run=$dry_run"
setup=$(batch "${subject_gpu_args[@]}" --export="ALL,L2S_SCRATCH_ROOT=$scratch" src/slurm/setup_gpu.sbatch)
echo "setup=$setup"
models=("Llama-3.1-8B" "Qwen3-8B" "Gemma-2-9B")
fit_ids=()
baseline_ids=()
fit_stage=fit
if "$zero_check"; then fit_stage=fit-zero; fi
for model in "${models[@]}"; do
  baseline=$(submit_stage "$setup" baselines "$model")
  fit=$(submit_stage "$setup" "$fit_stage" "$model")
  baseline_ids+=("$baseline")
  fit_ids+=("$fit")
  echo "$fit_stage=$fit baselines=$baseline model=$model"
done

if [[ "$through" == prepare ]]; then exit 0; fi

all_grid_ids=()
for model_index in 0 1 2; do
  model=${models[$model_index]}
  prereq=${fit_ids[$model_index]}
  if [[ "$track" == "iti" ]]; then
    id=$(submit_stage "$prereq" grid "$model" "L2S_METHODS=ITI" "L2S_SHARD=ITI-global")
    all_grid_ids+=("$id")
    echo "grid=$id model=$model method=ITI profile=global"
    continue
  fi
  case "$model" in
    Llama-3.1-8B) layers=(8 16 24) ;;
    Qwen3-8B) layers=(9 18 27) ;;
    Gemma-2-9B) layers=(11 21 32) ;;
  esac
  for method in CAA PCA LAT Linear-Probe; do
    for layer in "${layers[@]}"; do
      id=$(submit_stage "$prereq" grid "$model" \
        "L2S_METHODS=$method" "L2S_LAYERS=$layer" "L2S_SHARD=${method}-layer-${layer}")
      all_grid_ids+=("$id")
      echo "grid=$id model=$model method=$method layer=$layer"
    done
  done
  if [[ "$model" == "Gemma-2-9B" ]]; then
    for layer in 9 20 31; do
      id=$(submit_stage "$prereq" grid "$model" \
        "L2S_METHODS=SAE" "L2S_LAYERS=$layer" "L2S_SHARD=SAE-layer-${layer}")
      all_grid_ids+=("$id")
      echo "grid=$id model=$model method=SAE layer=$layer"
    done
  fi
done

echo "submitted ${#all_grid_ids[@]} grid shards for $track; logs: $project/src/slurm/out"
if [[ "$through" == grid ]]; then exit 0; fi
final_ids=("${baseline_ids[@]}" "${all_grid_ids[@]}")
grid_dependency=$(IFS=:; echo "${final_ids[*]}")
judge=$(submit_final judge "$grid_dependency")
echo "judge=$judge"
report=$(submit_final report "$judge")
echo "report=$report"
echo "submitted ${#all_grid_ids[@]} grid shards for $track; logs: $project/src/slurm/out"
