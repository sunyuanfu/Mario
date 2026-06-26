#!/usr/bin/env bash
set -euo pipefail

# Paper-faithful two-stage training on Movies with multi-GPU Stage 2.
# Stage 1 exports embeddings only. Final task metrics are reported in Stage 2.

if [[ -z "${LLM_MODEL_PATH:-}" ]]; then
  echo "ERROR: Set LLM_MODEL_PATH to a local model path (e.g., Llama-3.1-8B-Instruct)." >&2
  exit 1
fi

DATA_ROOT="${DATA_ROOT:-../data/MAGB}"
GPUS="${GPUS:-0,1,2,3}"
STAGE2_RESUME="${STAGE2_RESUME:-}"

echo "DATA_ROOT=${DATA_ROOT}"
echo "LLM_MODEL_PATH=${LLM_MODEL_PATH}"
echo "GPUS=${GPUS}"

NPROC="$(python - <<'PY'
import os
g = os.environ.get("GPUS", "0")
print(len([x for x in g.split(",") if x.strip() != ""]))
PY
)"

pushd stage1 >/dev/null
DATA_ROOT="${DATA_ROOT}" python main.py \
  --dataset Movies \
  --use_large_features \
  --n_epochs 80 \
  --lr 0.001 \
  --lr_scheduler_gamma 0.95
popd >/dev/null

pushd stage2 >/dev/null

COMMON_ARGS=(
  --dataset Movies
  --data-root "${DATA_ROOT}"
  --stage1-feature ../stage1/Movies_stage1_mlp.pth
  --llm-model-path "${LLM_MODEL_PATH}"
  --epochs 10
  --batch-size 4
  --inference-policy router
  --output-dir runs/mario_movies_stage2
)

if [[ -n "${STAGE2_RESUME}" ]]; then
  COMMON_ARGS+=(--resume-checkpoint "${STAGE2_RESUME}")
fi

CUDA_VISIBLE_DEVICES="${GPUS}" torchrun --nproc_per_node="${NPROC}" train_llm.py "${COMMON_ARGS[@]}"
popd >/dev/null
