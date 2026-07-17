#!/bin/bash
set -eo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

export CUDA_VISIBLE_DEVICES=0,1
python3 - <<'PY'
import json
from pathlib import Path
smoke = Path('/workspace/melodic_edm_training_pipeline/outputs/smoke/smoke_validation_report.json')
if not smoke.is_file() or json.loads(smoke.read_text())['status'] != 'pass':
    raise SystemExit('smoke validation gate has not passed')
PY
cd /workspace/melodic_edm_training_pipeline/vendor/ACE-Step-1.5
training=/workspace/melodic_edm_training_pipeline/outputs/training/melodic-edm-core-v1
mkdir -p "$training"
rm -f "$training/training_validation_report.json"
: > "$training/gpu_metrics.csv"
(
  while true; do
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
      >> "$training/gpu_metrics.csv" 2>/dev/null || true
    sleep 2
  done
) &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT TERM INT

resume_args=()
resume_path=$(python3 /workspace/melodic_edm_training_pipeline/scripts/latest_checkpoint.py "$training/checkpoints")
if [ -n "$resume_path" ]; then
  echo "[RESUME] $resume_path"
  resume_args=(--resume-from "$resume_path")
fi

set +e
.venv/bin/python -u -m acestep.training_v2.cli.train_fixed --yes \
  --dataset-dir /workspace/melodic_edm_training_pipeline/data/tensors_all \
  --validation-dataset-dir /workspace/melodic_edm_training_pipeline/data/tensors_validation \
  --output-dir /workspace/melodic_edm_training_pipeline/outputs/training/melodic-edm-core-v1 \
  --checkpoint-dir /workspace/melodic_edm_training_pipeline/checkpoints \
  --model-variant xl_base --base-model xl_base \
  --adapter-type lora --rank 32 --alpha 64 --dropout 0.1 \
  --lr 1e-4 --batch-size 1 --gradient-accumulation 8 --epochs 150 \
  --warmup-steps 100 --weight-decay 0.01 --optimizer-type adamw --scheduler-type cosine \
  --gradient-checkpointing --cfg-ratio 0.15 --shift 1.0 --num-inference-steps 50 \
  --num-devices 2 --strategy ddp --save-every 5 \
  --validate-every 5 --early-stopping-patience 5 \
  --log-every 10 --log-heavy-every 50 --sample-every-n-epochs 10 \
  "${resume_args[@]}" 2>&1 | tee -a "$training/training.log"
train_status=${PIPESTATUS[0]}
set -e
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
trap - EXIT TERM INT
if [ "$train_status" -ne 0 ]; then
  exit "$train_status"
fi

exec .venv/bin/python -u /workspace/melodic_edm_training_pipeline/scripts/validate_training.py \
  --project-root /workspace/melodic_edm_training_pipeline
