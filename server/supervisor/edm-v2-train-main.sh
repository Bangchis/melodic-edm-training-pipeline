#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

export CUDA_VISIBLE_DEVICES=0,1
project=/workspace/melodic_edm_training_pipeline
ace="$project/vendor/ACE-Step-1.5-v2"
output="$project/outputs/v2/train-validation"
python3 - <<'PY'
import json
from pathlib import Path
smoke = Path('/workspace/melodic_edm_training_pipeline/outputs/v2/smoke/smoke_validation_report.json')
if not smoke.is_file() or json.loads(smoke.read_text())['status'] != 'pass':
    raise SystemExit('v2 smoke validation gate has not passed')
PY
mkdir -p "$output"
: > "$output/gpu_metrics.csv"
(
  while true; do
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
      >> "$output/gpu_metrics.csv" 2>/dev/null || true
    sleep 2
  done
) &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT TERM INT

resume_args=()
resume_path=$(python3 "$project/scripts/latest_checkpoint.py" "$output/checkpoints")
if [ -n "$resume_path" ]; then
  echo "[RESUME] $resume_path"
  resume_args=(--resume-from "$resume_path")
fi
cd "$project"
set +e
"$ace/.venv/bin/python" -u -m acestep.training_v2.cli.train_fixed --yes \
  --dataset-dir "$project/data_v2/tensors_train" \
  --validation-dataset-dir "$project/data_v2/tensors_validation" \
  --output-dir "$output" \
  --checkpoint-dir "$project/checkpoints" \
  --model-variant xl_base --base-model xl_base \
  --adapter-type lora --rank 48 --alpha 96 --dropout 0.1 \
  --target-modules q_proj k_proj v_proj o_proj --attention-type both --strict-attention-scope \
  --lr 7.5e-5 --batch-size 1 --gradient-accumulation 8 --epochs 150 \
  --warmup-steps 75 --weight-decay 0.01 --optimizer-type adamw --scheduler-type cosine \
  --gradient-checkpointing --cfg-ratio 0.15 --shift 1.0 --num-inference-steps 50 \
  --num-devices 2 --strategy ddp --save-every 5 \
  --validate-every 5 --early-stopping-patience 5 \
  --log-every 10 --log-heavy-every 50 --sample-every-n-epochs 10 \
  "${resume_args[@]}" 2>&1 | tee -a "$output/training.log"
train_status=${PIPESTATUS[0]}
set -e
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
trap - EXIT TERM INT
if [ "$train_status" -ne 0 ]; then
  exit "$train_status"
fi
exec "$ace/.venv/bin/python" -u "$project/scripts/validate_training_v2.py" \
  --project-root "$project"
