#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

export CUDA_VISIBLE_DEVICES=0,1
project=/workspace/melodic_edm_training_pipeline
ace="$project/vendor/ACE-Step-1.5-v2"
export PYTHONPATH="$ace${PYTHONPATH:+:$PYTHONPATH}"
output="$project/outputs/v2/smoke"
python3 - <<'PY'
import json
from pathlib import Path
report = Path('/workspace/melodic_edm_training_pipeline/data_v2/tensor_validation_report.json')
if not report.is_file() or json.loads(report.read_text())['status'] != 'pass':
    raise SystemExit('v2 tensor validation gate has not passed')
output = Path('/workspace/melodic_edm_training_pipeline/outputs/v2/smoke')
if (output / 'final').exists():
    raise SystemExit('refusing to overwrite an existing v2 smoke run')
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

common=(
  --yes
  --dataset-dir "$project/data_v2/tensors_train"
  --validation-dataset-dir "$project/data_v2/tensors_validation"
  --output-dir "$output"
  --checkpoint-dir "$project/checkpoints"
  --model-variant xl_base --base-model xl_base
  --adapter-type lora --rank 32 --alpha 32 --dropout 0.1
  --target-modules q_proj k_proj v_proj o_proj --attention-type both --strict-attention-scope
  --lr 5e-5 --batch-size 1 --gradient-accumulation 8
  --warmup-steps 25 --weight-decay 0.01 --optimizer-type adamw --scheduler-type cosine
  --gradient-checkpointing --cfg-ratio 0.15 --shift 1.0 --num-inference-steps 50
  --num-devices 2 --strategy ddp --validate-every 5 --early-stopping-patience 0
  --log-every 5 --log-heavy-every 25 --sample-every-n-epochs 0
)
cd "$project"
set +e
"$ace/.venv/bin/python" -u -m acestep.training_v2.cli.train_fixed \
  "${common[@]}" --epochs 5 --save-every 5 2>&1 | tee "$output/training.log"
first_status=${PIPESTATUS[0]}
set -e
if [ "$first_status" -ne 0 ]; then
  exit "$first_status"
fi
resume_path=$(python3 "$project/scripts/latest_checkpoint.py" "$output/checkpoints")
if [ -z "$resume_path" ]; then
  echo "No resumable smoke checkpoint found" >&2
  exit 1
fi
set +e
"$ace/.venv/bin/python" -u -m acestep.training_v2.cli.train_fixed \
  "${common[@]}" --epochs 6 --max-steps 66 --save-every 5 \
  --resume-from "$resume_path" 2>&1 | tee -a "$output/training.log"
second_status=${PIPESTATUS[0]}
set -e
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
trap - EXIT TERM INT
if [ "$second_status" -ne 0 ]; then
  exit "$second_status"
fi
exec "$ace/.venv/bin/python" -u "$project/scripts/validate_smoke_v2.py" \
  --project-root "$project" --reload-adapter
