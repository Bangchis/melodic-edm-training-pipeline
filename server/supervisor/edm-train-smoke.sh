#!/bin/bash
set -eo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

export CUDA_VISIBLE_DEVICES=0,1
python3 - <<'PY'
import json
from pathlib import Path
report = Path('/workspace/melodic_edm_training_pipeline/data/tensor_validation_report.json')
if not report.is_file() or json.loads(report.read_text())['status'] != 'pass':
    raise SystemExit('tensor validation gate has not passed')
PY
cd /workspace/melodic_edm_training_pipeline/vendor/ACE-Step-1.5
smoke=/workspace/melodic_edm_training_pipeline/outputs/smoke
mkdir -p "$smoke"
# A smoke rerun must prove itself from fresh artifacts; never let a stale pass
# report unlock the main training job.
find "$smoke" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
: > "$smoke/gpu_metrics.csv"
(
  while true; do
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
      >> "$smoke/gpu_metrics.csv" 2>/dev/null || true
    sleep 2
  done
) &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT TERM INT

set +e
.venv/bin/python -u -m acestep.training_v2.cli.train_fixed --yes \
  --dataset-dir /workspace/melodic_edm_training_pipeline/data/tensors_all \
  --validation-dataset-dir /workspace/melodic_edm_training_pipeline/data/tensors_validation \
  --output-dir /workspace/melodic_edm_training_pipeline/outputs/smoke \
  --checkpoint-dir /workspace/melodic_edm_training_pipeline/checkpoints \
  --model-variant xl_base --base-model xl_base \
  --adapter-type lora --rank 32 --alpha 64 --dropout 0.1 \
  --lr 1e-4 --batch-size 1 --gradient-accumulation 8 --epochs 1 \
  --warmup-steps 10 --weight-decay 0.01 --optimizer-type adamw --scheduler-type cosine \
  --gradient-checkpointing --cfg-ratio 0.15 --shift 1.0 --num-inference-steps 50 \
  --num-devices 2 --strategy ddp --save-every 1 \
  --validate-every 1 --early-stopping-patience 0 \
  --log-every 10 --log-heavy-every 50 --sample-every-n-epochs 0 \
  2>&1 | tee "$smoke/training.log"
train_status=${PIPESTATUS[0]}
set -e
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
trap - EXIT TERM INT
if [ "$train_status" -ne 0 ]; then
  exit "$train_status"
fi

exec .venv/bin/python -u /workspace/melodic_edm_training_pipeline/scripts/validate_smoke.py \
  --project-root /workspace/melodic_edm_training_pipeline \
  --checkpoint-dir checkpoints \
  --reload-adapter
