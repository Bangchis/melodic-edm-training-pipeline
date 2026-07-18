#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

export CUDA_VISIBLE_DEVICES=0,1
project=/workspace/melodic_edm_training_pipeline
ace="$project/vendor/ACE-Step-1.5-v2"
export PYTHONPATH="$ace${PYTHONPATH:+:$PYTHONPATH}"
output="$project/outputs/v2/train-validation"
"$ace/.venv/bin/python" -u "$project/scripts/audit_v2_trainer_runtime.py" \
  --project-root "$project" --vendor-root "$ace"
python3 - <<'PY'
import json
from pathlib import Path

project = Path('/workspace/melodic_edm_training_pipeline')
tensor_report = project / 'data_v2/tensor_validation_report.json'
if not tensor_report.is_file():
    raise SystemExit('v2 tensor validation report is missing')
report = json.loads(tensor_report.read_text())
if (
    report.get('status') != 'pass'
    or report.get('records') != 231
    or report.get('train_tensors') != 196
    or report.get('validation_tensors') != 35
    or report.get('unique_audio_records') != 217
    or report.get('train_unique_tensors') != 184
    or report.get('validation_unique_tensors') != 33
    or report.get('deduplication_performed') is not True
    or report.get('prompt_embeddings_per_record') != 3
):
    raise SystemExit('v2 tensor validation gate has not passed the 231-catalog/217-unique/3-prompt contract')

records = []
for name in ('dataset_train.json', 'dataset_validation.json'):
    dataset = json.loads((project / 'data_v2' / name).read_text())
    records.extend(dataset['samples'])
if len(records) != 231:
    raise SystemExit(f'expected 231 instrumental records, found {len(records)}')
for record in records:
    if record.get('is_instrumental') is not True:
        raise SystemExit(f"non-instrumental record: {record.get('filename')}")
    if '[Instrumental]' not in record.get('lyrics', ''):
        raise SystemExit(f"missing instrumental structure: {record.get('filename')}")
    if record.get('caption_variant_types') != ['canonical', 'composition', 'production']:
        raise SystemExit(f"invalid prompt variants: {record.get('filename')}")
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
  --dataset-dir "$project/data_v2/tensors_train_unique" \
  --validation-dataset-dir "$project/data_v2/tensors_validation_unique" \
  --output-dir "$output" \
  --checkpoint-dir "$project/checkpoints" \
  --model-variant xl_base --base-model xl_base \
  --adapter-type lora --rank 32 --alpha 32 --dropout 0.1 \
  --target-modules q_proj k_proj v_proj o_proj --attention-type both --strict-attention-scope \
  --lr 5e-5 --batch-size 1 --gradient-accumulation 8 --epochs 30 \
  --warmup-steps 25 --weight-decay 0.01 --optimizer-type adamw --scheduler-type cosine \
  --gradient-checkpointing --cfg-ratio 0.15 --shift 1.0 --num-inference-steps 50 \
  --num-devices 2 --strategy ddp --save-every 5 \
  --validate-every 5 --early-stopping-patience 0 \
  --log-every 10 --log-heavy-every 50 --sample-every-n-epochs 0 \
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
