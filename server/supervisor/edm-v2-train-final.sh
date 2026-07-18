#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

export CUDA_VISIBLE_DEVICES=0,1
project=/workspace/melodic_edm_training_pipeline
ace="$project/vendor/ACE-Step-1.5-v2"
export PYTHONPATH="$ace${PYTHONPATH:+:$PYTHONPATH}"
output="$project/outputs/v2/final-all-data"
if [ -e "$output/final" ] || [ -e "$output/checkpoints" ]; then
  echo "Refusing to overwrite or resume the final-all-data run" >&2
  exit 1
fi
final_steps=$(python3 "$project/scripts/compute_final_steps.py" --project-root "$project")
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
cd "$project"
set +e
"$ace/.venv/bin/python" -u -m acestep.training_v2.cli.train_fixed --yes \
  --dataset-dir "$project/data_v2/tensors_all_unique" \
  --output-dir "$output" \
  --checkpoint-dir "$project/checkpoints" \
  --model-variant xl_base --base-model xl_base \
  --adapter-type lora --rank 32 --alpha 32 --dropout 0.1 \
  --target-modules q_proj k_proj v_proj o_proj --attention-type both --strict-attention-scope \
  --lr 5e-5 --batch-size 1 --gradient-accumulation 8 --epochs 50 \
  --max-steps "$final_steps" \
  --warmup-steps 25 --weight-decay 0.01 --optimizer-type adamw --scheduler-type cosine \
  --gradient-checkpointing --cfg-ratio 0.15 --shift 1.0 --num-inference-steps 50 \
  --num-devices 2 --strategy ddp --save-every 5 \
  --validate-every 5 --early-stopping-patience 0 \
  --log-every 10 --log-heavy-every 50 --sample-every-n-epochs 0 \
  2>&1 | tee "$output/training.log"
train_status=${PIPESTATUS[0]}
set -e
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
trap - EXIT TERM INT
if [ "$train_status" -ne 0 ]; then
  exit "$train_status"
fi
exec "$ace/.venv/bin/python" -u "$project/scripts/validate_final_v2.py" \
  --project-root "$project"
