#!/bin/bash
set -eo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

export CUDA_VISIBLE_DEVICES=0,1
cd /workspace/melodic_edm_training_pipeline/vendor/ACE-Step-1.5
exec .venv/bin/python -u -m acestep.training_v2.cli.train_fixed --yes \
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
  --log-every 10 --log-heavy-every 50 --sample-every-n-epochs 10
