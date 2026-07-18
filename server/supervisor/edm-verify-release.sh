#!/bin/bash
set -eo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

project=/workspace/melodic_edm_training_pipeline
verify="$project/outputs/release_verify"
rm -rf "$verify"
mkdir -p "$verify/download" "$verify/generated"
cd "$project/vendor/ACE-Step-1.5"
.venv/bin/python - <<'PY'
import os
from huggingface_hub import snapshot_download
token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
snapshot_download(
    repo_id='Bangchis/melodic-edm-core-v1', repo_type='model', token=token,
    local_dir='/workspace/melodic_edm_training_pipeline/outputs/release_verify/download',
)
PY

test -s "$verify/download/SHA256SUMS"
(
  cd "$verify/download"
  sha256sum -c SHA256SUMS
)

exec .venv/bin/python -u "$verify/download/infer_release.py" \
  --ace-root "$project/vendor/ACE-Step-1.5" \
  --checkpoint-root "$project/checkpoints" \
  --adapter-dir "$verify/download" \
  --prompt-index 0 \
  --output-dir "$verify/generated"
