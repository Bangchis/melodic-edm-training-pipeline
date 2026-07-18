#!/bin/bash
set -eo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"
set -a
. /workspace/.env
set +a

project=/workspace/melodic_edm_training_pipeline
ace="$project/vendor/ACE-Step-1.5"
model="$project/checkpoints/Qwen2.5-Omni-7B"
uv pip install --python "$ace/.venv/bin/python" qwen-omni-utils==0.0.9
cd "$ace"
exec .venv/bin/python -u - <<'PY'
import os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
repo = 'Qwen/Qwen2.5-Omni-7B'
target = Path('/workspace/melodic_edm_training_pipeline/checkpoints/Qwen2.5-Omni-7B')
token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
api = HfApi(token=token)
revision = api.model_info(repo).sha
print(f'Pinned {repo} at {revision}', flush=True)
snapshot_download(repo_id=repo, revision=revision, local_dir=str(target), token=token)
(target / 'PINNED_REVISION').write_text(revision + '\n', encoding='utf-8')
print('Qwen local annotator ready', flush=True)
PY
