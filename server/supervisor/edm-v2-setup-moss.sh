#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"
set -a
. /workspace/.env
set +a

project=/workspace/melodic_edm_training_pipeline
source_repo="$project/vendor/MOSS-Music"
venv="$project/.venvs/moss-music"
model_dir="$project/checkpoints/MOSS-Music-8B-Thinking"
source_revision=ad107c7ddaa06de168a0dfbc18d3e1e6a40c0e5e
model_revision=2ce899988b94b8ecc5dd0dacbc5ce1874d3500e3

if [ ! -d "$source_repo/.git" ]; then
  git clone https://github.com/OpenMOSS/MOSS-Music.git "$source_repo"
fi
git -C "$source_repo" fetch origin "$source_revision"
git -C "$source_repo" checkout --detach "$source_revision"

if [ ! -x "$venv/bin/python" ]; then
  uv venv --python 3.12 "$venv"
fi
uv pip install \
  --python "$venv/bin/python" \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -e "$source_repo[torch-runtime]"

"$venv/bin/python" -u - <<PY
import os
from pathlib import Path

from huggingface_hub import snapshot_download

target = Path("$model_dir")
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
snapshot_download(
    repo_id="OpenMOSS-Team/MOSS-Music-8B-Thinking",
    revision="$model_revision",
    local_dir=str(target),
    token=token,
)
(target / "PINNED_REVISION").write_text("$model_revision\n", encoding="utf-8")
(target / "SOURCE_COMMIT").write_text("$source_revision\n", encoding="utf-8")
print("MOSS-Music v2 annotator ready", flush=True)
PY
