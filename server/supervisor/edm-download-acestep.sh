#!/bin/bash
set -eo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

set -a
. /workspace/.env
set +a

cd /workspace/melodic_edm_training_pipeline/vendor/ACE-Step-1.5
exec .venv/bin/python -u -c '
import os
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path("/workspace/melodic_edm_training_pipeline/checkpoints")
root.mkdir(parents=True, exist_ok=True)
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
print("Downloading ACE-Step shared checkpoints", flush=True)
snapshot_download(
    repo_id="ACE-Step/Ace-Step1.5",
    local_dir=str(root),
    token=token,
    max_workers=8,
)
print("Downloading ACE-Step XL-Base", flush=True)
snapshot_download(
    repo_id="ACE-Step/acestep-v15-xl-base",
    local_dir=str(root / "acestep-v15-xl-base"),
    token=token,
    max_workers=8,
)
print("ACE-Step checkpoints ready", flush=True)
'
