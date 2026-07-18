#!/bin/bash
set -eo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

project=/workspace/melodic_edm_training_pipeline
release_root="$(mktemp -d /workspace/melodic-edm-v2-redownload.XXXXXX)"
revision="$(${project}/vendor/ACE-Step-1.5-v2/.venv/bin/python -c 'import json; print(json.load(open("/workspace/melodic_edm_training_pipeline/outputs/release/melodic-edm-core-v2/upload_report.json"))["sha"])')"

cd "${project}/vendor/ACE-Step-1.5-v2"
exec .venv/bin/python -u "${project}/scripts/download_and_infer_v2.py" \
  --repo-id Bangchis/melodic-edm-core-v2 \
  --revision "${revision}" \
  --download-dir "${release_root}/release" \
  --ace-root "${project}/vendor/ACE-Step-1.5-v2" \
  --checkpoint-root "${project}/checkpoints" \
  --adapter-subdirectory final-all-data \
  --prompt-index 0 \
  --output-dir "${release_root}/generated"
