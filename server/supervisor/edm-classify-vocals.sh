#!/bin/bash
set -eo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

set -a
. /workspace/.env
set +a

cd /workspace/melodic_edm_training_pipeline
exec /usr/bin/python3 -u scripts/classify_vocals.py
