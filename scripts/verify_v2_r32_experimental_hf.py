#!/usr/bin/env python3
"""Clean-redownload and checksum the immutable experimental HF preview."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from v2_common import atomic_json


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    package = root / "outputs" / "release" / "melodic-edm-core-v2-r32-experimental"
    upload = json.loads((package / "upload_report.json").read_text(encoding="utf-8"))
    if upload.get("status") != "pass" or not upload.get("sha"):
        raise RuntimeError("experimental upload report is not ready")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF token missing")

    from huggingface_hub import snapshot_download

    errors: list[str] = []
    verified = 0
    with tempfile.TemporaryDirectory(prefix="r32-hf-clean-", dir=root / "outputs") as temporary:
        clean = Path(temporary)
        snapshot_download(
            repo_id=upload["repo_id"],
            repo_type="model",
            revision=upload["sha"],
            token=token,
            local_dir=clean,
        )
        for line in (clean / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, relative = line.split("  ", 1)
            path = clean / relative
            if not path.is_file() or sha256(path) != expected:
                errors.append(f"checksum_mismatch:{relative}")
            else:
                verified += 1
        adapter = json.loads((clean / "experimental-r32" / "adapter_config.json").read_text())
        if (adapter.get("r"), adapter.get("lora_alpha"), adapter.get("lora_dropout")) != (32, 32, 0.1):
            errors.append("adapter_config_is_not_r32_alpha32_dropout01")
        notebook_text = (clean / "notebooks" / "melodic_edm_core_v2_colab.ipynb").read_text(encoding="utf-8")
        notebook = json.loads(notebook_text)
        if notebook.get("nbformat") != 4:
            errors.append("notebook_format_invalid")
        for marker in (
            "DURATION_SECONDS = 180",
            "SECTIONS =",
            "CUSTOM_LYRICS =",
            "WARN_SECTION_SECONDS_BELOW =",
            "USE_ACE_LM_THINKING = False",
            "USE_OPENROUTER_ENHANCER = True",
            "REFERENCE_ARTIST =",
            "REFERENCE_TRACK_TITLE =",
            "LORA_SCALE = 0.5",
            "MPLBACKEND",
        ):
            if marker not in notebook_text:
                errors.append(f"notebook_control_missing:{marker}")
    report = {
        "status": "pass" if not errors else "failed",
        "repo_id": upload["repo_id"],
        "verified_revision": upload["sha"],
        "checksums_verified": verified,
        "adapter": "rank32_alpha32_dropout0.1",
        "notebook_controls_verified": True if not errors else False,
        "clean_download_removed": True,
        "errors": errors,
    }
    atomic_json(package / "clean_download_verification_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
