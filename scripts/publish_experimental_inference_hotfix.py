#!/usr/bin/env python3
"""Atomically publish and clean-verify experimental inference-only fixes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

from huggingface_hub import (
    CommitOperationAdd,
    HfApi,
    hf_hub_download,
    snapshot_download,
)

from v2_common import atomic_json


CUSTOM_SECTION_FIXTURE = [
    (
        "Filtered Intro: soft piano and the main eight-bar glassy pluck motif, "
        "minimal ambience, no full drums"
    ),
    "Main Theme",
    "First Build",
    "First Melodic Drop",
    "Emotional Breakdown",
    "Second Build",
    "Final Melodic Drop",
    "Outro",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--repo-id", default="Bangchis/melodic-edm-core-v2-r32-experimental"
    )
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF token is not configured")

    api = HfApi(token=token)
    previous_revision = api.model_info(args.repo_id).sha
    current_sums = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            repo_type="model",
            filename="SHA256SUMS",
            revision=previous_revision,
            token=token,
        )
    )
    checksums: dict[str, str] = {}
    for line in current_sums.read_text(encoding="utf-8").splitlines():
        if line.strip():
            expected, relative = line.split("  ", 1)
            checksums[relative] = expected

    sources = {
        "scripts/enhance_prompt_openrouter.py": root
        / "scripts"
        / "enhance_prompt_openrouter.py",
        "scripts/v2_common.py": root / "scripts" / "v2_common.py",
        "notebooks/melodic_edm_core_v2_colab.ipynb": root
        / "notebooks"
        / "melodic_edm_core_v2_colab.ipynb",
        "README.md": root / "MODEL_CARD_V2_R32_EXPERIMENTAL.md",
    }
    missing = [relative for relative, path in sources.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"hotfix inputs missing: {missing}")
    checksums.update({relative: digest(path) for relative, path in sources.items()})

    output_root = root / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="experimental-hotfix-", dir=output_root) as temporary:
        temporary_root = Path(temporary)
        checksum_file = temporary_root / "SHA256SUMS"
        checksum_file.write_text(
            "".join(
                f"{expected}  {relative}\n"
                for relative, expected in sorted(checksums.items())
            ),
            encoding="utf-8",
        )
        operations = [
            CommitOperationAdd(path_in_repo=relative, path_or_fileobj=str(path))
            for relative, path in sources.items()
        ]
        operations.append(
            CommitOperationAdd(
                path_in_repo="SHA256SUMS", path_or_fileobj=str(checksum_file)
            )
        )
        commit = api.create_commit(
            repo_id=args.repo_id,
            repo_type="model",
            operations=operations,
            commit_message="Allow descriptive long-form section labels in Colab",
        )
        revision = str(commit.oid)
        clean = temporary_root / "clean"
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="model",
            revision=revision,
            token=token,
            local_dir=clean,
        )
        errors: list[str] = []
        verified = 0
        for line in (clean / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, relative = line.split("  ", 1)
            path = clean / relative
            if not path.is_file() or digest(path) != expected:
                errors.append(f"checksum_mismatch:{relative}")
            verified += 1

        sys.path.insert(0, str(clean / "scripts"))
        from enhance_prompt_openrouter import (
            attach_inference_style_reference,
            sections_to_lyrics,
        )

        lyrics = sections_to_lyrics(CUSTOM_SECTION_FIXTURE)
        if (
            lyrics.count("[Instrumental]") != len(CUSTOM_SECTION_FIXTURE)
            or f"[{CUSTOM_SECTION_FIXTURE[0]}]" not in lyrics
            or "[Final Melodic Drop]" not in lyrics
        ):
            errors.append("custom_section_runtime_failed")
        style_caption = attach_inference_style_reference(
            "Instrumental melodic house with a clear hook and wide synth chords.",
            "Xomu",
            "Mannenzakura",
        )
        if not style_caption.startswith(
            'Instrumental music in the characteristic style of Xomu, '
            'drawing on the musical character of the reference track "Mannenzakura".'
        ):
            errors.append("artist_track_style_reference_runtime_failed")
        notebook_text = (
            clean / "notebooks" / "melodic_edm_core_v2_colab.ipynb"
        ).read_text(encoding="utf-8")
        json.loads(notebook_text)
        for marker in (
            "STRUCTURE_LYRICS = CUSTOM_LYRICS.strip() or sections_to_lyrics(SECTIONS)",
            "music_conditions['lyrics'] = STRUCTURE_LYRICS",
            "REFERENCE_ARTIST =",
            "REFERENCE_TRACK_TITLE =",
            "attach_inference_style_reference",
        ):
            if marker not in notebook_text:
                errors.append(f"notebook_marker_missing:{marker}")

    report = {
        "status": "pass" if not errors else "failed",
        "repo_id": args.repo_id,
        "previous_revision": previous_revision,
        "verified_revision": revision,
        "checksums_verified": verified,
        "custom_sections_verified": len(CUSTOM_SECTION_FIXTURE),
        "errors": errors,
    }
    report_path = (
        root
        / "outputs"
        / "release"
        / "melodic-edm-core-v2-r32-experimental"
        / "custom_section_hotfix_verification.json"
    )
    atomic_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
