#!/usr/bin/env python3
"""Run the final machine-readable acceptance audit for the V2 objective."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from v2_common import atomic_json


def load_json(root: Path, relative: str, errors: list[str]) -> dict[str, Any]:
    """Load one required JSON document and record a concise error."""
    path = root / relative
    if not path.is_file():
        errors.append(f"required_report_missing:{relative}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"required_report_invalid:{relative}:{type(exc).__name__}")
        return {}


def require_pass(relative: str, value: dict[str, Any], errors: list[str]) -> None:
    """Require a report status of pass."""
    if value.get("status") != "pass":
        errors.append(f"report_not_passed:{relative}:{value.get('status')}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    errors: list[str] = []
    reports: dict[str, dict[str, Any]] = {}
    required_reports = (
        "data_v2/moss_validation_report.json",
        "data_v2/annotation_merge_report.json",
        "data_v2/dataset_build_report.json",
        "data_v2/tensor_validation_report.json",
        "data_v2/metadata_upload_report.json",
        "data_v2/annotation_quality_audit.json",
        "data_v2/annotation_fidelity_audit.json",
        "data_v2/claim_consensus_report.json",
        "data_v2/caption_repair_report.json",
        "data_v2/downstream_reset_report.json",
        "outputs/v2/baseline-xl-base/generation_report.json",
        "outputs/v2/baseline-xl-base/listening_scores.json",
        "outputs/v2/baseline-xl-base/listening_quality_report.json",
        "outputs/v2/smoke/smoke_validation_report.json",
        "outputs/v2/train-validation/training_validation_report.json",
        "outputs/v2/checkpoint-evaluation/generation_report.json",
        "outputs/v2/checkpoint-evaluation/listening_scores.json",
        "outputs/v2/checkpoint-evaluation/selection.json",
        "outputs/v2/checkpoint-evaluation/hf_evaluation_upload_report.json",
        "outputs/release/melodic-edm-core-v2-preview/preview_report.json",
        "outputs/v2/checkpoint-evaluation/preview_upload_report.json",
        "outputs/v2/checkpoint-evaluation/preview_clean_verification_report.json",
        "outputs/v2/final-all-data/final_validation_report.json",
        "outputs/v2/audio_dataset_prepare_report.json",
        "outputs/v2/audio_dataset_upload_report.json",
        "outputs/v2/audio_dataset_clean_verification_report.json",
        "outputs/v2/final-all-data/evaluation/generation_report.json",
        "outputs/v2/final-all-data/evaluation/listening_scores.json",
        "outputs/v2/final-all-data/evaluation/listening_quality_report.json",
        "outputs/release/melodic-edm-core-v2/release_report.json",
        "outputs/release/melodic-edm-core-v2/upload_report.json",
        "outputs/release/melodic-edm-core-v2/clean_verification_report.json",
    )
    for relative in required_reports:
        value = load_json(root, relative, errors)
        reports[relative] = value
        require_pass(relative, value, errors)
    sync_relative = "outputs/v2/train-validation/hf_sync_state.json"
    sync_state = load_json(root, sync_relative, errors)
    reports[sync_relative] = sync_state
    uploaded_epochs = {int(epoch) for epoch in sync_state.get("uploaded_epochs", {})}
    if not sync_state.get("completed_at") or uploaded_epochs != {5, 10, 15, 20}:
        errors.append("private_checkpoint_sync_incomplete")
    if any(epoch % 5 for epoch in uploaded_epochs):
        errors.append(f"non_fifth_epoch_uploaded:{sorted(uploaded_epochs)}")

    config = load_json(root, "configs/v2/train_val.json", errors)
    adapter = config.get("adapter", {})
    optimization = config.get("optimization", {})
    data = config.get("data", {})
    expected_values = {
        "adapter.rank": (adapter.get("rank"), 32),
        "adapter.alpha": (adapter.get("alpha"), 32),
        "adapter.dropout": (adapter.get("dropout"), 0.1),
        "adapter.targets": (adapter.get("target_modules"), ["q_proj", "k_proj", "v_proj", "o_proj"]),
        "adapter.attention_scope": (
            adapter.get("attention_scope"),
            "separate_self_and_cross_attention_projections",
        ),
        "optimization.learning_rate": (optimization.get("learning_rate"), 0.00005),
        "optimization.cfg_dropout": (optimization.get("cfg_dropout"), 0.15),
        "optimization.warmup": (optimization.get("warmup_optimizer_steps"), 25),
        "optimization.gpus": (optimization.get("gpus"), 2),
        "optimization.effective_batch": (optimization.get("effective_batch"), 16),
        "data.records": (data.get("records"), 231),
        "data.train_records": (data.get("train_records"), 196),
        "data.validation_records": (data.get("validation_records"), 35),
        "data.test_records": (data.get("test_records"), 0),
        "data.caption_variants": (
            data.get("caption_variants"),
            ["canonical", "composition", "production"],
        ),
    }
    for label, (observed, expected) in expected_values.items():
        if observed != expected:
            errors.append(f"config_mismatch:{label}:{observed}:{expected}")

    merge = reports["data_v2/annotation_merge_report.json"]
    moss = reports["data_v2/moss_validation_report.json"]
    if moss.get("prompt_revision") != "audio-blind-v2.2":
        errors.append("moss_annotation_prompt_is_not_audio_blind_v2_2")
    if merge.get("moss_prompt_revision") != "audio-blind-v2.2":
        errors.append("merged_annotations_do_not_use_audio_blind_v2_2")
    if (merge.get("records_merged"), merge.get("train_records"), merge.get("validation_records")) != (231, 196, 35):
        errors.append("merged_dataset_counts_invalid")
    if merge.get("parent_split_crossings"):
        errors.append("parent_song_split_crossing_present")
    claims = reports["data_v2/claim_consensus_report.json"]
    if claims.get("records") != 231 or int(claims.get("claim_total", 0)) <= 0:
        errors.append("multi_view_claim_consensus_incomplete")
    if claims.get("claim_verifier_revision") != "multi-view-audio-claims-v2.4":
        errors.append("claim_verifier_revision_not_v2_4")
    repairs = reports["data_v2/caption_repair_report.json"]
    if repairs.get("caption_compiler_revision") != "audio-grounded-caption-compiler-v2.5":
        errors.append("caption_compiler_revision_not_v2_5")
    reset = reports["data_v2/downstream_reset_report.json"]
    if reset.get("fresh_rank32_outputs_required") is not True:
        errors.append("stale_pre_audio_blind_training_outputs_not_reset")
    tensors = reports["data_v2/tensor_validation_report.json"]
    if (tensors.get("train_tensors"), tensors.get("validation_tensors"), tensors.get("all_tensors")) != (196, 35, 231):
        errors.append("tensor_counts_invalid")
    if tensors.get("prompt_embeddings_per_record") != 3:
        errors.append("prompt_embedding_count_invalid")

    smoke = reports["outputs/v2/smoke/smoke_validation_report.json"]
    baseline_scores = reports["outputs/v2/baseline-xl-base/listening_scores.json"]
    baseline_quality = reports["outputs/v2/baseline-xl-base/listening_quality_report.json"]
    if baseline_scores.get("scorer_revision") != "fixed-prompt-audio-judge-v2.2":
        errors.append("baseline_prompt_scorer_revision_not_v2_2")
    if (
        baseline_quality.get("quality_accepted") is not True
        or baseline_quality.get("quality_profile") != "baseline"
    ):
        errors.append("pristine_xl_base_absolute_quality_not_accepted")
    if smoke.get("optimizer_steps") != 66 or not smoke.get("adapter_reload_verified"):
        errors.append("smoke_resume_or_reload_evidence_invalid")
    training = reports["outputs/v2/train-validation/training_validation_report.json"]
    if training.get("validation_epochs") != [5, 10, 15, 20]:
        errors.append("required_validation_epochs_missing")
    if training.get("checkpoint_epochs") != [5, 10, 15, 20]:
        errors.append("required_checkpoint_epochs_missing")
    selection = reports["outputs/v2/checkpoint-evaluation/selection.json"]
    checkpoint_scores = reports["outputs/v2/checkpoint-evaluation/listening_scores.json"]
    if checkpoint_scores.get("scorer_revision") != "fixed-prompt-audio-judge-v2.2":
        errors.append("checkpoint_prompt_scorer_revision_not_v2_2")
    if (
        int(selection.get("best_optimizer_step", 0)) <= 0
        or selection.get("quality_accepted") is not True
        or float(selection.get("selected_lora_scale", 0.0)) not in (0.25, 0.5, 1.0)
    ):
        errors.append("best_optimizer_step_invalid")
    selection_method = selection.get("selection_method", {})
    if (
        selection_method.get("prompt_alignment_weight") != 0.30
        or selection_method.get("candidate_must_not_underperform_baseline_prompt_alignment") is not True
    ):
        errors.append("prompt_alignment_not_authoritative_in_checkpoint_selection")
    preview_upload = reports["outputs/v2/checkpoint-evaluation/preview_upload_report.json"]
    preview_clean = reports["outputs/v2/checkpoint-evaluation/preview_clean_verification_report.json"]
    if preview_upload.get("private") is not True or not preview_upload.get("sha"):
        errors.append("best_val_preview_not_private_or_unpinned")
    if (
        preview_clean.get("verified_revision") != preview_upload.get("sha")
        or not preview_clean.get("adapter_hash_match")
    ):
        errors.append("best_val_preview_clean_verification_invalid")
    final = reports["outputs/v2/final-all-data/final_validation_report.json"]
    if final.get("records") != 231 or final.get("initialization") != "fresh_xl_base_and_fresh_rank32_lora":
        errors.append("fresh_final_training_evidence_invalid")
    final_quality = reports[
        "outputs/v2/final-all-data/evaluation/listening_quality_report.json"
    ]
    final_scores = reports["outputs/v2/final-all-data/evaluation/listening_scores.json"]
    if final_scores.get("scorer_revision") != "fixed-prompt-audio-judge-v2.2":
        errors.append("final_prompt_scorer_revision_not_v2_2")
    if (
        final_quality.get("quality_accepted") is not True
        or final_quality.get("quality_profile") != "candidate"
    ):
        errors.append("final_absolute_listening_quality_not_accepted")
    audio_prepare = reports["outputs/v2/audio_dataset_prepare_report.json"]
    audio_upload = reports["outputs/v2/audio_dataset_upload_report.json"]
    audio_clean = reports["outputs/v2/audio_dataset_clean_verification_report.json"]
    if (
        audio_prepare.get("records"),
        audio_prepare.get("train_records"),
        audio_prepare.get("validation_records"),
        audio_prepare.get("deduplication_performed"),
    ) != (231, 196, 35, False):
        errors.append("audio_dataset_prepare_counts_or_dedup_invalid")
    if audio_upload.get("private") is not True or not audio_upload.get("sha"):
        errors.append("audio_dataset_not_private_or_unpinned")
    if (
        audio_clean.get("verified_revision") != audio_upload.get("sha")
        or audio_clean.get("records") != 231
        or audio_clean.get("train_records") != 196
        or audio_clean.get("validation_records") != 35
        or audio_clean.get("checksum_mismatches") != 0
    ):
        errors.append("audio_dataset_clean_verification_invalid")

    upload = reports["outputs/release/melodic-edm-core-v2/upload_report.json"]
    if upload.get("private") is not True or not upload.get("sha"):
        errors.append("final_model_upload_not_private_or_unpinned")
    clean = reports["outputs/release/melodic-edm-core-v2/clean_verification_report.json"]
    if clean.get("verified_revision") != upload.get("sha") or not clean.get("adapter_hash_match"):
        errors.append("clean_redownload_revision_or_hash_invalid")
    release = reports["outputs/release/melodic-edm-core-v2/release_report.json"]
    if release.get("secret_scan_findings") != 0 or release.get("generated_examples") != 6:
        errors.append("release_secret_or_audio_example_gate_invalid")

    for relative in (
        "TRAINING_V2.md",
        "COLAB_INFERENCE_V2.md",
        "MODEL_CARD_V2.md",
        "notebooks/melodic_edm_core_v2_colab.ipynb",
    ):
        if not (root / relative).is_file():
            errors.append(f"documentation_missing:{relative}")
    report = {
        "status": "pass" if not errors else "failed",
        "objective": "melodic_edm_core_v2_rank32_quality_gated_fresh231_release",
        "reports_checked": [*required_reports, sync_relative],
        "configuration_checks": len(expected_values),
        "best_optimizer_step": selection.get("best_optimizer_step"),
        "final_optimizer_steps": final.get("observed_optimizer_steps"),
        "hugging_face_model_revision": upload.get("sha"),
        "hugging_face_audio_dataset_revision": audio_upload.get("sha"),
        "clean_inference_audio": clean.get("inference", {}).get("audio_path"),
        "errors": errors,
    }
    atomic_json(root / "outputs" / "v2" / "final_acceptance_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
