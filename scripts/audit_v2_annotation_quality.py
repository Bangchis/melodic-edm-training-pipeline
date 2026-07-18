#!/usr/bin/env python3
"""Audit V2 caption specificity and exact annotation-to-tensor alignment."""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from v2_common import atomic_json, read_jsonl


CAPTION_TYPES = ("canonical", "composition", "production")
CAPTION_FUSION_REVISION = "openrouter-per-track-prior-audio-fusion-v2.8"
GENERIC_TERMS = (
    "clean",
    "polished",
    "wide",
    "expansive",
    "repetitive",
    "classic edm structure",
    "instrumental melodic edm",
)


def tokens(text: str) -> set[str]:
    """Return lowercase alphanumeric tokens for a transparent similarity check."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(first: set[str], second: set[str]) -> float:
    """Return token-set Jaccard similarity."""
    return len(first & second) / max(1, len(first | second))


def caption_map(annotation: dict[str, Any]) -> dict[str, str]:
    """Return exact required caption variants."""
    return {
        str(item.get("type")): str(item.get("text") or "").strip()
        for item in annotation.get("caption_variants", [])
        if isinstance(item, dict)
    }


def similarity_summary(
    sample_ids: list[str],
    texts: list[str],
    parents: dict[str, str],
) -> dict[str, Any]:
    """Measure exact/near duplication while excluding same-parent aliases from warnings."""
    token_sets = [tokens(text) for text in texts]
    nearest: list[float] = []
    cross_parent_high: list[dict[str, Any]] = []
    for index, current in enumerate(token_sets):
        best = 0.0
        best_index = -1
        for other_index, other in enumerate(token_sets):
            if index == other_index:
                continue
            score = jaccard(current, other)
            if score > best:
                best = score
                best_index = other_index
        nearest.append(best)
        if (
            best_index >= 0
            and best >= 0.85
            and parents[sample_ids[index]] != parents[sample_ids[best_index]]
        ):
            cross_parent_high.append({
                "sample_id": sample_ids[index],
                "nearest_sample_id": sample_ids[best_index],
                "jaccard": round(best, 4),
            })
    counts = Counter(texts)
    return {
        "exact_duplicate_extras": sum(count - 1 for count in counts.values() if count > 1),
        "nearest_jaccard_mean": statistics.mean(nearest) if nearest else 0.0,
        "nearest_jaccard_p90": (
            statistics.quantiles(nearest, n=10)[8] if len(nearest) >= 10 else max(nearest, default=0.0)
        ),
        "nearest_jaccard_max": max(nearest, default=0.0),
        "cross_parent_similarity_ge_0_85": cross_parent_high,
    }


def check_tensors(
    root: Path,
    annotations: dict[str, dict[str, Any]],
) -> tuple[dict[str, int], list[dict[str, Any]], list[str]]:
    """Load tensors sequentially and prove caption/sample/split identity alignment."""
    import torch

    split_counts: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    embedding_duplicates: list[str] = []
    for split, folder in (("train", "tensors_train"), ("validation", "tensors_validation")):
        for path in sorted((root / "data_v2" / folder).glob("*.pt")):
            sample_id = path.stem
            split_counts[split] += 1
            annotation = annotations.get(sample_id)
            if annotation is None:
                errors.append({"sample_id": sample_id, "reason": "annotation_missing"})
                continue
            value = torch.load(path, map_location="cpu", weights_only=False)
            metadata = value.get("metadata", {})
            captions = caption_map(annotation)
            expected = [captions.get(name, "") for name in CAPTION_TYPES]
            checks = {
                "filename": metadata.get("filename") == f"{sample_id}.flac",
                "canonical": metadata.get("caption") == expected[0],
                "caption_variants": metadata.get("caption_variants") == expected,
                "split": annotation.get("split") == split,
                "embedding_count": len(value.get("encoder_hidden_states_variants", [])) == 3,
                "mask_count": len(value.get("encoder_attention_mask_variants", [])) == 3,
            }
            failed = [name for name, passed in checks.items() if not passed]
            if failed:
                errors.append({"sample_id": sample_id, "reason": "tensor_alignment", "failed": failed})
            embeddings = value.get("encoder_hidden_states_variants", [])
            if len(embeddings) == 3 and any(
                torch.equal(embeddings[0], item) for item in embeddings[1:]
            ):
                embedding_duplicates.append(sample_id)
            del value
    return dict(split_counts), errors, embedding_duplicates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--check-tensors", action="store_true")
    parser.add_argument("--output", default="data_v2/annotation_quality_audit.json")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    manifest = read_jsonl(root / "data_v2" / "manifest.jsonl")
    by_id = {str(row["sample_id"]): row for row in manifest}
    annotations: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "data_v2" / "annotations").glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        annotations[str(value["sample_id"])] = value

    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    if set(annotations) != set(by_id):
        errors.append({
            "reason": "manifest_annotation_id_mismatch",
            "missing_annotations": sorted(set(by_id) - set(annotations)),
            "extra_annotations": sorted(set(annotations) - set(by_id)),
        })
    parents = {sample_id: str(row["parent_song_id"]) for sample_id, row in by_id.items()}
    sample_ids = sorted(set(annotations) & set(by_id))
    captions_by_type: dict[str, list[str]] = {name: [] for name in CAPTION_TYPES}
    identity_leaks: list[dict[str, str]] = []
    nonempty_uncertainty: list[str] = []
    caption_fusion_records = 0
    for sample_id in sample_ids:
        annotation = annotations[sample_id]
        mapped = caption_map(annotation)
        if tuple(mapped) != CAPTION_TYPES:
            errors.append({"sample_id": sample_id, "reason": "caption_type_order", "observed": list(mapped)})
            continue
        for name in CAPTION_TYPES:
            captions_by_type[name].append(mapped[name])
        combined = " ".join(mapped.values()).lower()
        for field in ("expected_artist", "expected_title"):
            identity = str(by_id[sample_id].get(field) or "").strip().lower()
            if len(identity) >= 4 and identity in combined:
                identity_leaks.append({"sample_id": sample_id, "field": field, "value": identity})
        uncertainty = (
            annotation.get("master_annotation", {})
            .get("moss_music_supplement", {})
            .get("uncertain_or_conflicting_facts", [])
        )
        if uncertainty:
            nonempty_uncertainty.append(sample_id)
        fusion = annotation.get("master_annotation", {}).get("caption_fusion", {})
        repair = annotation.get("caption_repair", {})
        fusion_valid = (
            fusion.get("revision") == CAPTION_FUSION_REVISION
            and fusion.get("uses_prior_per_track_annotation") is True
            and fusion.get("uses_independent_audio_analysis") is True
            and fusion.get("binding_multi_view_claim_decisions") is True
            and bool(fusion.get("sources_sha256"))
            and fusion.get("sources_sha256") == repair.get("fusion_sources_sha256")
        )
        if fusion_valid:
            caption_fusion_records += 1
        else:
            errors.append({
                "sample_id": sample_id,
                "reason": "per_track_prior_audio_fusion_lineage_invalid",
            })

    similarity = {
        name: similarity_summary(sample_ids, captions_by_type[name], parents)
        for name in CAPTION_TYPES
    }
    cross_parent_near_duplicates = sum(
        len(value["cross_parent_similarity_ge_0_85"])
        for value in similarity.values()
    )
    if cross_parent_near_duplicates:
        warnings.append(f"cross_parent_near_duplicate_captions:{cross_parent_near_duplicates}")
    if not nonempty_uncertainty:
        warnings.append("all_231_moss_supplements_claim_no_uncertainty")
    canonical = captions_by_type["canonical"]
    generic_term_counts = {
        term: sum(term in text.lower() for text in canonical)
        for term in GENERIC_TERMS
    }
    template_starts = sum(
        text.lower().startswith(("this instrumental", "an instrumental"))
        for text in canonical
    )
    if template_starts > len(canonical) * 0.5:
        warnings.append(f"canonical_template_starts_high:{template_starts}/{len(canonical)}")

    split_counts: dict[str, int] = {}
    tensor_errors: list[dict[str, Any]] = []
    embedding_duplicates: list[str] = []
    if args.check_tensors:
        split_counts, tensor_errors, embedding_duplicates = check_tensors(root, annotations)
        errors.extend(tensor_errors)
        if embedding_duplicates:
            errors.append({
                "reason": "exact_duplicate_prompt_embeddings_within_record",
                "sample_ids": embedding_duplicates,
            })

    report = {
        "status": "pass" if not errors else "failed",
        "content_quality_status": "needs_review" if warnings or identity_leaks else "pass",
        "records": len(sample_ids),
        "parent_songs": len(set(parents.values())),
        "duplicate_parent_records_retained": len(sample_ids) - len(set(parents.values())),
        "caption_similarity": similarity,
        "canonical_template_starts": template_starts,
        "generic_term_counts": generic_term_counts,
        "identity_leaks": identity_leaks,
        "moss_nonempty_uncertainty_records": len(nonempty_uncertainty),
        "caption_fusion_revision": CAPTION_FUSION_REVISION,
        "caption_fusion_records": caption_fusion_records,
        "tensor_check_enabled": args.check_tensors,
        "tensor_split_counts": split_counts,
        "tensor_alignment_errors": tensor_errors,
        "exact_duplicate_prompt_embedding_records": embedding_duplicates,
        "limitations": [
            "Schema and tensor alignment do not prove that every audible instrument claim is true.",
            "A stratified human listening audit is still required before training acceptance.",
        ],
        "warnings": warnings,
        "errors": errors,
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    atomic_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
