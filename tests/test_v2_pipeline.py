"""Focused invariants for the second grouped multi-prompt training release."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from v2_common import (  # noqa: E402
    CAPTION_COMPILER_REVISION,
    CAPTION_TYPES,
    TRACK_STYLE_REFERENCE_REVISION,
    attach_track_style_reference,
    caption_map,
    detach_track_style_reference,
    extract_json_object,
    grouped_split,
    object_sha256,
    parent_song_id,
    track_style_reference,
    validate_caption_set,
    validate_track_style_caption_set,
)
from annotate_moss_music import MODEL_REVISION, build_prompt, validate_supplement  # noqa: E402
from score_v2_checkpoints_moss import parse_score  # noqa: E402
from merge_v2_tensors import hardlink_tensor  # noqa: E402
from build_v2_dedup_tensor_views import select_unique_records  # noqa: E402
from infer_v2_release import merged_generation_settings  # noqa: E402
from audit_v2_annotation_fidelity_moss import (  # noqa: E402
    DEFAULT_MAX_TOKENS,
    accepted_adjudication,
    parse_review,
    stratified_rows,
)
from repair_v2_fidelity_failures_openrouter import (  # noqa: E402
    needs_fidelity_repair,
    request_for as fidelity_repair_request,
    validate_adjudication_caption_policy,
    validate_fidelity_caption_policy,
)
from adjudicate_v2_audio_qwen import blind_prompt  # noqa: E402
from repair_v2_annotations_moss import (  # noqa: E402
    exact_claim_asserted,
    fusion_source_material,
    openrouter_generate,
    parse_repair,
    qualified_claim_mentioned,
    unverified_new_claims,
    validate_claim_constraints,
    validate_training_caption_policy,
)
from verify_v2_audio_claims_moss import (  # noqa: E402
    consensus_for,
    extract_instrument_claims,
    migrate_cached_consensus,
    parse_claim_review,
)
from orchestrate_v2 import (  # noqa: E402
    archive_stale_baseline,
    archive_stale_training_outputs,
    json_pass,
    json_value,
)
from v2_listening_quality import summarize_quality  # noqa: E402
from validate_v2_listening_quality import compare_prompt_alignment  # noqa: E402
from summarize_v2_seed_robustness import summarize as summarize_seed_robustness  # noqa: E402
from compare_v2_seed_robustness import compare as compare_seed_robustness  # noqa: E402
from v2_common import normalize_instrumental_structure  # noqa: E402


def words(prefix: str, count: int) -> str:
    """Return deterministic caption text with exactly *count* tokens."""
    return " ".join([prefix] + [f"word{index}" for index in range(1, count)])


class V2PipelineTest(unittest.TestCase):
    def test_exact_audio_dedup_selects_one_deterministic_representative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio_a = root / "a.flac"
            audio_b = root / "b.flac"
            audio_c = root / "c.flac"
            audio_a.write_bytes(b"same-audio")
            audio_b.write_bytes(b"same-audio")
            audio_c.write_bytes(b"different-audio")
            representatives, groups = select_unique_records(root, [
                {"sample_id": "song__002", "split": "train", "final_audio_path": str(audio_b)},
                {"sample_id": "song__001", "split": "train", "final_audio_path": str(audio_a)},
                {"sample_id": "song__003", "split": "validation", "final_audio_path": str(audio_c)},
            ])
        self.assertEqual(["song__001", "song__003"], [item["sample_id"] for item in representatives])
        duplicate = next(item for item in groups if item["records"] == 2)
        self.assertEqual("song__001", duplicate["representative_sample_id"])
        self.assertEqual(1, duplicate["redundant_records"])

    def test_exact_audio_dedup_rejects_cross_split_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            audio = Path(temporary) / "audio.flac"
            audio.write_bytes(b"same-audio")
            with self.assertRaisesRegex(ValueError, "crosses train/validation"):
                select_unique_records(Path(temporary), [
                    {"sample_id": "song__001", "split": "train", "final_audio_path": str(audio)},
                    {"sample_id": "song__002", "split": "validation", "final_audio_path": str(audio)},
                ])

    def test_instrumental_structure_repairs_only_impossible_section_positions(self) -> None:
        lyrics = """[Break]
[Instrumental]

[Theme]
[Instrumental]

[Intro]
[Instrumental]

[Outro]
[Instrumental]

[Theme]
[Instrumental]

[Break]
[Instrumental]
"""
        normalized, changes = normalize_instrumental_structure(lyrics)
        self.assertEqual(
            ["Intro", "Theme", "Build", "Break", "Theme", "Outro"],
            [
                line[1:-1]
                for line in normalized.splitlines()
                if line.startswith("[") and line != "[Instrumental]"
            ],
        )
        self.assertEqual(4, len(changes))

    def test_instrumental_structure_keeps_valid_sequence_unchanged(self) -> None:
        lyrics = "[Intro]\n[Instrumental]\n\n[Theme]\n[Instrumental]\n\n[Drop]\n[Instrumental]\n\n[Outro]\n[Instrumental]\n"
        normalized, changes = normalize_instrumental_structure(lyrics)
        self.assertEqual(lyrics, normalized)
        self.assertEqual([], changes)

    """Protect grouping, prompt coverage and MOSS response parsing."""

    def test_training_config_names_separate_self_and_cross_attention_scope(self) -> None:
        config = json.loads(
            (SCRIPTS.parent / "configs" / "v2" / "train_val.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            "separate_self_and_cross_attention_projections",
            config["adapter"]["attention_scope"],
        )
        self.assertEqual(32, config["adapter"]["rank"])
        self.assertEqual(32, config["adapter"]["alpha"])
        self.assertEqual(30, config["optimization"]["maximum_epochs"])
        self.assertEqual(0.00005, config["optimization"]["learning_rate"])
        self.assertEqual(
            ["canonical", "composition", "production"],
            config["data"]["caption_variants"],
        )
        self.assertEqual(0.5, config["optimization"]["evaluation_lora_scale"])
        self.assertEqual(217, config["data"]["unique_audio_records"])
        self.assertEqual(184, config["data"]["train_unique_audio_records"])
        self.assertEqual(33, config["data"]["validation_unique_audio_records"])
        quality_gate = config["optimization"]["absolute_listening_quality_gate"]
        self.assertEqual(3, quality_gate["minimum_candidate_prompt_alignment_per_sample"])
        self.assertTrue(quality_gate["candidate_prompt_alignment_not_worse_than_baseline"])
        self.assertEqual(0.34, quality_gate["maximum_final_prompt_alignment_regression_from_best_val"])
        trainer_patch = (
            SCRIPTS.parent / "patches" / "acestep-xl-validation-caption-variants.patch"
        ).read_text(encoding="utf-8")
        self.assertIn("min(int(warmup_steps), max(1, int(total_steps) - 1))", trainer_patch)
        self.assertIn("-    warmup_steps = min(warmup_steps, max(1, total_steps // 10))", trainer_patch)
        self.assertIn("cumulative_prompt_counts = torch.zeros(3", trainer_patch)
        self.assertIn(
            '"caption_variant_types": ["canonical", "composition", "production"]',
            trainer_patch,
        )

    def test_v2_dataset_embeds_all_three_fused_prompt_variants(self) -> None:
        builder = (SCRIPTS / "build_v2_dataset.py").read_text(encoding="utf-8")
        validator = (SCRIPTS / "validate_v2_tensors.py").read_text(encoding="utf-8")
        self.assertIn('"caption_variants": [item["text"] for item in variants]', builder)
        self.assertIn('"caption_variant_types": list(CAPTION_TYPES)', builder)
        self.assertIn('"prompt_embeddings_per_record": 3', validator)

    def test_two_gpu_runtime_audit_blocks_unpatched_training(self) -> None:
        audit = (SCRIPTS / "audit_v2_trainer_runtime.py").read_text(encoding="utf-8")
        for marker in (
            "should_sync_gradient(",
            "rescale_remainder_gradients(",
            "use_distributed_sampler=False",
            '"tail_microbatches": tail_microbatches',
            '"final_tail_microbatches": final_tail_microbatches',
        ):
            self.assertIn(marker, audit)
        for name in ("smoke", "main", "final"):
            launcher = (
                SCRIPTS.parent / "server" / "supervisor" / f"edm-v2-train-{name}.sh"
            ).read_text(encoding="utf-8")
            self.assertIn("audit_v2_trainer_runtime.py", launcher)
        objective = (SCRIPTS / "audit_v2_objective.py").read_text(encoding="utf-8")
        self.assertIn("ddp_remainder_runtime_proof_invalid", objective)
        self.assertIn("final_all_data_ddp_remainder_runtime_proof_invalid", objective)

    def test_smoke_resume_validator_matches_the_saved_step_60_boundary(self) -> None:
        validator = (SCRIPTS / "validate_smoke_v2.py").read_text(encoding="utf-8")
        launcher = (
            SCRIPTS.parent / "server" / "supervisor" / "edm-v2-train-smoke.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('r"Resumed(?: LoRA)? from epoch 5, step 60', validator)
        self.assertIn("--max-steps 66", launcher)

    def test_robust_eval_prompts_match_training_form_density(self) -> None:
        config = json.loads(
            (SCRIPTS.parent / "configs" / "v2" / "robust_eval_prompts.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(3, len(config["prompts"]))
        for prompt in config["prompts"]:
            words_in_caption = len(prompt["caption"].split())
            section_count = prompt["lyrics"].count("[Instrumental]")
            self.assertEqual("held_out_artist_track_style_reference", prompt["conditioning_mode"])
            self.assertEqual("validation", prompt["reference_split"])
            self.assertTrue(
                prompt["caption"].startswith(
                    track_style_reference(prompt["reference_artist"], prompt["reference_track"])
                )
            )
            self.assertGreaterEqual(words_in_caption, 40)
            self.assertLessEqual(words_in_caption, 80)
            self.assertGreaterEqual(prompt["duration"] / section_count, 20)
            self.assertNotIn(" - ", prompt["lyrics"])

    def test_fixed_eval_prompts_use_long_form_held_out_style_references(self) -> None:
        config = json.loads(
            (SCRIPTS.parent / "configs" / "v2" / "fixed_eval_prompts.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(3, len(config["prompts"]))
        for prompt in config["prompts"]:
            section_count = prompt["lyrics"].count("[Instrumental]")
            self.assertTrue(
                prompt["caption"].startswith(
                    track_style_reference(prompt["reference_artist"], prompt["reference_track"])
                )
            )
            self.assertGreaterEqual(prompt["duration"] / section_count, 20)
            self.assertIn("fully instrumental", prompt["caption"])

    def test_changed_fixed_prompt_document_archives_stale_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs" / "v2").mkdir(parents=True)
            baseline = root / "outputs" / "v2" / "baseline-xl-base"
            baseline.mkdir(parents=True)
            (root / "configs" / "v2" / "fixed_eval_prompts.json").write_text(
                json.dumps({"prompts": [{"id": "new"}]}), encoding="utf-8"
            )
            (baseline / "generation_report.json").write_text(
                json.dumps({"status": "pass", "prompt_document_sha256": "old"}),
                encoding="utf-8",
            )
            archived = archive_stale_baseline(root)
            self.assertIsNotNone(archived)
            self.assertTrue((archived / "generation_report.json").is_file())
            self.assertFalse(baseline.exists())

    def test_robust_evaluation_runs_after_quality_gated_checkpoint_selection(self) -> None:
        orchestrator = (SCRIPTS / "orchestrate_v2.py").read_text(encoding="utf-8")
        self.assertLess(
            orchestrator.index('"edm-v2-select-checkpoint"'),
            orchestrator.index('run_parallel(["edm-v2-evaluate-robust"'),
        )
        launcher = (
            SCRIPTS.parent / "server" / "supervisor" / "edm-v2-evaluate-robust.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--selected-only", launcher)
        self.assertNotIn("--final-only", launcher)
        evaluator = (SCRIPTS / "evaluate_v2_checkpoints.py").read_text(encoding="utf-8")
        self.assertIn('selection.get("quality_accepted") is not True', evaluator)
        self.assertIn('"checkpoint_selection_sha256": selection_sha256', evaluator)

    def test_pristine_baseline_uses_the_same_instrumental_structure_fallback(self) -> None:
        source = (SCRIPTS / "evaluate_v2_baseline.py").read_text(encoding="utf-8")
        self.assertIn("DEFAULT_LYRICS", source)
        self.assertNotIn("import LYRICS", source)
        self.assertIn('prompt.get("lyrics", DEFAULT_LYRICS)', source)

    def test_seed_robustness_requires_four_of_five_clean_outputs_per_prompt(self) -> None:
        clean = {
            "scores": {
                "prompt_alignment": 3,
                "melody": 3,
                "structure": 3,
                "audio_quality": 3,
            },
            "failure_modes": {
                "distorted": False,
                "collapsed": False,
                "static_loop": False,
                "intelligible_vocals": False,
            },
        }
        failed = {
            **clean,
            "scores": {**clean["scores"], "melody": 2},
        }
        records = [
            {**(clean if index < 4 else failed), "source_prompt_id": "prompt-a", "seed": index}
            for index in range(5)
        ]
        result = summarize_seed_robustness(
            records,
            expected_seeds=5,
            minimum_score=3,
            minimum_pass_rate=0.8,
        )
        self.assertEqual("pass", result["status"])
        self.assertEqual(0.8, result["prompts"]["prompt-a"]["pass_rate"])

    def test_seed_robustness_rejects_lucky_single_seed(self) -> None:
        records = []
        for index in range(5):
            records.append({
                "source_prompt_id": "prompt-a",
                "seed": index,
                "scores": {
                    "prompt_alignment": 4 if index == 0 else 1,
                    "melody": 4 if index == 0 else 1,
                    "structure": 4 if index == 0 else 1,
                    "audio_quality": 4 if index == 0 else 1,
                },
                "failure_modes": {
                    "distorted": False,
                    "collapsed": index > 0,
                    "static_loop": False,
                    "intelligible_vocals": False,
                },
            })
        result = summarize_seed_robustness(
            records,
            expected_seeds=5,
            minimum_score=3,
            minimum_pass_rate=0.8,
        )
        self.assertEqual("failed", result["status"])
        self.assertEqual(0.2, result["prompts"]["prompt-a"]["pass_rate"])

    def test_seed_comparison_pairs_identical_prompt_and_seed(self) -> None:
        def record(seed: int, score: int) -> dict:
            return {
                "source_prompt_id": "prompt-a",
                "seed": seed,
                "scores": {field: score for field in ("prompt_alignment", "melody", "structure", "audio_quality")},
                "failure_modes": {
                    "distorted": False,
                    "collapsed": False,
                    "static_loop": False,
                    "intelligible_vocals": False,
                },
            }

        result = compare_seed_robustness([record(1, 2)], [record(1, 4)])
        self.assertEqual("pass", result["status"])
        self.assertEqual(1, result["outcomes"]["base_only_pass"])
        self.assertEqual(-2, result["mean_score_delta_lora_minus_base"]["melody"])

    def test_main_orchestrator_requires_multi_genre_five_seed_quality(self) -> None:
        source = (SCRIPTS / "orchestrate_v2.py").read_text(encoding="utf-8")
        self.assertIn('"edm-v2-evaluate-robust"', source)
        self.assertIn('"edm-v2-evaluate-robust-base"', source)
        self.assertIn('"edm-v2-score-moss-robust"', source)
        self.assertIn('"edm-v2-compare-robust"', source)
        self.assertIn('"five-seed-lora-quality"', source)
        for launcher in ("edm-v2-evaluate-robust.sh", "edm-v2-evaluate-robust-base.sh"):
            content = (SCRIPTS.parent / "server" / "supervisor" / launcher).read_text(encoding="utf-8")
            self.assertIn("configs/v2/robust_eval_prompts.json", content)
            self.assertIn("--seed-offsets 0,1,2,3,4", content)

    def test_orchestrator_json_gate_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gate.json"
            path.write_text('{"status":"pass","revision":"r2"}', encoding="utf-8")
            self.assertTrue(json_pass(path))
            self.assertEqual("r2", json_value(path, "revision"))

    def test_annotation_lineage_change_archives_all_downstream_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "outputs" / "v2" / "smoke").mkdir(parents=True)
            (root / "outputs" / "v2" / "smoke" / "old.json").write_text("{}")
            (root / "outputs" / "release" / "melodic-edm-core-v2-preview").mkdir(parents=True)
            report = archive_stale_training_outputs(root)
            self.assertTrue(report["fresh_rank32_outputs_required"])
            self.assertTrue((root / "outputs" / "v2").is_dir())
            self.assertFalse((root / "outputs" / "v2" / "smoke" / "old.json").exists())
            self.assertEqual(2, len(report["archived"]))

    def test_annotation_fidelity_sample_is_deterministic_and_stratified(self) -> None:
        rows = [
            {"sample_id": "alpha__001"},
            {"sample_id": "alpha__002"},
            {"sample_id": "alpha__003"},
            {"sample_id": "beta__001"},
            {"sample_id": "beta__002"},
            {"sample_id": "beta__003"},
        ]
        selected = stratified_rows(rows, 2)
        self.assertEqual(
            ["alpha__001", "alpha__003", "beta__001", "beta__003"],
            [row["sample_id"] for row in selected],
        )

    def test_moss_annotation_prompt_is_identity_and_prior_claim_blind(self) -> None:
        prompt = build_prompt(
            {"expected_artist": "Secret Artist", "expected_title": "Secret Title"},
            {"main_instruments": [{"name": "invented_pipa"}]},
            {"bpm": 128, "keyscale": "F# minor"},
            {"type": "object"},
        )
        self.assertNotIn("Secret Artist", prompt)
        self.assertNotIn("Secret Title", prompt)
        self.assertNotIn("invented_pipa", prompt)
        self.assertNotIn("128", prompt)
        self.assertIn("independently from the waveform", prompt)

    def test_moss_repair_forces_json_before_reasoning(self) -> None:
        source = (SCRIPTS / "annotate_moss_music.py").read_text(encoding="utf-8")
        self.assertIn("first output character must be {", source)
        self.assertIn("emit no reasoning or markdown outside the object", source)

    def test_annotation_fidelity_review_requires_grounded_fields(self) -> None:
        review, errors = parse_review({
            "audible_fidelity": 4,
            "specificity": 3,
            "melody_arrangement_accuracy": 4,
            "production_accuracy": 4,
            "evidence": {field: "Audible evidence." for field in (
                "audible_fidelity", "specificity", "melody_arrangement_accuracy",
                "production_accuracy",
            )},
            "unsupported_claims": [],
            "recommendation": "keep",
        })
        self.assertEqual([], errors)
        self.assertEqual(4, review["scores"]["audible_fidelity"])

    def test_annotation_fidelity_allows_thinking_model_to_finish_json(self) -> None:
        self.assertGreaterEqual(DEFAULT_MAX_TOKENS, 3600)
        source = (SCRIPTS / "audit_v2_annotation_fidelity_moss.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("first output character must be {", source)
        self.assertIn("emit no reasoning, markdown, or thinking block", source)

    def test_fidelity_repair_targets_only_absolute_gate_failures(self) -> None:
        passing = {
            "scores": {
                "audible_fidelity": 3,
                "specificity": 2,
                "melody_arrangement_accuracy": 2,
                "production_accuracy": 2,
            },
            "recommendation": "revise",
        }
        failing = {**passing, "scores": {**passing["scores"], "production_accuracy": 1}}
        self.assertFalse(needs_fidelity_repair(passing))
        self.assertTrue(needs_fidelity_repair(failing))

    def test_fidelity_repair_prompt_uses_listening_evidence_without_identity(self) -> None:
        prompt = fidelity_repair_request(
            {name: words(name, 45) for name in CAPTION_TYPES},
            {
                "scores": {field: 1 for field in (
                    "audible_fidelity", "specificity",
                    "melody_arrangement_accuracy", "production_accuracy",
                )},
                "evidence": {"audible_fidelity": "Synth lead, not pipa."},
                "unsupported_claims": ["pipa"],
                "recommendation": "revise",
            },
        )
        self.assertIn("authoritative", prompt)
        self.assertIn("Synth lead, not pipa.", prompt)
        self.assertIn('"pipa"', prompt)
        self.assertNotIn("Secret Artist", prompt)

    def test_fidelity_repair_prompt_can_include_narrow_independent_adjudication(self) -> None:
        prompt = fidelity_repair_request(
            {name: words(name, 45) for name in CAPTION_TYPES},
            {"scores": {}, "evidence": {}, "unsupported_claims": [], "recommendation": "reject"},
            {"decision": "hybrid", "caption_guidance": "Use cautious hybrid wording."},
        )
        self.assertIn("Independent multi-source adjudication", prompt)
        self.assertIn("Use cautious hybrid wording.", prompt)

    def test_audio_adjudication_override_requires_two_blind_qwen_views_and_exact_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs" / "v2").mkdir(parents=True)
            (root / "data").mkdir()
            captions = {
                name: "Electronic plucked synth " + words(name, 42)
                for name in CAPTION_TYPES
            }
            config = {
                "revision": "r1",
                "samples": {
                    "track__001": {
                        "decision": "hybrid",
                        "allow_gate_override": True,
                        "required_caption_terms": ["electronic", "plucked", "synth"],
                        "forbidden_caption_terms": ["pure orchestra"],
                        "qwen_record": "data/qwen.json",
                        "primary_sources": ["https://example.com/source"],
                        "rationale": "Two independent blind views agree.",
                    }
                },
            }
            (root / "configs" / "v2" / "audio_adjudication_overrides.json").write_text(
                json.dumps(config), encoding="utf-8"
            )
            qwen = {
                "status": "pass",
                "identity_blind": True,
                "revision": "q1",
                "views": {
                    "full": {"annotation": {}},
                    "overview_montage": {"annotation": {}},
                },
            }
            (root / "data" / "qwen.json").write_text(json.dumps(qwen), encoding="utf-8")
            result = {
                "sample_id": "track__001",
                "captions_sha256": object_sha256(captions),
                "scores": {"audible_fidelity": 1},
                "recommendation": "reject",
            }
            self.assertIsNotNone(
                accepted_adjudication(root, "track__001", captions, result)
            )
            result["captions_sha256"] = "wrong"
            self.assertIsNone(
                accepted_adjudication(root, "track__001", captions, result)
            )

    def test_caption_repair_override_does_not_automatically_bypass_gate(self) -> None:
        result = {
            "scores": {
                "audible_fidelity": 2,
                "specificity": 2,
                "melody_arrangement_accuracy": 2,
                "production_accuracy": 2,
            },
            "recommendation": "revise",
        }
        adjudication = {
            "force_caption_repair": True,
            "allow_gate_override": False,
            "required_caption_terms": ["synth"],
            "forbidden_caption_terms": ["orchestral"],
        }
        self.assertTrue(needs_fidelity_repair(result, adjudication))
        self.assertEqual(
            validate_adjudication_caption_policy(
                {name: "bright synth electronic arrangement" for name in CAPTION_TYPES},
                adjudication,
            ),
            [],
        )
        self.assertIn(
            "adjudication_forbidden_term_present:orchestral",
            validate_adjudication_caption_policy(
                {name: "bright synth orchestral arrangement" for name in CAPTION_TYPES},
                adjudication,
            ),
        )

    def test_fidelity_repair_rejects_static_recurrence_wording(self) -> None:
        captions = {name: words(name, 45) for name in CAPTION_TYPES}
        captions["composition"] = words("A motif repeats throughout the track", 45)
        errors = validate_fidelity_caption_policy(captions)
        self.assertIn("composition_contains_unqualified_full_track_repetition", errors)

    def test_qwen_adjudication_prompt_is_identity_and_prior_caption_blind(self) -> None:
        prompt = blind_prompt({}, {"schema": {"type": "object"}}, "full")
        self.assertIn("Artist, title, catalog family", prompt)
        self.assertIn("dominant production", prompt)
        self.assertNotIn("Pride & Fear", prompt)
        self.assertNotIn("TheFatRat", prompt)

    def test_caption_repair_requires_three_valid_corrected_captions(self) -> None:
        value = {
            "audible_fidelity": 2,
            "specificity": 2,
            "melody_arrangement_accuracy": 3,
            "production_accuracy": 3,
            "evidence": {field: "Audio-grounded evidence." for field in (
                "audible_fidelity", "specificity", "melody_arrangement_accuracy",
                "production_accuracy",
            )},
            "unsupported_claims": ["invented pipa"],
            "recommendation": "revise",
            "corrected_captions": {name: words(name, 45) for name in CAPTION_TYPES},
        }
        repair, errors = parse_repair(value)
        self.assertEqual([], errors)
        self.assertEqual("revise", repair["recommendation"])
        self.assertEqual(["invented pipa"], repair["unsupported_claims"])

    def test_claim_extraction_is_not_a_blacklist(self) -> None:
        annotation = {
            "caption_variants": [{"text": "A pipa lead with orchestral strings."}],
            "master_annotation": {
                "base_annotation": {"main_instruments": [{"name": "pipa"}]},
                "moss_music_supplement": {
                    "instruments_and_roles": [{"name": "orchestral_strings"}]
                },
            },
        }
        self.assertEqual(["pipa", "orchestral strings"], extract_instrument_claims(annotation))

    def test_like_free_text_does_not_create_exact_instrument_claim(self) -> None:
        annotation = {
            "caption_variants": [{"text": "A pipa-like lead and choir-like pad."}],
            "master_annotation": {},
        }
        self.assertEqual([], extract_instrument_claims(annotation))

    def test_long_structured_timbre_description_keeps_specific_synth_claim(self) -> None:
        annotation = {
            "caption_variants": [],
            "master_annotation": {
                "base_annotation": {
                    "main_instruments": [{"name": "bright, high pitched synth lead"}]
                }
            },
        }
        self.assertEqual(["synth lead"], extract_instrument_claims(annotation))

    def test_composite_instrument_names_are_split_without_punctuation_or_like_hallucination(self) -> None:
        annotation = {
            "caption_variants": [],
            "master_annotation": {
                "moss_music_supplement": {
                    "instruments_and_roles": [
                        {"name": "Traditional Chinese instruments (guzheng/dizi)"},
                        {"name": "Synthesizers/Orchestral strings"},
                        {"name": "Synthesizer (oud-like)"},
                    ]
                }
            },
        }
        claims = extract_instrument_claims(annotation)
        self.assertIn("guzheng", claims)
        self.assertIn("dizi", claims)
        self.assertIn("synthesizer", claims)
        self.assertIn("orchestral strings", claims)
        self.assertNotIn("oud", claims)
        self.assertFalse(any(")" in claim or "(" in claim for claim in claims))
        composite_only = {
            "caption_variants": [],
            "master_annotation": {
                "moss_music_supplement": {
                    "instruments_and_roles": [{"name": "Synthesizers/Brass"}]
                }
            },
        }
        self.assertEqual(["synthesizer", "brass"], extract_instrument_claims(composite_only))
        normalized_composites = {
            "caption_variants": [],
            "master_annotation": {
                "moss_music_supplement": {
                    "instruments_and_roles": [
                        {"name": "String section (violins/cellos)"},
                        {"name": "Bass (synthesizer)"},
                        {"name": "Vocal samples (chops/vocal chops)"},
                        {"name": "Synth pads"},
                    ]
                }
            },
        }
        normalized = extract_instrument_claims(normalized_composites)
        self.assertEqual(
            ["orchestral strings", "violin", "cello", "synth bass", "vocal chops", "synth pad"],
            normalized,
        )
        self.assertFalse(any("(" in claim or ")" in claim for claim in normalized))

    def test_claim_review_requires_exact_coverage(self) -> None:
        review, errors = parse_claim_review({"claims": [{
            "claim": "pipa", "verdict": "present", "confidence": 0.9,
            "evidence": "Audible plucked attacks.", "audible_alternative": "",
        }]}, ["pipa"])
        self.assertEqual([], errors)
        self.assertEqual("present", review["claims"][0]["verdict"])

    def test_claim_review_rejects_zero_confidence_present_with_negative_evidence(self) -> None:
        _review, errors = parse_claim_review({"claims": [{
            "claim": "pipa", "verdict": "present", "confidence": 0,
            "evidence": "No pipa is audible in the track.", "audible_alternative": "",
        }]}, ["pipa"])
        self.assertIn("present_verdict_contradicts_evidence:pipa", errors)

    def test_claim_review_conservatively_normalizes_low_confidence_binary_verdict(self) -> None:
        review, errors = parse_claim_review({"claims": [{
            "claim": "pipa", "verdict": "present", "confidence": 0.3,
            "evidence": "A weak plucked attack may be audible.",
            "audible_alternative": "plucked lead",
        }]}, ["pipa"])
        self.assertEqual([], errors)
        item = review["claims"][0]
        self.assertEqual("uncertain", item["verdict"])
        self.assertEqual("present", item["original_verdict"])
        self.assertEqual("low_confidence_binary_to_uncertain", item["normalization"])

    def test_claim_verifier_forces_json_before_reasoning(self) -> None:
        source = (SCRIPTS / "verify_v2_audio_claims_moss.py").read_text(encoding="utf-8")
        self.assertIn("first output character must be {", source)
        self.assertIn("emit no reasoning or markdown outside the object", source)

    def test_claim_review_accepts_claim_keyed_json(self) -> None:
        review, errors = parse_claim_review({"claims": {"pipa": {
            "verdict": "present", "confidence": 0.9,
            "evidence": "Audible plucked attacks.", "audible_alternative": "",
        }}}, ["pipa"])
        self.assertEqual([], errors)
        self.assertEqual("pipa", review["claims"][0]["claim"])

    def test_consensus_preserves_specific_name_only_with_agreement(self) -> None:
        def review(neutral: str, challenge: str, montage: str):
            return {
                "full_neutral": {"claims": [{"claim": "pipa", "verdict": neutral, "confidence": 0.9, "evidence": "n", "audible_alternative": "plucked lead"}]},
                "full_challenge": {"claims": [{"claim": "pipa", "verdict": challenge, "confidence": 0.9, "evidence": "c", "audible_alternative": "plucked lead"}]},
                "overview_montage": {"claims": [{"claim": "pipa", "verdict": montage, "confidence": 0.9, "evidence": "m", "audible_alternative": "plucked lead"}]},
            }
        self.assertEqual("present", consensus_for("pipa", review("present", "present", "present"))["decision"])
        self.assertEqual("absent", consensus_for("pipa", review("absent", "absent", "uncertain"))["decision"])
        self.assertEqual("uncertain", consensus_for("pipa", review("present", "absent", "present"))["decision"])

    def test_clean_v25_claim_evidence_can_migrate_without_new_audio_inference(self) -> None:
        views = {
            name: {"claims": [{
                "claim": "pipa", "verdict": "present", "confidence": 0.9,
                "evidence": "Distinct plucked attacks are audible.", "audible_alternative": "",
            }]}
            for name in ("full_neutral", "full_challenge", "overview_montage")
        }
        for previous_revision in (
            "multi-view-audio-claims-v2.5",
            "multi-view-audio-claims-v2.6",
        ):
            migrated = migrate_cached_consensus({
                "claim_verifier_revision": previous_revision,
                "model_revision": MODEL_REVISION,
                "audio_sha256": "audio-hash",
                "claims": ["pipa"],
                "views": views,
            }, ["pipa"], "audio-hash", "new-input-hash")
            self.assertIsNotNone(migrated)
            self.assertEqual("multi-view-audio-claims-v2.7", migrated["claim_verifier_revision"])
            self.assertEqual("present", migrated["decisions"][0]["decision"])

    def test_contradictory_v25_claim_evidence_must_be_reheard(self) -> None:
        views = {
            name: {"claims": [{
                "claim": "pipa", "verdict": "present", "confidence": 0,
                "evidence": "No pipa is audible.", "audible_alternative": "",
            }]}
            for name in ("full_neutral", "full_challenge", "overview_montage")
        }
        migrated = migrate_cached_consensus({
            "claim_verifier_revision": "multi-view-audio-claims-v2.5",
            "model_revision": MODEL_REVISION,
            "audio_sha256": "audio-hash",
            "claims": ["pipa"],
            "views": views,
        }, ["pipa"], "audio-hash", "new-input-hash")
        self.assertIsNone(migrated)

    def test_claim_retry_explicitly_converts_low_confidence_binary_verdicts(self) -> None:
        source = (SCRIPTS / "verify_v2_audio_claims_moss.py").read_text(encoding="utf-8")
        self.assertIn("change that verdict to uncertain", source)
        for shard in (0, 1):
            launcher = (
                SCRIPTS.parent / "server" / "supervisor" / f"edm-v2-verify-claims-{shard}.sh"
            ).read_text(encoding="utf-8")
            self.assertIn("--attempts 5", launcher)

    def test_like_qualifier_is_not_an_exact_instrument_assertion(self) -> None:
        self.assertTrue(exact_claim_asserted("A pipa carries the hook.", "pipa"))
        self.assertFalse(exact_claim_asserted("A pipa-like plucked lead carries the hook.", "pipa"))
        self.assertTrue(qualified_claim_mentioned("A pipa-like plucked lead carries the hook.", "pipa"))
        self.assertFalse(qualified_claim_mentioned("A generic plucked-string lead carries the hook.", "pipa"))

    def test_uncertain_claim_may_be_omitted_instead_of_forced_into_caption(self) -> None:
        source = (SCRIPTS / "repair_v2_annotations_moss.py").read_text(encoding="utf-8")
        self.assertNotIn("uncertain_claim_qualified_token_missing", source)
        self.assertIn("claim_name_mentioned(scoped_text, claim)", source)

    def test_present_claim_is_evidence_not_a_required_caption_keyword(self) -> None:
        decisions = [
            {"claim": "pipa", "decision": "present", "audible_alternative": ""},
            {"claim": "electronic bass", "decision": "present", "audible_alternative": ""},
        ]
        self.assertEqual(
            [],
            validate_claim_constraints(
                "A bright syncopated motif develops into a spacious melodic drop.", decisions
            ),
        )
        source = (SCRIPTS / "repair_v2_annotations_moss.py").read_text(encoding="utf-8")
        self.assertNotIn("verified_present_claim_missing", source)
        self.assertIn("not a coverage checklist", source)

    def test_absent_and_uncertain_claims_remain_per_track_safety_constraints(self) -> None:
        self.assertIn(
            "verified_absent_claim_retained:pipa",
            validate_claim_constraints(
                "A pipa-like lead carries the hook.",
                [{"claim": "pipa", "decision": "absent", "audible_alternative": "plucked lead"}],
            ),
        )
        self.assertEqual(
            [],
            validate_claim_constraints(
                "A pipa-like plucked lead carries the hook.",
                [{"claim": "pipa", "decision": "uncertain", "audible_alternative": "plucked lead"}],
            ),
        )

    def test_final_training_caption_policy_rejects_metadata_and_hype(self) -> None:
        errors = validate_training_caption_policy({
            "canonical": "A polished 4/4 EDM track in F# minor with a repetitive lead suitable for gaming.",
            "composition": "A standard EDM structure supports the motif.",
            "production": "The lead moves over a professional 128 BPM mix.",
        })
        self.assertIn("canonical_contains_embedded_time_signature", errors)
        self.assertIn("canonical_contains_embedded_exact_key", errors)
        self.assertIn("canonical_contains_quality_hype", errors)
        self.assertIn("canonical_contains_static_loop_cue", errors)
        self.assertIn("canonical_contains_intended_use_case", errors)
        self.assertIn("composition_contains_generic_edm_structure", errors)
        self.assertIn("production_contains_embedded_bpm", errors)

    def test_caption_instruct_keeps_specific_prior_when_moss_only_omits_it(self) -> None:
        source = (SCRIPTS / "repair_v2_annotations_moss.py").read_text(encoding="utf-8")
        self.assertIn("omitted detail", source)
        self.assertIn("is not a contradiction", source)
        self.assertIn("never replace a compatible specific detail", source)
        self.assertIn("binding claim decisions override exact sound-source", source)

    def test_caption_workers_can_resume_ready_consensus_without_waiting_for_all_rows(self) -> None:
        source = (SCRIPTS / "repair_v2_annotations_moss.py").read_text(encoding="utf-8")
        self.assertIn('"--ready-only"', source)
        self.assertIn("skipped_missing_consensus", source)
        for shard in (0, 1):
            launcher = (
                SCRIPTS.parent / "server" / "supervisor" / f"edm-v2-repair-captions-{shard}.sh"
            ).read_text(encoding="utf-8")
            self.assertIn("--ready-only", launcher)
            self.assertIn("--attempts 5", launcher)

    def test_ddp_remainder_and_exact_validation_patch_is_required(self) -> None:
        patch = (
            SCRIPTS.parent / "patches" / "acestep-ddp-remainder-validation.patch"
        ).read_text(encoding="utf-8")
        self.assertIn("should_sync_gradient", patch)
        self.assertIn("rescale_remainder_gradients", patch)
        self.assertIn("use_distributed_sampler=False", patch)
        self.assertIn("test_final_partial_batch_forces_ddp_sync", patch)

    def test_caption_compiler_cannot_introduce_unverified_exact_instrument(self) -> None:
        decisions = [{
            "claim": "pipa", "decision": "uncertain", "audible_alternative": "plucked lead",
        }]
        self.assertEqual(
            ["piano"],
            unverified_new_claims("A pipa-like plucked hook is doubled by piano.", decisions),
        )
        self.assertEqual(
            [],
            unverified_new_claims("A pipa-like plucked hook has no newly named source.", decisions),
        )

    def test_generic_electronic_timbre_words_do_not_require_exact_source_consensus(self) -> None:
        decisions = [{
            "claim": "pipa", "decision": "uncertain", "audible_alternative": "plucked lead",
        }]
        self.assertEqual(
            [],
            unverified_new_claims(
                "A synthesizer lead moves above electronic drums, bass and percussion.", decisions
            ),
        )
        self.assertIn(
            "verified_absent_claim_retained:synthesizer",
            validate_claim_constraints(
                "A synthesizer carries the hook.",
                [{"claim": "synthesizer", "decision": "absent", "audible_alternative": ""}],
            ),
        )

    def test_broader_string_wording_requires_supported_specific_track_claim(self) -> None:
        self.assertEqual(
            [],
            unverified_new_claims(
                "Plucked strings carry the accompaniment.",
                [{
                    "claim": "plucked string instrument",
                    "decision": "present",
                    "audible_alternative": "",
                }],
            ),
        )
        self.assertEqual(
            ["strings"],
            unverified_new_claims("Strings carry the accompaniment.", []),
        )

    def test_caption_fusion_uses_old_song_prompt_and_independent_audio_facts(self) -> None:
        annotation = {
            "master_annotation": {
                "base_annotation": {
                    "canonical_caption": "Old prompt for this exact song with a rising motif.",
                    "caption_variants": [
                        {"type": "full", "text": "Old full prompt."},
                        {"type": "composition", "text": "Old composition prompt."},
                    ],
                    "moods": ["adventurous"],
                    "melody": {"description": "rising two-bar motif"},
                },
                "moss_music_supplement": {
                    "melody_and_motifs": "A short rising phrase repeats with varied endings.",
                    "production": "Wide synth chords and controlled sub bass.",
                },
            }
        }
        moss_captions = {name: words(name, 45) for name in CAPTION_TYPES}
        sources = fusion_source_material(annotation, moss_captions)
        self.assertEqual(
            "Old prompt for this exact song with a rising motif.",
            sources["prior_per_track_annotation"]["canonical_caption"],
        )
        self.assertEqual(
            "A short rising phrase repeats with varied endings.",
            sources["independent_audio_analysis"]["audible_facts"]["melody_and_motifs"],
        )
        self.assertEqual(
            moss_captions,
            sources["independent_audio_analysis"]["caption_proposals"],
        )

    def test_openrouter_caption_fusion_is_text_only(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps({
                    "model": "google/gemini-3.1-flash-lite",
                    "usage": {"total_tokens": 12},
                    "choices": [{"message": {"content": '{"ok":true}'}}],
                }).encode("utf-8")

        captured = {}

        def fake_open(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return Response()

        with patch("repair_v2_annotations_moss.urllib.request.urlopen", side_effect=fake_open):
            response, metadata = openrouter_generate(
                "Fuse these text packets", "test-key", "google/gemini-3.1-flash-lite", 900, 30
            )
        self.assertEqual('{"ok":true}', response)
        self.assertEqual(12, metadata["usage"]["total_tokens"])
        serialized = json.dumps(captured["payload"])
        self.assertNotIn("input_audio", serialized)
        self.assertNotIn("audio_url", serialized)

    def test_preview_is_verified_before_final_training(self) -> None:
        source = (SCRIPTS / "orchestrate_v2.py").read_text(encoding="utf-8")
        self.assertLess(source.index('"edm-v2-upload-preview"'), source.index('"edm-v2-train-final"'))
        self.assertLess(source.index('"edm-v2-verify-preview"'), source.index('"edm-v2-train-final"'))

    def test_all_audio_is_uploaded_and_verified_after_final_training(self) -> None:
        source = (SCRIPTS / "orchestrate_v2.py").read_text(encoding="utf-8")
        final = source.index('"edm-v2-train-final"')
        prepare = source.index('"edm-v2-prepare-audio-dataset"')
        upload = source.index('"edm-v2-upload-audio-dataset"')
        verify = source.index('"edm-v2-verify-audio-dataset"')
        package = source.index('"edm-v2-package-release"')
        self.assertLess(final, prepare)
        self.assertLess(prepare, upload)
        self.assertLess(upload, verify)
        self.assertLess(verify, package)
        audit = (SCRIPTS / "audit_v2_objective.py").read_text(encoding="utf-8")
        self.assertIn("audio_dataset_clean_verification_report.json", audit)
        self.assertIn('audio_clean.get("records") != 231', audit)

    def test_final_release_requires_absolute_listening_quality(self) -> None:
        orchestrator = (SCRIPTS / "orchestrate_v2.py").read_text(encoding="utf-8")
        self.assertLess(
            orchestrator.index('"edm-v2-quality-final"'),
            orchestrator.index('"edm-v2-package-release"'),
        )
        package = (SCRIPTS / "package_v2_release.py").read_text(encoding="utf-8")
        self.assertIn("final_listening_quality_report.json", package)
        self.assertIn("pristine-xl-base-absolute-quality", orchestrator)

    def test_caption_repairs_finish_before_tensor_preprocessing(self) -> None:
        orchestrator = (SCRIPTS / "orchestrate_v2.py").read_text(encoding="utf-8")
        self.assertLess(
            orchestrator.index('"edm-v2-verify-claims-0"'),
            orchestrator.index('"edm-v2-repair-captions-0"'),
        )
        self.assertLess(
            orchestrator.index('"edm-v2-apply-caption-repairs"'),
            orchestrator.index('"edm-v2-preprocess-train-0"'),
        )
        apply_script = (SCRIPTS / "apply_v2_annotation_repairs.py").read_text(encoding="utf-8")
        self.assertIn('"tensors_train"', apply_script)
        self.assertIn('"tensor_validation_report.json"', apply_script)
        self.assertIn('"build_v2_dataset.py"', apply_script)
        self.assertIn("CAPTION_COMPILER_REVISION", apply_script)
        self.assertIn("archive_stale_model_outputs", apply_script)
        self.assertIn("fresh_rank32_outputs_required", apply_script)
        verifier = (SCRIPTS / "verify_v2_audio_claims_moss.py").read_text(encoding="utf-8")
        self.assertIn("CLAIM_VERIFIER_REVISION", verifier)
        pilot = (SCRIPTS.parent / "server" / "supervisor" / "edm-v2-repair-captions-pilot.sh").read_text(encoding="utf-8")
        self.assertIn("--num-shards 231", pilot)
        self.assertIn("--shard-index 0", pilot)

    def test_checkpoint_sync_includes_non_fifth_best_val(self) -> None:
        source = (SCRIPTS / "sync_v2_checkpoints_hf.py").read_text(encoding="utf-8")
        self.assertIn("def sync_best", source)
        self.assertIn('path_in_repo="checkpoints/best_val"', source)
        self.assertIn("epoch % 5 == 0", source)
        validator = (SCRIPTS / "validate_training_v2.py").read_text(encoding="utf-8")
        self.assertIn("{5, 10, 15, 20, 25, 30}", validator)

    def test_user_cutoff_excludes_later_evaluation_checkpoints(self) -> None:
        source = (SCRIPTS / "evaluate_v2_checkpoints.py").read_text(encoding="utf-8")
        self.assertIn('training_stop_override.json', source)
        self.assertIn('int(match.group(1)) > cutoff', source)
        finalize = (SCRIPTS / "finalize_v2_user_stop.py").read_text(encoding="utf-8")
        self.assertIn('source_checkpoints_deleted": False', finalize)

    def test_checkpoint_evaluation_uses_only_fixed_lora_scale(self) -> None:
        source = (SCRIPTS / "evaluate_v2_checkpoints.py").read_text(encoding="utf-8")
        self.assertIn("LORA_SCALES = (0.5,)", source)
        self.assertIn("handler.set_lora_scale(label, lora_scale)", source)
        self.assertIn('"--final-only"', source)
        launcher = (
            SCRIPTS.parent / "server" / "supervisor" / "edm-v2-evaluate-checkpoints.sh"
        ).read_text(encoding="utf-8")
        self.assertNotIn("--final-only", launcher)
        selection = (SCRIPTS / "select_v2_checkpoint.py").read_text(encoding="utf-8")
        self.assertIn('"selected_lora_scale": selected["lora_scale"]', selection)

    def test_moss_retry_separates_style_alignment_from_melody_coherence(self) -> None:
        scorer = (SCRIPTS / "score_v2_checkpoints_moss.py").read_text(encoding="utf-8")
        self.assertIn('"melody_score_contradicts_coherent_evidence" in last_error', scorer)
        self.assertIn("belongs only in prompt_alignment", scorer)
        self.assertIn('"at least 3 unless the audio itself is melodically incoherent', scorer)

    def test_v2_preview_and_final_packages_include_prompt_enhancer(self) -> None:
        for name in ("package_v2_preview.py", "package_v2_release.py"):
            source = (SCRIPTS / name).read_text(encoding="utf-8")
            self.assertIn('scripts" / "prompt_enhancer.py', source)
            self.assertIn('scripts" / "enhance_prompt_openrouter.py', source)

    def test_experimental_inference_hotfix_is_clean_verified(self) -> None:
        source = (SCRIPTS / "publish_experimental_inference_hotfix.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("CommitOperationAdd", source)
        self.assertIn("snapshot_download", source)
        self.assertIn("SHA256SUMS", source)
        self.assertIn("CUSTOM_SECTION_FIXTURE", source)
        self.assertIn("custom_section_hotfix_verification.json", source)

    def test_annotation_wip_backup_excludes_audio_tokens_and_raw_responses(self) -> None:
        source = (SCRIPTS / "upload_v2_annotation_wip.py").read_text(encoding="utf-8")
        self.assertIn('"moss_annotations", "claim_consensus", "caption_repairs"', source)
        self.assertIn('"raw_model_responses_included": False', source)
        self.assertIn('"audio_included": False', source)
        self.assertIn('"tokens_included": False', source)
        self.assertIn("SECRET_PATTERNS", source)
        self.assertIn("clean_download_hash_match", source)

    def test_colab_inference_overrides_notebook_only_matplotlib_backend(self) -> None:
        inference = (SCRIPTS / "infer_v2_release.py").read_text(encoding="utf-8")
        self.assertIn('os.environ["MPLBACKEND"] = "Agg"', inference)
        self.assertIn('"thinking": False', inference)
        self.assertIn('"ace_lm_model": "acestep-5Hz-lm-1.7B"', inference)
        self.assertIn("LLMHandler", inference)
        self.assertIn("experimental-r32", inference)
        notebook = (SCRIPTS.parent / "notebooks" / "melodic_edm_core_v2_colab.ipynb").read_text(encoding="utf-8")
        self.assertIn("'MPLBACKEND': 'Agg'", notebook)

    def test_colab_exposes_sampling_lora_and_optional_enhancer_in_one_cell(self) -> None:
        notebook_path = SCRIPTS.parent / "notebooks" / "melodic_edm_core_v2_colab.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = ["".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        controls = [cell for cell in code_cells if "INFERENCE_STEPS =" in cell]
        self.assertEqual(1, len(controls))
        for setting in (
            "USE_OPENROUTER_ENHANCER =",
            "REFERENCE_ARTIST =",
            "REFERENCE_TRACK_TITLE =",
            "REQUIRED_PROMPT_TERMS =",
            "USE_ACE_LM_THINKING =",
            "ACE_LM_MODEL =",
            "USE_LORA =",
            "LORA_SCALE =",
            "GUIDANCE_SCALE =",
            "SAMPLER_MODE =",
            "DCW_ENABLED =",
            "ENABLE_NORMALIZATION =",
            "BATCH_SIZE =",
            "AUDIO_FORMAT =",
        ):
            self.assertIn(setting, controls[0])
        self.assertIn("DURATION_SECONDS = 180", controls[0])
        self.assertIn("SECTIONS =", controls[0])
        self.assertIn("CUSTOM_LYRICS =", controls[0])
        self.assertIn("WARN_SECTION_SECONDS_BELOW =", controls[0])
        self.assertIn("never changed from the section count", controls[0])
        self.assertIn("0–3 is safer than a long list", controls[0])
        self.assertIn("USE_LORA = False", controls[0])
        self.assertIn("USE_OPENROUTER_ENHANCER = False", controls[0])
        self.assertIn("REFERENCE_ARTIST = 'Xomu'", controls[0])
        self.assertIn("REFERENCE_TRACK_TITLE = 'Lanterns'", controls[0])
        self.assertIn("REQUIRED_PROMPT_TERMS = []", controls[0])
        self.assertIn("generated-xomu-lanterns", controls[0])
        self.assertNotIn("pipa", controls[0].casefold())
        self.assertNotIn("dizi", controls[0].casefold())
        payload_builder = "\n".join(code_cells)
        self.assertIn("importlib.reload(prompt_enhancer_module)", payload_builder)
        self.assertIn("A stale release is loaded", payload_builder)
        self.assertIn("STRUCTURE_LYRICS = CUSTOM_LYRICS.strip() or sections_to_lyrics(SECTIONS)", payload_builder)
        self.assertIn("music_conditions['lyrics'] = STRUCTURE_LYRICS", payload_builder)
        self.assertIn("attach_inference_style_reference", payload_builder)
        self.assertIn("Values are unchanged", payload_builder)
        self.assertIn(
            "Bangchis/melodic-edm-core-v2-r32-experimental",
            notebook_path.read_text(encoding="utf-8"),
        )
        self.assertIn("if USE_OPENROUTER_ENHANCER:", "\n".join(code_cells))
        self.assertIn("if not USE_LORA or LORA_SCALE == 0:", "\n".join(code_cells))

    def test_absolute_quality_gate_requires_each_sample_to_follow_prompt(self) -> None:
        quality = summarize_quality([{
            "scores": {
                "prompt_alignment": 2, "melody": 5, "structure": 5, "audio_quality": 5,
            },
            "failure_modes": {
                "distorted": False,
                "collapsed": False,
                "static_loop": False,
                "intelligible_vocals": False,
            },
        }])
        self.assertFalse(quality["quality_accepted"])
        self.assertIn("individual_score_below_minimum:prompt_alignment:2:3", quality["errors"])

    def test_baseline_profile_measures_specialized_alignment_without_blocking_it(self) -> None:
        records = [{
            "scores": {
                "prompt_alignment": 2, "melody": 3, "structure": 3, "audio_quality": 4,
            },
            "failure_modes": {
                "distorted": False,
                "collapsed": False,
                "static_loop": False,
                "intelligible_vocals": False,
            },
        }]
        self.assertTrue(summarize_quality(records, profile="baseline")["quality_accepted"])
        self.assertFalse(summarize_quality(records, profile="candidate")["quality_accepted"])

    def test_final_prompt_alignment_cannot_materially_regress_from_best_val(self) -> None:
        base_quality = {
            "status": "pass", "quality_accepted": True, "errors": [],
        }
        current = [
            {"prompt_id": prompt_id, "scores": {"prompt_alignment": score}}
            for prompt_id, score in zip(("a", "b", "c"), (4, 4, 4))
        ]
        reference = [
            {"checkpoint": "best", "prompt_id": prompt_id, "scores": {"prompt_alignment": score}}
            for prompt_id, score in zip(("a", "b", "c"), (5, 4, 4))
        ]
        accepted = compare_prompt_alignment(
            base_quality, current, reference,
            reference_checkpoint="best", maximum_regression=0.34,
        )
        self.assertTrue(accepted["quality_accepted"])
        rejected = compare_prompt_alignment(
            base_quality,
            [
                {"prompt_id": prompt_id, "scores": {"prompt_alignment": score}}
                for prompt_id, score in zip(("a", "b", "c"), (3, 4, 4))
            ],
            reference,
            reference_checkpoint="best", maximum_regression=0.34,
        )
        self.assertFalse(rejected["quality_accepted"])
        self.assertTrue(any(error.startswith("final_prompt_alignment_regressed") for error in rejected["errors"]))

    def test_release_inference_accepts_user_sampling_and_output_settings(self) -> None:
        sampling, output = merged_generation_settings({
            "sampling": {
                "inference_steps": 72,
                "guidance_scale": 5.5,
                "sampler_mode": "heun",
                "dcw_enabled": False,
            },
            "output": {
                "batch_size": 2,
                "use_random_seed": False,
                "seeds": [11, 22],
                "audio_format": "flac",
            },
        })
        self.assertEqual(72, sampling["inference_steps"])
        self.assertEqual(5.5, sampling["guidance_scale"])
        self.assertEqual("heun", sampling["sampler_mode"])
        self.assertFalse(sampling["dcw_enabled"])
        self.assertEqual([11, 22], output["seeds"])
        self.assertEqual("flac", output["audio_format"])

    def test_tensor_merger_replaces_unsafe_symlink_with_hardlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            destination = root / "merged"
            source_dir.mkdir()
            destination.mkdir()
            source = source_dir / "sample.pt"
            source.write_bytes(b"tensor")
            target = destination / source.name
            target.symlink_to(os.path.relpath(source, destination))
            hardlink_tensor(source, target)
            self.assertFalse(target.is_symlink())
            self.assertTrue(os.path.samefile(source, target))

    def test_tensor_merger_refuses_unrelated_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.pt"
            target = root / "target.pt"
            source.write_bytes(b"source")
            target.write_bytes(b"different")
            with self.assertRaisesRegex(FileExistsError, "unrelated tensor"):
                hardlink_tensor(source, target)

    def test_exact_grouped_split_never_crosses_parent(self) -> None:
        rows = [
            {"sample_id": "a1", "video_id": "same"},
            {"sample_id": "a2", "video_id": "same"},
            {"sample_id": "b", "video_id": "b"},
            {"sample_id": "c", "video_id": "c"},
            {"sample_id": "d", "video_id": "d"},
        ]
        split = grouped_split(rows, validation_count=2, seed=42)
        self.assertEqual(2, sum(value == "validation" for value in split.values()))
        self.assertEqual(split["a1"], split["a2"])

    def test_parent_id_uses_shared_video_identity(self) -> None:
        first = {"video_id": "abc", "expected_title": "one"}
        second = {"video_id": "abc", "expected_title": "other catalog label"}
        self.assertEqual("youtube:abc", parent_song_id(first))
        self.assertEqual(parent_song_id(first), parent_song_id(second))

    def test_extracts_json_after_thinking_block(self) -> None:
        parsed = extract_json_object('<think>private reasoning</think>\n```json\n{"confidence": 0.9}\n```')
        self.assertEqual({"confidence": 0.9}, parsed)

    def test_caption_map_has_exact_required_order(self) -> None:
        mapped = caption_map({name: words(name, 45) for name in CAPTION_TYPES})
        self.assertEqual(list(CAPTION_TYPES), list(mapped))
        self.assertEqual([], validate_caption_set(mapped))

    def test_caption_validation_rejects_identity_and_short_text(self) -> None:
        captions = {
            "canonical": "Xomu " + words("melodic", 38),
            "composition": words("motif", 30),
            "production": words("synth", 30),
        }
        errors = validate_caption_set(captions, artist="Xomu", title="Lanterns")
        self.assertIn("canonical_word_count_39_outside_40_80", errors)
        self.assertIn("canonical_contains_artist_identity", errors)

    def test_exact_artist_track_style_reference_is_attached_to_all_three_views(self) -> None:
        audio_only = {
            "canonical": words("instrumental", 45),
            "composition": words("motif", 30),
            "production": words("synth", 30),
        }
        artist = "YUAN / 徐梦圆"
        title = "China-A"
        captions = attach_track_style_reference(audio_only, artist, title)
        prefix = track_style_reference(artist, title)
        self.assertEqual("artist-track-style-reference-v2", TRACK_STYLE_REFERENCE_REVISION)
        self.assertTrue(all(text.startswith(prefix + " ") for text in captions.values()))
        self.assertEqual([], validate_track_style_caption_set(captions, artist, title))
        self.assertEqual(
            caption_map(audio_only)["canonical"],
            detach_track_style_reference(captions["canonical"], artist, title),
        )

    def test_substituted_artist_or_title_style_reference_is_rejected(self) -> None:
        audio_only = {
            "canonical": words("instrumental", 45),
            "composition": words("motif", 30),
            "production": words("synth", 30),
        }
        captions = attach_track_style_reference(audio_only, "Xomu", "Lanterns")
        errors = validate_track_style_caption_set(captions, "TheFatRat", "Unity")
        self.assertEqual(3, sum("prefix_mismatch" in error for error in errors))

    def test_moss_supplement_rejects_missing_confidence(self) -> None:
        value = {
            "audible_facts": {
                "genre_and_style": ["melodic electronic"],
                "moods": ["uplifting"],
                "instruments_and_roles": [],
                "melody_and_motifs": "repeating pentatonic motif",
                "harmony": "wide sustained chords",
                "rhythm": "four-on-the-floor drums",
                "arrangement_and_sections": "intro, build, drop and outro",
                "production": "wide synths and clean sub bass",
                "uncertain_or_conflicting_facts": [],
            },
            "captions": {name: words(name, 45) for name in CAPTION_TYPES},
        }
        _, errors = validate_supplement(value, {"expected_artist": "", "expected_title": ""})
        self.assertIn("confidence_missing", errors)
        self.assertIn("confidence_outside_open_0_1", errors)

    def test_moss_supplement_normalizes_comma_lists(self) -> None:
        value = {
            "confidence": 0.9,
            "audible_facts": {
                "genre_and_style": "melodic EDM, cinematic electronic",
                "moods": "uplifting, adventurous",
                "instruments_and_roles": [
                    {"name": "synth pluck", "role": "main melody", "confidence": 0.8}
                ],
                "melody_and_motifs": "repeating pentatonic motif",
                "harmony": "wide sustained chords",
                "rhythm": "four-on-the-floor drums",
                "arrangement_and_sections": "intro, build, drop and outro",
                "production": "wide synths and clean sub bass",
                "uncertain_or_conflicting_facts": "none",
            },
            "captions": {name: words(name, 45) for name in CAPTION_TYPES},
        }
        supplement, errors = validate_supplement(
            value, {"expected_artist": "", "expected_title": ""}
        )
        self.assertEqual([], errors)
        self.assertEqual(
            ["melodic EDM", "cinematic electronic"],
            supplement["audible_facts"]["genre_and_style"],
        )
        self.assertEqual([], supplement["audible_facts"]["uncertain_or_conflicting_facts"])

    def test_moss_checkpoint_score_requires_evidence_per_dimension(self) -> None:
        score, errors = parse_score({
            "prompt_alignment": 4,
            "melody": 4,
            "structure": 3,
            "audio_quality": 5,
            "evidence": {"prompt_alignment": "Audible prompt instruments are present."},
        })
        self.assertEqual(4, score["scores"]["prompt_alignment"])
        self.assertIn("evidence_melody_missing", errors)
        self.assertIn("evidence_structure_missing", errors)
        self.assertIn("evidence_audio_quality_missing", errors)

    def test_moss_checkpoint_score_rejects_quality_evidence_contradiction(self) -> None:
        _, errors = parse_score({
            "prompt_alignment": 1,
            "melody": 3,
            "structure": 3,
            "audio_quality": 1,
            "evidence": {
                "prompt_alignment": "The requested instruments are absent.",
                "melody": "A stable melodic line is present.",
                "structure": "The arrangement develops over time.",
                "audio_quality": "The audio is clean and professionally mixed.",
            },
        })
        self.assertIn("audio_quality_score_contradicts_positive_evidence", errors)

    def test_moss_checkpoint_score_records_explicit_failure_modes(self) -> None:
        score, errors = parse_score({
            "prompt_alignment": 4,
            "melody": 4,
            "structure": 4,
            "audio_quality": 4,
            "evidence": {
                "prompt_alignment": "The requested sound is audible.",
                "melody": "The melody is coherent.",
                "structure": "Sections develop clearly.",
                "audio_quality": "The output is clean.",
            },
            "failure_modes": {
                "distorted": False,
                "collapsed": False,
                "static_loop": False,
                "intelligible_vocals": False,
            },
        })
        self.assertEqual([], errors)
        self.assertEqual(
            {
                "distorted": False,
                "collapsed": False,
                "static_loop": False,
                "intelligible_vocals": False,
            },
            score["failure_modes"],
        )

    def test_absolute_quality_gate_rejects_static_loop_even_with_high_scores(self) -> None:
        quality = summarize_quality([{
            "scores": {
                "prompt_alignment": 5, "melody": 5, "structure": 5, "audio_quality": 5,
            },
            "failure_modes": {
                "distorted": False,
                "collapsed": False,
                "static_loop": True,
                "intelligible_vocals": False,
            },
        }])
        self.assertFalse(quality["quality_accepted"])
        self.assertIn("audible_failure_mode:0:static_loop", quality["errors"])

    def test_moss_loop_evidence_overrides_inconsistent_false_flag(self) -> None:
        score, errors = parse_score({
            "prompt_alignment": 3,
            "melody": 2,
            "structure": 2,
            "audio_quality": 5,
            "evidence": {
                "prompt_alignment": "Some requested traits are audible.",
                "melody": "A simple motif repeats.",
                "structure": "The track consists of a single looped section that ends abruptly.",
                "audio_quality": "The output is clean.",
            },
            "failure_modes": {
                "distorted": False,
                "collapsed": False,
                "static_loop": False,
                "intelligible_vocals": False,
            },
        })
        self.assertEqual([], errors)
        self.assertTrue(score["failure_modes"]["static_loop"])

    def test_absolute_quality_gate_rejects_intelligible_vocals(self) -> None:
        quality = summarize_quality([{
            "scores": {
                "prompt_alignment": 5, "melody": 5, "structure": 5, "audio_quality": 5,
            },
            "failure_modes": {
                "distorted": False,
                "collapsed": False,
                "static_loop": False,
                "intelligible_vocals": True,
            },
        }])
        self.assertFalse(quality["quality_accepted"])
        self.assertIn("audible_failure_mode:0:intelligible_vocals", quality["errors"])

    def test_moss_checkpoint_scoring_is_resumable(self) -> None:
        source = (SCRIPTS / "score_v2_checkpoints_moss.py").read_text(encoding="utf-8")
        self.assertIn('previous.get("results", [])', source)
        self.assertIn('status": "in_progress"', source)
        self.assertIn(" CACHED", source)
        self.assertIn("SCORER_REVISION", source)
        self.assertIn('cached_row.get("audio_sha256") == audio_sha256', source)
        self.assertIn('cached_row.get("prompt_sha256") == prompt_sha256', source)

    def test_checkpoint_selector_does_not_amplify_trivial_diversity_noise(self) -> None:
        source = (SCRIPTS / "select_v2_checkpoint.py").read_text(encoding="utf-8")
        self.assertIn("minimum_range=0.01", source)
        self.assertIn('"diversity_minimum_meaningful_range": 0.01', source)

    def test_checkpoint_selector_prioritizes_prompt_alignment_over_other_listening_scores(self) -> None:
        source = (SCRIPTS / "select_v2_checkpoint.py").read_text(encoding="utf-8")
        self.assertIn('0.30 * value["prompt_alignment_score"]', source)
        self.assertIn('0.25 * value["other_listening_score"]', source)
        self.assertIn('value["prompt_alignment_not_worse_than_baseline"]', source)

    def test_final_objective_audit_requires_new_prompt_fidelity_lineage(self) -> None:
        source = (SCRIPTS / "audit_v2_objective.py").read_text(encoding="utf-8")
        for revision in (
            "multi-view-audio-claims-v2.7",
            "fixed-prompt-audio-judge-v2.2",
        ):
            self.assertIn(revision, source)
        self.assertEqual("openrouter-per-track-salient-audio-fusion-v3.2", CAPTION_COMPILER_REVISION)
        self.assertIn("CAPTION_COMPILER_REVISION", source)
        self.assertIn("per_record_caption_fusion_lineage_not_proven", source)
        self.assertIn("per_record_artist_track_style_reference_not_proven", source)
        self.assertIn("caption_compiler_provider_is_not_openrouter", source)
        self.assertIn("checkpoint_evaluation_scale_is_not_fixed_0_5", source)
        self.assertIn("dataset_does_not_use_three_fused_prompt_variants", source)
        self.assertIn("prompt_alignment_not_authoritative_in_checkpoint_selection", source)
        self.assertIn("selected_checkpoint_multi_seed_generation_invalid", source)
        self.assertIn("multi_style_multi_seed_quality_evidence_incomplete", source)


if __name__ == "__main__":
    unittest.main()
