"""Focused invariants for the second grouped multi-prompt training release."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from v2_common import (  # noqa: E402
    CAPTION_TYPES,
    caption_map,
    extract_json_object,
    grouped_split,
    parent_song_id,
    validate_caption_set,
)
from annotate_moss_music import validate_supplement  # noqa: E402
from score_v2_checkpoints_moss import parse_score  # noqa: E402
from merge_v2_tensors import hardlink_tensor  # noqa: E402
from infer_v2_release import merged_generation_settings  # noqa: E402


def words(prefix: str, count: int) -> str:
    """Return deterministic caption text with exactly *count* tokens."""
    return " ".join([prefix] + [f"word{index}" for index in range(1, count)])


class V2PipelineTest(unittest.TestCase):
    """Protect grouping, prompt coverage and MOSS response parsing."""

    def test_training_config_names_separate_self_and_cross_attention_scope(self) -> None:
        config = json.loads(
            (SCRIPTS.parent / "configs" / "v2" / "train_val.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            "separate_self_and_cross_attention_projections",
            config["adapter"]["attention_scope"],
        )

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

    def test_checkpoint_sync_includes_non_tenth_best_val(self) -> None:
        source = (SCRIPTS / "sync_v2_checkpoints_hf.py").read_text(encoding="utf-8")
        self.assertIn("def sync_best", source)
        self.assertIn('path_in_repo="checkpoints/best_val"', source)

    def test_user_cutoff_excludes_later_evaluation_checkpoints(self) -> None:
        source = (SCRIPTS / "evaluate_v2_checkpoints.py").read_text(encoding="utf-8")
        self.assertIn('training_stop_override.json', source)
        self.assertIn('int(match.group(1)) > cutoff', source)
        finalize = (SCRIPTS / "finalize_v2_user_stop.py").read_text(encoding="utf-8")
        self.assertIn('source_checkpoints_deleted": False', finalize)

    def test_v2_preview_and_final_packages_include_prompt_enhancer(self) -> None:
        for name in ("package_v2_preview.py", "package_v2_release.py"):
            source = (SCRIPTS / name).read_text(encoding="utf-8")
            self.assertIn('scripts" / "prompt_enhancer.py', source)
            self.assertIn('scripts" / "enhance_prompt_openrouter.py', source)

    def test_colab_inference_overrides_notebook_only_matplotlib_backend(self) -> None:
        inference = (SCRIPTS / "infer_v2_release.py").read_text(encoding="utf-8")
        self.assertIn('os.environ["MPLBACKEND"] = "Agg"', inference)
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
        self.assertIn("if USE_OPENROUTER_ENHANCER:", "\n".join(code_cells))
        self.assertIn("if not USE_LORA or LORA_SCALE == 0:", "\n".join(code_cells))

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

    def test_moss_checkpoint_scoring_is_resumable(self) -> None:
        source = (SCRIPTS / "score_v2_checkpoints_moss.py").read_text(encoding="utf-8")
        self.assertIn('previous.get("results", [])', source)
        self.assertIn('status": "in_progress"', source)
        self.assertIn(" CACHED", source)

    def test_checkpoint_selector_does_not_amplify_trivial_diversity_noise(self) -> None:
        source = (SCRIPTS / "select_v2_checkpoint.py").read_text(encoding="utf-8")
        self.assertIn("minimum_range=0.01", source)
        self.assertIn('"diversity_minimum_meaningful_range": 0.01', source)


if __name__ == "__main__":
    unittest.main()
