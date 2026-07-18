from __future__ import annotations

import os
import json
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_mir import map_sections, normalize_edm_bpm, prepare_unique_inputs  # noqa: E402
from annotate_openrouter import parse_json_content, sanitize_annotation, sanitize_caption_text, validate_annotation  # noqa: E402
from build_acestep_dataset import choose_splits, choose_window, render_audio  # noqa: E402
from merge_tensors import write_loader_manifest  # noqa: E402
from prompt_enhancer import compile_caption  # noqa: E402
from validate_tensors import expected_by_split  # noqa: E402
from validate_smoke import resolve_adapter_dir  # noqa: E402
from validate_training import select_checkpoints  # noqa: E402
from annotate_qwen_local import (  # noqa: E402
    compile_canonical_caption,
    compile_section_captions,
    distinctive_detail_candidates,
)


class RecordPreservingTests(unittest.TestCase):
    def test_v2_ace_wrappers_pin_the_v2_worktree_on_pythonpath(self) -> None:
        wrappers = (
            "edm-v2-preprocess.sh",
            "edm-v2-train-smoke.sh",
            "edm-v2-train-main.sh",
            "edm-v2-train-final.sh",
            "edm-v2-evaluate-checkpoints.sh",
            "edm-v2-evaluate-final.sh",
        )
        for name in wrappers:
            source = (ROOT / "server" / "supervisor" / name).read_text(encoding="utf-8")
            self.assertIn('ace="$project/vendor/ACE-Step-1.5-v2"', source, name)
            self.assertIn('export PYTHONPATH="$ace${PYTHONPATH:+:$PYTHONPATH}"', source, name)

    def test_prompt_compiler_uses_only_structured_musical_facts(self) -> None:
        caption = compile_caption({
            "genre": "Chinese melodic gaming EDM",
            "mood": "uplifting and adventurous",
            "melody": "A bright two-bar pentatonic pipa hook repeats with altered endings and short dizi responses",
            "arrangement": "An atmospheric intro rises through a compact build into an energetic four-on-the-floor drop",
            "production": "Wide supersaw chords, clean sub bass, punchy electronic drums and spacious fantasy reverb support the melody",
        })
        self.assertTrue(40 <= len(caption.split()) <= 80)
        self.assertTrue(caption.startswith("Instrumental Chinese melodic gaming EDM"))
        with self.assertRaisesRegex(ValueError, "artist-name shortcuts"):
            compile_caption({
                "genre": "TheFatRat style EDM",
                "mood": "uplifting",
                "melody": "A detailed repeating melodic hook with several audible variations across each phrase",
                "arrangement": "An atmospheric intro develops into a short build and energetic melodic drop",
                "production": "Wide layered chords, sub bass, electronic drums and spacious reverb support the arrangement",
            })

    def test_release_delivery_sources_are_present(self) -> None:
        required = (
            "configs/inference_config.json",
            "configs/release_requirements.txt",
            "scripts/infer_release.py",
            "scripts/prompt_enhancer.py",
            "scripts/download_and_infer.py",
            "scripts/backup_resume_hf.py",
            "server/supervisor/edm-backup-resume.sh",
            "server/supervisor/edm-backup-resume.conf",
        )
        self.assertEqual([name for name in required if not (ROOT / name).is_file()], [])

    def test_only_clear_edm_half_time_bpm_is_doubled(self) -> None:
        self.assertEqual(normalize_edm_bpm(75), (150, "double_half_time_below_80"))
        self.assertEqual(normalize_edm_bpm(85), (85, None))
        self.assertEqual(normalize_edm_bpm(128), (128, None))
        self.assertEqual(normalize_edm_bpm(300), (None, None))

    def test_prechorus_maps_to_build_and_last_chorus_to_final_drop(self) -> None:
        segments = [
            SimpleNamespace(label="intro", start=0, end=10),
            SimpleNamespace(label="pre-chorus", start=10, end=20),
            SimpleNamespace(label="chorus", start=20, end=40),
            SimpleNamespace(label="chorus", start=40, end=60),
        ]
        self.assertEqual(
            [section["label"] for section in map_sections(segments)],
            ["Intro", "Build", "Drop", "Final Drop"],
        )

    def test_annotation_sections_must_exactly_match_mir(self) -> None:
        taxonomy = json.loads((ROOT / "configs" / "taxonomy.json").read_text(encoding="utf-8"))
        caption = (
            "Instrumental melodic EDM with an uplifting adventurous mood, led by a bright repeating synth pluck "
            "hook and supported by wide supersaw chords, clean sub bass, punchy electronic drums, airy pads, short "
            "rising transitions, an atmospheric opening, and a spacious energetic four-on-the-floor drop with clear "
            "melodic call-and-response phrases."
        )
        result = {
            "primary_genre": "melodic_edm",
            "secondary_genres": ["electro_house"],
            "style_families": ["gaming_melodic"],
            "moods": ["uplifting"],
            "main_instruments": [{"name": "synth_pluck", "role": "main_hook", "confidence": 0.9}],
            "canonical_caption": caption,
            "caption_variants": [
                {"type": "full", "text": caption},
                {"type": "composition", "text": "A repeating synth hook develops through varied endings."},
                {"type": "production", "text": "Wide chords, clean sub bass and punchy drums."},
                {"type": "tags", "text": "instrumental, melodic EDM, synth hook, uplifting drop"},
            ],
            "section_captions": [
                {"label": "Intro", "caption": "Atmospheric electronic opening with airy pads."},
                {"label": "Drop", "caption": "Energetic melodic drop with wide chords and punchy drums."},
            ],
            "annotation_confidence": 0.9,
        }
        row = {"expected_artist": "Example Artist"}
        mir = {"sections": [{"label": "Intro"}, {"label": "Drop"}]}
        self.assertEqual(validate_annotation(result, row, taxonomy, mir), [])
        production_text = result["caption_variants"][2]["text"]
        result["caption_variants"][2]["text"] = result["caption_variants"][1]["text"]
        self.assertIn("caption_variants_must_differ", validate_annotation(result, row, taxonomy, mir))
        result["caption_variants"][2]["text"] = production_text
        result["annotation_confidence"] = {"unexpected": 0.9}
        self.assertIn("invalid_annotation_confidence", validate_annotation(result, row, taxonomy, mir))
        result["annotation_confidence"] = 0.9
        result["section_captions"][0]["caption"] = "Builds"
        self.assertTrue(any(
            error.startswith("section_captions_word_count:Intro:1")
            for error in validate_annotation(result, row, taxonomy, mir)
        ))
        result["section_captions"][0]["caption"] = "Atmospheric electronic opening with airy pads."
        result["section_captions"] = [
            {"label": "Intro", "caption": "The intro section features soft and melodic."},
            {"label": "Drop", "caption": "The drop section features soft and melodic."},
        ]
        self.assertIn(
            "section_captions_semantic_duplicates",
            validate_annotation(result, row, taxonomy, mir),
        )
        result["section_captions"] = [
            {"label": "Intro", "caption": "Atmospheric electronic opening with airy pads."},
            {"label": "Drop", "caption": "Energetic melodic drop with wide chords and punchy drums."},
        ]
        result["section_captions"].append(
            {"label": "Break", "caption": "Quiet break with sparse plucks."}
        )
        self.assertEqual(validate_annotation(result, row, taxonomy, mir), [])
        result["section_captions"] = [item for item in result["section_captions"] if item["label"] != "Drop"]
        self.assertIn(
            "section_captions_missing_or_duplicate_mir_labels",
            validate_annotation(result, row, taxonomy, mir),
        )

    def test_low_confidence_instrument_is_sanitized_not_rejected(self) -> None:
        result = {
            "main_instruments": [
                {"name": "synth_lead", "role": "main_melody", "confidence": 0.9},
                {"name": "choir_texture", "role": "atmosphere", "confidence": 0.5},
            ]
        }
        audit = sanitize_annotation(result)
        self.assertEqual([item["name"] for item in result["main_instruments"]], ["synth_lead"])
        self.assertEqual(audit[0]["action"], "removed_low_confidence_instrument")

    def test_common_instrument_role_aliases_are_normalized(self) -> None:
        result = {
            "main_instruments": [
                {"name": "taiko", "role": "percussion", "confidence": 0.8},
                {"name": "unknown", "role": "unknown", "confidence": 0.2},
            ]
        }
        audit = sanitize_annotation(result)
        self.assertEqual(
            [item["role"] for item in result["main_instruments"]],
            ["drums", "atmosphere"],
        )
        self.assertEqual([item["action"] for item in audit], [
            "normalized_instrument_role", "normalized_instrument_role",
        ])

    def test_common_instrument_name_aliases_are_normalized(self) -> None:
        result = {
            "main_instruments": [
                {"name": "bass", "role": "bass", "confidence": 0.8},
                {"name": "drums", "role": "drums", "confidence": 0.9},
            ]
        }
        audit = sanitize_annotation(result)
        self.assertEqual(
            [item["name"] for item in result["main_instruments"]],
            ["sub_bass", "electronic_drums"],
        )
        self.assertEqual([item["action"] for item in audit], [
            "normalized_instrument_name", "normalized_instrument_name",
        ])

    def test_non_audio_quality_and_use_case_phrases_are_removed(self) -> None:
        text = (
            "The production is polished and spacious, with bright synths, making it ideal for energetic gaming content."
        )
        cleaned = sanitize_caption_text(text)
        self.assertEqual(cleaned, "The production is spacious, with bright synths.")
        self.assertEqual(
            sanitize_caption_text("A hook in F# minor scale with bright synths."),
            "A hook in minor tonality with bright synths.",
        )
        self.assertEqual(
            sanitize_caption_text("A strong emphasis on the F# minor key."),
            "A strong emphasis on the minor tonality.",
        )
        self.assertEqual(
            sanitize_caption_text("A strong emphasis on the a tonal center."),
            "A strong emphasis on the tonal center.",
        )
        self.assertEqual(
            sanitize_caption_text("A melodic outro with F major tonality."),
            "A melodic outro with major tonality.",
        )
        self.assertEqual(
            sanitize_caption_text("A melodic outro with major tonality tonality."),
            "A melodic outro with major tonality.",
        )
        self.assertEqual(
            sanitize_caption_text("A well-produced electronic track with a driving beat."),
            "An electronic track with a driving beat.",
        )

    def test_annotation_json_parser_accepts_provider_code_fence(self) -> None:
        self.assertEqual(parse_json_content("```json\n{\"status\": \"ok\"}\n```"), {"status": "ok"})
        self.assertEqual(parse_json_content("Result:\n{\"status\": \"ok\"}"), {"status": "ok"})
        self.assertEqual(
            parse_json_content('{"status": "ok"}\n{"provider_note": "extra"}'),
            {"status": "ok"},
        )
        self.assertEqual(
            parse_json_content(
                '{"confidence": 0.9, "name": "pipa"}\n'
                '{"primary_genre": "melodic_edm", "canonical_caption": "Instrumental music"}'
            )["primary_genre"],
            "melodic_edm",
        )

    def test_local_master_annotation_compiles_bounded_caption(self) -> None:
        annotation = {
            "primary_genre": "melodic_edm",
            "moods": ["uplifting", "adventurous", "energetic"],
            "main_instruments": [
                {"name": "piano", "role": "main_melody"},
                {"name": "synth_pluck", "role": "chordal_texture"},
                {"name": "electronic_drums", "role": "drums"},
            ],
            "melody": {"description": "A short repeated mid-register motif develops through altered endings"},
            "arrangement": {
                "intro": "soft piano introduces the motif",
                "buildup": "layered plucks increase the energy",
                "drop": "a bright synth hook leads the drop",
            },
            "production": {"bass": "Clean sub bass supports the harmony"},
        }
        caption = compile_canonical_caption(annotation)
        self.assertTrue(caption.startswith("Instrumental"))
        self.assertGreaterEqual(len(caption.split()), 40)
        self.assertLessEqual(len(caption.split()), 65)

    def test_local_section_compiler_covers_required_labels(self) -> None:
        annotation = {
            "arrangement": {"intro": "builds up", "drop": "intensifies", "outro": "piano fades away"},
            "main_instruments": [{"name": "piano"}, {"name": "synth_pluck"}],
        }
        mir = {"sections": [{"label": "Intro"}, {"label": "Drop"}, {"label": "Outro"}]}
        sections = compile_section_captions(annotation, mir)
        self.assertEqual([item["label"] for item in sections], ["Intro", "Drop", "Outro"])
        self.assertEqual(len({item["caption"] for item in sections}), 3)
        self.assertEqual(
            sections[0]["caption"],
            "The intro section develops the opening texture with piano and synth pluck.",
        )
        self.assertEqual(
            sections[1]["caption"],
            "The drop section intensifies the rhythmic and melodic drive with piano and synth pluck.",
        )
        self.assertEqual(sections[2]["caption"], "Piano fades away in the outro section.")

    def test_local_caption_compiler_repairs_section_fragments(self) -> None:
        annotation = {
            "primary_genre": "melodic_edm",
            "moods": ["uplifting", "energetic"],
            "main_instruments": [
                {"name": "synth_lead", "role": "main_melody"},
                {"name": "electronic_drums", "role": "drums"},
            ],
            "melody": {"description": "A repeated two-bar synth motif with altered endings"},
            "section_captions": [{
                "label": "Intro",
                "caption": "During the intro section, builds up with synth plucks and electronic drums.",
            }],
            "production": {"bass": "deep and driving", "space": "wide and immersive"},
        }
        caption = compile_canonical_caption(annotation)
        self.assertNotIn("During the intro section, builds", caption)
        self.assertIn("The intro section builds", caption)

    def test_local_section_compiler_handles_default_verbs_and_synth_lead_noun(self) -> None:
        annotation = {
            "arrangement": {"break": "Transitions", "outro": "soft synth lead texture"},
            "main_instruments": [{"name": "synth_lead"}, {"name": "electronic_drums"}],
        }
        sections = compile_section_captions(
            annotation, {"sections": [{"label": "Break"}, {"label": "Outro"}]}
        )
        self.assertTrue(sections[0]["caption"].startswith("The break section reduces"))
        self.assertEqual(
            sections[1]["caption"],
            "The outro section features soft synth lead texture.",
        )

    def test_local_section_compiler_replaces_adjective_only_fragments(self) -> None:
        annotation = {
            "arrangement": {"intro": "soft and melodic", "outro": "piano fades away"},
            "main_instruments": [{"name": "piano"}, {"name": "synth_pluck"}],
        }
        sections = compile_section_captions(
            annotation, {"sections": [{"label": "Intro"}, {"label": "Outro"}]}
        )
        self.assertIn("develops the opening texture", sections[0]["caption"])
        self.assertEqual(sections[1]["caption"], "Piano fades away in the outro section.")

    def test_local_section_compiler_replaces_repeated_long_arrangement_phrases(self) -> None:
        annotation = {
            "arrangement": {
                "intro": "maintains the melody with added layers",
                "outro": "maintains the melody with added layers",
            },
            "main_instruments": [{"name": "pipa"}, {"name": "guzheng"}],
        }
        sections = compile_section_captions(
            annotation, {"sections": [{"label": "Intro"}, {"label": "Outro"}]}
        )
        self.assertIn("develops the opening texture", sections[0]["caption"])
        self.assertIn("winds down the arrangement", sections[1]["caption"])

    def test_collision_details_are_grounded_in_master_fields(self) -> None:
        annotation = {
            "arrangement": {"outro": "slow and reflective", "drop": "powerful and energetic"},
            "production": {"description": "Wide synth layers and driving electronic drums"},
        }
        candidates = distinctive_detail_candidates(annotation)
        self.assertIn("The outro has a slow and reflective character.", candidates)
        self.assertIn("The drop has a powerful and energetic character.", candidates)
        self.assertIn("Wide synth layers and driving electronic drums.", candidates)

    def test_mir_validator_requires_exact_record_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data" / "mir").mkdir(parents=True)
            manifest = {
                "sample_id": "catalog__001",
                "record_key": "catalog__001",
                "quality_status": "accepted",
                "duration": 120.0,
            }
            (root / "data" / "training_audio_manifest.jsonl").write_text(
                json.dumps(manifest) + "\n", encoding="utf-8"
            )
            mir = {
                "sample_id": "catalog__001",
                "analysis_status": "complete",
                "bpm": 128,
                "keyscale": "F minor",
                "key_confidence": 0.8,
                "timesignature": "4",
                "beats": [0, 1, 2, 3],
                "downbeats": [0, 4, 8],
                "sections": [{"label": "Drop", "start": 0, "end": 120}],
            }
            mir_path = root / "data" / "mir" / "catalog__001.json"
            mir_path.write_text(json.dumps(mir), encoding="utf-8")
            command = [sys.executable, str(ROOT / "scripts" / "validate_mir.py"), "--project-root", str(root)]
            self.assertEqual(subprocess.run(command, capture_output=True, check=False).returncode, 0)
            mir_path.unlink()
            self.assertEqual(subprocess.run(command, capture_output=True, check=False).returncode, 1)

    def test_mir_inputs_keep_unique_sample_basenames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_a = root / "a" / "instrumental.flac"
            source_b = root / "b" / "instrumental.flac"
            source_a.parent.mkdir()
            source_b.parent.mkdir()
            source_a.write_bytes(b"first")
            source_b.write_bytes(b"second")
            paths = prepare_unique_inputs(
                [
                    {"sample_id": "catalog__001", "training_audio_path": str(source_a)},
                    {"sample_id": "catalog__002", "training_audio_path": str(source_b)},
                ],
                root / "inputs",
            )
            self.assertEqual([Path(path).name for path in paths], ["catalog__001.flac", "catalog__002.flac"])
            self.assertTrue(Path(paths[0]).samefile(source_a))
            self.assertTrue(Path(paths[1]).samefile(source_b))
            self.assertFalse(Path(paths[0]).is_symlink())

    def test_shared_audio_records_stay_in_same_split(self) -> None:
        rows = [
            {"sample_id": "a", "raw_path": "/raw/shared.flac"},
            {"sample_id": "b", "raw_path": "/raw/shared.flac"},
            {"sample_id": "c", "raw_path": "/raw/other.flac"},
        ]
        splits = choose_splits(rows, val_ratio=0.34, seed=42)
        self.assertEqual(set(splits), {"a", "b", "c"})
        self.assertEqual(splits["a"], splits["b"])

    def test_tensor_gate_preserves_every_record_id(self) -> None:
        expected = expected_by_split([
            {"sample_id": "catalog__001", "split": "train"},
            {"sample_id": "catalog__002", "split": "train"},
            {"sample_id": "catalog__003", "split": "validation"},
        ])
        self.assertEqual(expected["train"], {"catalog__001", "catalog__002"})
        self.assertEqual(expected["validation"], {"catalog__003"})
        with self.assertRaisesRegex(ValueError, "duplicate_sample_id"):
            expected_by_split([
                {"sample_id": "catalog__001", "split": "train"},
                {"sample_id": "catalog__001", "split": "validation"},
            ])

    def test_tensor_loader_manifest_keeps_every_shard_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "data" / "tensors_all"
            part0 = root / "data" / "tensors_part0" / "catalog__001.pt"
            part1 = root / "data" / "tensors_part1" / "catalog__002.pt"
            output.mkdir(parents=True)
            part0.parent.mkdir(parents=True)
            part1.parent.mkdir(parents=True)
            part0.touch()
            part1.touch()

            manifest_path = write_loader_manifest(output, [part0, part1], root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(
                manifest["samples"],
                ["data/tensors_part0/catalog__001.pt", "data/tensors_part1/catalog__002.pt"],
            )

    def test_checkpoint_selection_uses_middle_best_and_final(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            for epoch in (5, 10, 15, 20):
                checkpoint = output / "checkpoints" / f"epoch_{epoch}_loss_1.0000"
                checkpoint.mkdir(parents=True)
                (checkpoint / "adapter").mkdir()
                (checkpoint / "training_state.pt").touch()
            (output / "checkpoints" / "best_val" / "adapter").mkdir(parents=True)
            (output / "final" / "adapter").mkdir(parents=True)
            selected = select_checkpoints(output)
            self.assertEqual(Path(selected["middle"]).parent.name, "epoch_10_loss_1.0000")
            self.assertEqual(Path(selected["best_val"]).parent.name, "best_val")
            self.assertEqual(Path(selected["last"]).parent.name, "final")

    def test_adapter_resolver_accepts_nested_and_legacy_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "checkpoint" / "adapter"
            nested.mkdir(parents=True)
            self.assertEqual(resolve_adapter_dir(root / "checkpoint"), nested)
            self.assertEqual(resolve_adapter_dir(root / "legacy"), root / "legacy")

    def test_full_audio_uses_unique_hard_link(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "instrumental.flac"
            target = root / "sample__001.flac"
            source.write_bytes(b"audio")
            render_audio(source, target, 0.0, 180.0, 180.0)
            self.assertTrue(target.samefile(source))
            self.assertFalse(target.is_symlink())
            self.assertEqual(os.stat(target).st_ino, os.stat(source).st_ino)

    def test_overlength_window_prefers_drop_regions(self) -> None:
        mir = {
            "downbeats": list(range(0, 401, 4)),
            "sections": [
                {"label": "Intro", "start": 0, "end": 80},
                {"label": "Theme", "start": 80, "end": 180},
                {"label": "Drop", "start": 180, "end": 300},
                {"label": "Final Drop", "start": 300, "end": 400},
            ],
        }
        start, end = choose_window(400.0, mir, 240.0)
        self.assertLessEqual(end - start, 240.0)
        self.assertGreater(end, 300.0)
        self.assertLess(start, 300.0)


if __name__ == "__main__":
    unittest.main()
