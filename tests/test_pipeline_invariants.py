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
from annotate_openrouter import validate_annotation  # noqa: E402
from build_acestep_dataset import choose_splits, choose_window, render_audio  # noqa: E402


class RecordPreservingTests(unittest.TestCase):
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
        result["section_captions"].append(
            {"label": "Break", "caption": "Quiet break with sparse plucks."}
        )
        self.assertIn("section_caption_labels_must_match_mir", validate_annotation(result, row, taxonomy, mir))

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
