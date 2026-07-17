from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_mir import prepare_unique_inputs  # noqa: E402
from build_acestep_dataset import choose_splits, choose_window, render_audio  # noqa: E402


class RecordPreservingTests(unittest.TestCase):
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
