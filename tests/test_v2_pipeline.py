"""Focused invariants for the second grouped multi-prompt training release."""
from __future__ import annotations

import sys
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


def words(prefix: str, count: int) -> str:
    """Return deterministic caption text with exactly *count* tokens."""
    return " ".join([prefix] + [f"word{index}" for index in range(1, count)])


class V2PipelineTest(unittest.TestCase):
    """Protect grouping, prompt coverage and MOSS response parsing."""

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


if __name__ == "__main__":
    unittest.main()
