"""Unit tests for the pure validation helpers.

These deliberately import only legal_parser_agent.validation (not .agent),
so they run without google-adk installed — e.g. in a lightweight CI job.
"""

import json

from legal_parser_agent.validation import clean_json_text, flag_needs_review


class TestCleanJsonText:
    def test_plain_json_untouched(self):
        assert clean_json_text('{"a": 1}') == '{"a": 1}'

    def test_strips_json_fence(self):
        raw = '```json\n{"a": 1}\n```'
        assert json.loads(clean_json_text(raw)) == {"a": 1}

    def test_strips_bare_fence(self):
        raw = '```\n{"a": 1}\n```'
        assert json.loads(clean_json_text(raw)) == {"a": 1}

    def test_strips_surrounding_whitespace(self):
        assert clean_json_text('  {"a": 1}  \n') == '{"a": 1}'


class TestFlagNeedsReview:
    def _doc(self, confidence=0.9, notes="", is_subpoena=True):
        return {
            "is_subpoena": is_subpoena,
            "extraction_metadata": {"confidence_score": confidence, "notes": notes},
        }

    def test_high_confidence_passes(self):
        assert flag_needs_review(self._doc(confidence=0.95)) is False

    def test_low_confidence_flagged(self):
        assert flag_needs_review(self._doc(confidence=0.5)) is True

    def test_threshold_boundary(self):
        assert flag_needs_review(self._doc(confidence=0.7)) is False
        assert flag_needs_review(self._doc(confidence=0.69)) is True

    def test_unclear_notes_flagged_even_with_high_confidence(self):
        assert flag_needs_review(self._doc(confidence=0.95, notes="Date field unclear")) is True

    def test_non_subpoena_never_flagged(self):
        assert flag_needs_review(self._doc(confidence=0.1, is_subpoena=False)) is False

    def test_missing_metadata_flagged(self):
        assert flag_needs_review({"is_subpoena": True}) is True

    def test_null_metadata_flagged(self):
        assert flag_needs_review({"is_subpoena": True, "extraction_metadata": None}) is True

    def test_null_confidence_flagged(self):
        doc = {"is_subpoena": True, "extraction_metadata": {"confidence_score": None, "notes": None}}
        assert flag_needs_review(doc) is True
