"""Pure helpers for cleaning and validating extraction output.

Kept free of ADK/Gemini imports so they can be unit-tested (and run in CI)
without the heavy agent dependencies.
"""

import re

# Below this confidence score, an extraction is flagged for human review.
REVIEW_CONFIDENCE_THRESHOLD = 0.7


def clean_json_text(raw_text):
    """Strip markdown code fences the model sometimes wraps JSON in."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    return cleaned


def flag_needs_review(data):
    """Return True if a human should double-check this extraction.

    A document is flagged unless the model explicitly said it isn't a
    subpoena, and otherwise whenever confidence is low or the model's own
    notes mention ambiguity.
    """
    if data.get("is_subpoena") is False:
        return False

    metadata = data.get("extraction_metadata") or {}
    confidence = metadata.get("confidence_score") or 0.0
    notes = (metadata.get("notes") or "").lower()

    return confidence < REVIEW_CONFIDENCE_THRESHOLD or "unclear" in notes
