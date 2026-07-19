"""Score benchmark results against ground truth and write REPORT.md.

Matching rules (deliberately lenient where legal formatting varies):
- Names/companies: case-insensitive, punctuation-stripped, containment either way.
- SSNs / account numbers: digits-and-letters only comparison.
- Dates: normalized to YYYY-MM-DD where recognizable.
- Subpoena subtype: correct if ANY ground-truth keyword appears in the value.

Per-field outcomes:
- correct        extracted value matches ground truth
- wrong          extracted a value, but it doesn't match
- missed         ground truth has a value, extraction returned null
- hallucinated   ground truth is null, extraction returned a value
"""

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

BENCH_DIR = Path(__file__).parent
RESULTS_DIR = BENCH_DIR / "results"
REPORT = BENCH_DIR / "REPORT.md"

MONTHS = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], 1)}


def norm_text(v):
    return re.sub(r"[^a-z0-9 ]", "", str(v).lower()).strip()


def norm_id(v):
    return re.sub(r"[^a-z0-9]", "", str(v).lower())


def norm_date(v):
    s = str(v).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", s)
    if m:
        return f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    m = re.fullmatch(r"([A-Za-z]+) (\d{1,2}),? (\d{4})", s)
    if m and m.group(1).lower() in MONTHS:
        return f"{m.group(3)}-{MONTHS[m.group(1).lower()]:02d}-{int(m.group(2)):02d}"
    return s


def get_path(data, dotted):
    cur = data
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def match(field, expected, actual):
    if field.endswith((".ssn", ".bank_account_number", ".case_number")):
        return norm_id(expected) == norm_id(actual)
    if field.endswith((".date_from", ".date_to", ".due_date", ".dob")):
        return norm_date(expected) == norm_date(actual)
    a, b = norm_text(expected), norm_text(actual)
    return bool(a) and bool(b) and (a in b or b in a)


def score():
    gt = json.loads((BENCH_DIR / "ground_truth.json").read_text(encoding="utf-8"))

    per_field = defaultdict(lambda: defaultdict(int))
    doc_rows = []
    classification = {"tp": 0, "fn": 0, "tn": 0, "fp": 0}
    errors = []

    for name, truth in gt.items():
        result_path = RESULTS_DIR / f"{name}.json"
        if not result_path.exists():
            errors.append(f"{name}: no result file")
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if "error" in result and "is_subpoena" not in result:
            errors.append(f"{name}: {result['error']}")
            continue

        predicted_subpoena = bool(result.get("is_subpoena"))
        if truth["is_subpoena"]:
            classification["tp" if predicted_subpoena else "fn"] += 1
        else:
            classification["fp" if predicted_subpoena else "tn"] += 1

        subtype_ok = None
        if truth["is_subpoena"] and truth["subtype_keywords"]:
            subtype = norm_text(result.get("subpoena_subtype") or "")
            subtype_ok = any(norm_text(k) in subtype for k in truth["subtype_keywords"])
            per_field["subpoena_subtype"]["correct" if subtype_ok else "wrong"] += 1

        n_correct = n_total = 0
        for field, expected in truth["fields"].items():
            actual = get_path(result, field)
            has_expected = expected not in (None, "")
            has_actual = actual not in (None, "")
            if has_expected:
                n_total += 1
                if has_actual and match(field, expected, actual):
                    per_field[field]["correct"] += 1
                    n_correct += 1
                elif has_actual:
                    per_field[field]["wrong"] += 1
                else:
                    per_field[field]["missed"] += 1
            elif has_actual:
                per_field[field]["hallucinated"] += 1

        conf = (result.get("extraction_metadata") or {}).get("confidence_score")
        doc_rows.append({
            "name": name,
            "classified": predicted_subpoena == truth["is_subpoena"],
            "fields": f"{n_correct}/{n_total}" if n_total else "—",
            "confidence": conf,
            "needs_review": result.get("needs_review"),
        })

    # ---- report ----
    total_correct = sum(f["correct"] for f in per_field.values())
    total_wrong = sum(f["wrong"] for f in per_field.values())
    total_missed = sum(f["missed"] for f in per_field.values())
    total_halluc = sum(f["hallucinated"] for f in per_field.values())
    total_expected = total_correct + total_wrong + total_missed

    lines = [
        "# Extraction Accuracy Benchmark",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}  ",
        "Model: `gemini-2.5-flash` via the ADK backend (`process_and_validate_document`), schema-enforced output.",
        "",
        "## Corpus",
        "",
        f"{len(gt)} documents in `benchmark/documents/`: 8 legal-process documents "
        "(federal grand jury subpoena, IRS summons, state AG civil investigative demand, "
        "civil subpoena duces tecum, Medicaid Fraud Control Unit subpoena, Adult Protective "
        "Services administrative subpoena, a bilingual Spanish-language judicial subpoena, "
        "and a federal court order compelling production) plus 2 non-legal distractors "
        "(an invoice and a residential lease) to measure false positives.",
        "",
        "**All documents are synthetic.** They follow the structure and language of real "
        "legal process but every name, number, and case identifier is fictional. This avoids "
        "publishing real PII while still testing realistic formats. Ground truth was "
        "hand-labeled in `ground_truth.json`.",
        "",
        "## Document classification",
        "",
        "| | Predicted legal process | Predicted other |",
        "|---|---|---|",
        f"| **Is legal process** | {classification['tp']} | {classification['fn']} |",
        f"| **Is other** | {classification['fp']} | {classification['tn']} |",
        "",
        "## Field extraction",
        "",
        f"Across all expected fields: **{total_correct}/{total_expected} correct "
        f"({100 * total_correct / total_expected:.0f}%)** — {total_wrong} wrong, "
        f"{total_missed} missed, {total_halluc} hallucinated (value invented where ground truth is null)."
        if total_expected else "No field results.",
        "",
        "| Field | Correct | Wrong | Missed | Hallucinated |",
        "|---|---|---|---|---|",
    ]
    for field in sorted(per_field):
        f = per_field[field]
        lines.append(f"| `{field}` | {f['correct']} | {f['wrong']} | {f['missed']} | {f['hallucinated']} |")

    lines += [
        "",
        "## Per-document results",
        "",
        "| Document | Classified correctly | Fields correct | Confidence | Flagged for review |",
        "|---|---|---|---|---|",
    ]
    for row in doc_rows:
        conf = f"{row['confidence']:.2f}" if isinstance(row["confidence"], (int, float)) else "—"
        review = {True: "yes", False: "no"}.get(row["needs_review"], "—")
        lines.append(f"| {row['name']} | {'✅' if row['classified'] else '❌'} | {row['fields']} | {conf} | {review} |")

    if errors:
        lines += ["", "## Errors", ""] + [f"- {e}" for e in errors]

    lines += [
        "",
        "## Method notes",
        "",
        "- Matching is deliberately lenient where formatting legitimately varies: names and "
        "companies match case-insensitively with containment; identifiers compare "
        "alphanumerics only; dates are normalized to ISO format.",
        "- `subpoena_subtype` counts as correct if any expected keyword appears in the value.",
        "- Scoring code: `benchmark/score.py`. Re-run with `uv run python benchmark/run_benchmark.py`.",
        "",
    ]

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT}")
    if total_expected:
        print(f"Field accuracy: {total_correct}/{total_expected} ({100 * total_correct / total_expected:.0f}%)")
    print(f"Classification: {classification}")


if __name__ == "__main__":
    score()
