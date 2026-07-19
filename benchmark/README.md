# Extraction Accuracy Benchmark

Measures how accurately the parser extracts structured fields from legal-process
documents, against hand-labeled ground truth.

## Contents

- `documents/` — 10 test documents: 8 legal-process documents covering varied
  issuers (federal grand jury, IRS, state Attorney General, civil litigation,
  Medicaid Fraud Control Unit, Adult Protective Services, a bilingual
  Spanish-language judicial subpoena, and a federal court order), plus 2
  non-legal distractors (an invoice, a lease) to measure false positives.
- `ground_truth.json` — hand-labeled expected values for every document.
- `run_benchmark.py` — runs each document through the extraction agent.
- `score.py` — compares results to ground truth and writes `REPORT.md`.

**All documents are synthetic.** They mirror the structure and language of real
legal process, but every name, SSN, account number, and case identifier is
fictional. This lets the corpus (and any real document you compare against it)
be published without exposing anyone's personal information.

## Running it

From the repo root, with `GOOGLE_API_KEY` set in your environment or in
`legal_parser_agent/.env`:

```bash
uv run python benchmark/run_benchmark.py
```

This calls the Gemini API once per document (10 calls), writes raw outputs to
`benchmark/results/`, and generates `benchmark/REPORT.md` with:

- document classification (legal process vs. not),
- per-field correct / wrong / missed / hallucinated counts,
- per-document field accuracy, model confidence, and review flags.

To re-score existing results without re-calling the API:

```bash
uv run python benchmark/score.py
```

## Extending the corpus

Add a `.txt` document to `documents/` and a matching entry in
`ground_truth.json` (same base filename). Fields use dotted paths into the
extraction schema (e.g. `case_details.due_date`); use `null` for fields the
document genuinely doesn't contain — the scorer uses those to catch
hallucinated values.
