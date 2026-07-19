"""Run every benchmark document through the extraction agent.

Usage (from the repo root, with GOOGLE_API_KEY set or in legal_parser_agent/.env):

    uv run python benchmark/run_benchmark.py

Writes one JSON result per document to benchmark/results/, then invokes the
scorer to produce benchmark/REPORT.md. Documents that fail are recorded with
an "error" key so a partial run still scores.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from legal_parser_agent.agent import process_and_validate_document  # noqa: E402

DOCS_DIR = Path(__file__).parent / "documents"
RESULTS_DIR = Path(__file__).parent / "results"


async def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    docs = sorted(DOCS_DIR.glob("*.txt"))
    if not docs:
        sys.exit("No documents found in benchmark/documents/")

    for i, doc in enumerate(docs, 1):
        out_path = RESULTS_DIR / f"{doc.stem}.json"
        print(f"[{i}/{len(docs)}] {doc.name} ... ", end="", flush=True)
        try:
            data = await process_and_validate_document(doc.read_bytes(), "text/plain")
            out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            print("ok")
        except Exception as e:
            out_path.write_text(json.dumps({"error": str(e)}, indent=2), encoding="utf-8")
            print(f"FAILED: {e}")

    print("\nScoring...")
    from score import score  # noqa: E402  (same directory)

    score()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    asyncio.run(main())
