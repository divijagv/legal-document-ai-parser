"""
Command-line entry point for the local backend.

The primary product is the static web app (index.html/app.js), which needs no
Python at all. This CLI is a convenience for running the same ADK-based
extraction agent locally, e.g. for testing the prompt/schema against sample
documents, or scripting batch extraction.

Usage:
    export GOOGLE_API_KEY="your-key-here"     # or put it in legal_parser_agent/.env
    python main.py path/to/subpoena.pdf
    python main.py path/to/subpoena.pdf --pretty
"""

import argparse
import asyncio
import json
import mimetypes
import sys
from pathlib import Path

from legal_parser_agent.agent import process_and_validate_document


def guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type:
        return mime_type
    if path.suffix.lower() == ".txt":
        return "text/plain"
    return "application/octet-stream"


async def analyze_file(path: Path) -> dict:
    file_bytes = path.read_bytes()
    mime_type = guess_mime_type(path)
    return await process_and_validate_document(file_bytes, mime_type)


def main():
    parser = argparse.ArgumentParser(description="Extract structured data from a legal document using Gemini.")
    parser.add_argument("file", help="Path to a PDF, image, or text file to analyze.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the JSON output (default).")
    parser.add_argument("--compact", action="store_true", help="Print compact single-line JSON instead.")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.is_file():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        result = asyncio.run(analyze_file(path))
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.compact:
        print(json.dumps(result))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
