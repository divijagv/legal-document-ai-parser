import logging
import os
import asyncio
from flask import Flask, render_template, request, jsonify
from legal_parser_agent.agent import process_and_validate_document

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20MB, matches the web app's client-side limit
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

ALLOWED_MIME_TYPES = {"application/pdf", "image/png", "image/jpeg", "text/plain"}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    # Per-request API key, supplied by the browser. This is read once per
    # request and passed explicitly to the agent rather than mutated onto
    # the process environment, so concurrent requests from different users
    # can't clobber each other's key.
    api_key = request.headers.get('x-goog-api-key')
    if not api_key:
        return jsonify({"error": "Missing API key. Send it via the x-goog-api-key header."}), 401

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    file_bytes = file.read()
    mime_type = file.content_type

    if mime_type not in ALLOWED_MIME_TYPES:
        return jsonify({"error": f"Unsupported file type: {mime_type}"}), 400

    logger.info("Processing file: %s (%s, %d bytes)", file.filename, mime_type, len(file_bytes))

    # Scoped for the duration of this request only.
    previous_key = os.environ.get("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = api_key
    try:
        result = asyncio.run(process_and_validate_document(file_bytes, mime_type))
        return jsonify(result)
    except ValueError as e:
        # Known, user-facing errors (bad key, quota, invalid JSON, etc.)
        logger.warning("Analysis failed for %s: %s", file.filename, e)
        return jsonify({"error": str(e)}), 400
    except Exception:
        # Unexpected errors: log full details server-side only. Never return
        # a traceback to the client — it can leak internals and, in this
        # case, extraction results containing PII.
        logger.exception("Unexpected error analyzing %s", file.filename)
        return jsonify({"error": "Something went wrong analyzing this document. Please try again."}), 500
    finally:
        if previous_key is not None:
            os.environ["GOOGLE_API_KEY"] = previous_key
        else:
            os.environ.pop("GOOGLE_API_KEY", None)


if __name__ == '__main__':
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode, port=8080)
