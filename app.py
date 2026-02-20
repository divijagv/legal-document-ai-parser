import os
import asyncio
from flask import Flask, render_template, request, jsonify
from legal_parser_agent.agent import process_and_validate_document

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # 1. Get API Key (pass from headers to env if needed, or assume env)
    api_key = request.headers.get('x-goog-api-key')
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

    # 2. Get File
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # 3. Read File Bytes
        file_bytes = file.read()
        mime_type = file.content_type
        
        print(f"Processing file: {file.filename}, mime_type: {mime_type}, size: {len(file_bytes)} bytes")

        # 4. Call Agent (Async)
        # Using asyncio.run to call the async agent function from synchronous Flask
        result = asyncio.run(process_and_validate_document(file_bytes, mime_type))
        
        print(f"Analysis result: {result}")
        return jsonify(result)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error during analysis: {error_details}")
        return jsonify({"error": str(e), "details": error_details}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080)
