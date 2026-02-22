# Legal Document AI Parser

An AI-powered document intelligence platform for extracting structured data from unstructured legal documents (subpoenas, court orders, etc.).

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Google Gemini API Key

### Setup (Mac/Linux)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/divijagv/agentic-ai-legal-doc-parser.git
   cd "document parser"
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv_mac
   source venv_mac/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install flask google-adk google-genai
   ```

### Running the Application

1. **Set your API Key:**
   ```bash
   export GOOGLE_API_KEY="YOUR_ACTUAL_API_KEY"
   ```

2. **Start the Flask server:**
   ```bash
   python3 app.py
   ```

3. **Access the UI:**
   Open [http://127.0.0.1:8080](http://127.0.0.1:8080) in your browser.

## 🔒 Security
- This application does not store your API keys on the server.
- The UI allows you to provide a key that is stored locally in your browser session.
- Ensure you do not commit your real API keys to version control.

## 🛠 Features
- **AI-Powered Extraction**: Specifically tuned to parse legal subpoenas and court orders.
- **Multimodal Support**: Handles PDFs and Image files.
- **Interactive Assistant**: Ask follow-up questions about the extracted data.
- **Export**: Download results as JSON or CSV.
