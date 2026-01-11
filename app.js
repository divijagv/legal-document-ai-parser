import { GoogleGenerativeAI } from "@google/generative-ai";

// Config & State
const state = {
    apiKey: localStorage.getItem('gemini_api_key') || '',
    file: null,
    isAnalyzing: false
};

// UI Elements
const elements = {
    apiKeyInput: document.getElementById('api-key'),
    dropZone: document.getElementById('drop-zone'),
    fileInput: document.getElementById('file-input'),
    fileInfo: document.getElementById('file-info'),
    fileName: document.querySelector('.file-name'),
    removeFile: document.getElementById('remove-file'),
    analyzeBtn: document.getElementById('analyze-btn'),
    statusSection: document.getElementById('status-section'),
    resultSection: document.getElementById('result-section'),
    jsonOutput: document.getElementById('json-output'),
    copyBtn: document.getElementById('copy-json'),
    progressBar: document.querySelector('.loader-progress')
};

// Initial Setup
if (state.apiKey) {
    elements.apiKeyInput.value = state.apiKey;
}

// Event Listeners
elements.apiKeyInput.addEventListener('input', (e) => {
    state.apiKey = e.target.value;
    localStorage.setItem('gemini_api_key', state.apiKey);
    updateAnalyzeButton();
});

elements.dropZone.addEventListener('click', () => elements.fileInput.click());

elements.fileInput.addEventListener('change', (e) => {
    handleFileSelect(e.target.files[0]);
});

elements.dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    elements.dropZone.classList.add('drag-over');
});

elements.dropZone.addEventListener('dragleave', () => {
    elements.dropZone.classList.remove('drag-over');
});

elements.dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    elements.dropZone.classList.remove('drag-over');
    handleFileSelect(e.dataTransfer.files[0]);
});

elements.removeFile.addEventListener('click', () => {
    state.file = null;
    elements.fileInput.value = '';
    elements.fileInfo.classList.add('hidden');
    elements.dropZone.classList.remove('hidden');
    updateAnalyzeButton();
});

elements.analyzeBtn.addEventListener('click', () => analyzeDocument());

elements.copyBtn.addEventListener('click', () => {
    navigator.clipboard.writeText(elements.jsonOutput.textContent);
    const originalText = elements.copyBtn.textContent;
    elements.copyBtn.textContent = 'Copied!';
    setTimeout(() => elements.copyBtn.textContent = originalText, 2000);
});

// Logic
function handleFileSelect(file) {
    if (!file) return;
    state.file = file;
    elements.fileName.textContent = file.name;
    elements.fileInfo.classList.remove('hidden');
    elements.dropZone.classList.add('hidden');
    updateAnalyzeButton();
}

function updateAnalyzeButton() {
    elements.analyzeBtn.disabled = !state.apiKey || !state.file || state.isAnalyzing;
}

async function analyzeDocument() {
    if (state.isAnalyzing) return;

    state.isAnalyzing = true;
    updateAnalyzeButton();

    elements.statusSection.classList.remove('hidden');
    elements.resultSection.classList.add('hidden');
    updateProgress(20, 'upload');

    try {
        const genAI = new GoogleGenerativeAI(state.apiKey);
        const model = genAI.getGenerativeModel({
            model: "gemini-1.5-flash",
            generationConfig: { responseMimeType: "application/json" }
        });

        const fileData = await fileToGenerativePart(state.file);
        updateProgress(50, 'parse');

        const prompt = `System Role: You are a specialized legal document parser. Your objective is to identify any legal subpoena, summons, or court order and extract pertinent details into a structured format.

Task: Analyze the provided document. If it is a subpoena of any kind, extract the data using the schema below. If the document is not a subpoena or a similar legal request for information, return exactly {}.

Instructions:
1. Universal Detection: Identify the document type regardless of the issuing agency (e.g., Federal, State, Criminal, Civil, or Administrative).
2. Subtype Labeling: Use the subpoena_subtype field to describe the specific nature of the document.
3. Handling Missing Data: If a field is not found in the text, return null. Do not guess or hallucinate values.
4. JSON Output ONLY.

Schema:
{
  "is_subpoena": boolean,
  "subpoena_subtype": "string", 
  "customer_details": {
    "name": "string",
    "ssn": "string",
    "bank_account_number": "string",
    "dob": "string"
  }
}`;

        const result = await model.generateContent([prompt, fileData]);
        const response = await result.response;
        const text = response.text();

        updateProgress(100, 'done');

        elements.jsonOutput.textContent = JSON.stringify(JSON.parse(text), null, 2);
        elements.resultSection.classList.remove('hidden');

    } catch (error) {
        console.error(error);
        alert('Error analyzing document: ' + error.message);
    } finally {
        state.isAnalyzing = false;
        updateAnalyzeButton();
        setTimeout(() => elements.statusSection.classList.add('hidden'), 3000);
    }
}

async function fileToGenerativePart(file) {
    const base64EncodedDataPromise = new Promise((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result.split(',')[1]);
        reader.readAsDataURL(file);
    });
    return {
        inlineData: { data: await base64EncodedDataPromise, mimeType: file.type },
    };
}

function updateProgress(percent, step) {
    elements.progressBar.style.width = `${percent}%`;
    document.querySelectorAll('.step').forEach(el => {
        if (el.dataset.step === step) el.classList.add('active');
        else el.classList.remove('active');
    });
}
