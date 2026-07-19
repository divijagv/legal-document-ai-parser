import { GoogleGenerativeAI } from "https://esm.run/@google/generative-ai";

/**
 * Legal Document AI Parser — client-side app.
 *
 * This app runs entirely in the browser. Your Gemini API key and documents
 * are sent directly from your browser to Google's Generative Language API —
 * they never pass through any server we control.
 */

const MODEL_NAME = "gemini-2.5-flash";
const MAX_FILE_SIZE_MB = 20;
const ACCEPTED_EXTENSIONS = [".pdf", ".png", ".jpg", ".jpeg", ".txt"];

const EXTRACTION_PROMPT = `System Role: You are a specialized legal document parser. Your objective is to identify any legal subpoena, summons, or court order and extract pertinent details into a structured format.

Task: Analyze the provided document. If it is a subpoena, summons, court order, or similar legal request for information, extract the data using the schema below. If the document is clearly not one of these, set "is_subpoena" to false and leave the other fields null.

Instructions:
1. Universal Detection: Identify the document type regardless of the issuing agency (e.g., Federal, State, Criminal, Civil, or Administrative).
2. Subtype Labeling: Use "subpoena_subtype" to describe the specific nature of the document (e.g., IRS, Medicaid, Adult Protective Services, civil litigation, etc.).
3. Customer Details (the party to whom the subpoena is addressed, NOT the requestor): name, company, ssn, tax_id, bank_account_number, bank_account_type, dob, phone, email.
4. Case Details: case_number, date_from (start of records period), date_to (end of records period), due_date (submission deadline).
5. Requestor Information: name, company, address, email, state_code, requestor_entity_type.
6. Handling Missing Data: if a field is not found, return null. Do not guess or hallucinate values.
7. Translation: translate any non-English legal terms or entity names into English.
8. Confidence & Notes: provide "confidence_score" (0.0-1.0) based on text clarity, and "notes" explaining any ambiguity or missing critical fields.
9. Provide a 2-sentence "document_summary" describing the document and its purpose.

Return ONLY valid JSON matching this exact schema, no markdown fences, no commentary:
{
  "is_subpoena": boolean,
  "subpoena_subtype": "string|null",
  "customer_details": {
    "name": "string|null", "company": "string|null", "ssn": "string|null",
    "tax_id": "string|null", "bank_account_number": "string|null",
    "bank_account_type": "string|null", "dob": "string|null",
    "phone": "string|null", "email": "string|null"
  },
  "requestor_information": {
    "name": "string|null", "company": "string|null", "address": "string|null",
    "email": "string|null", "state_code": "string|null", "requestor_entity_type": "string|null"
  },
  "case_details": {
    "case_number": "string|null", "date_from": "YYYY-MM-DD|null",
    "date_to": "YYYY-MM-DD|null", "due_date": "YYYY-MM-DD|null"
  },
  "extraction_metadata": { "confidence_score": 0.0, "notes": "string|null" },
  "document_summary": "string|null"
}`;

// Gemini structured-output schema mirroring EXTRACTION_PROMPT. With this set,
// the API guarantees valid JSON in exactly this shape — no fences, no prose.
const nullableString = { type: "string", nullable: true };
const EXTRACTION_SCHEMA = {
  type: "object",
  properties: {
    is_subpoena: { type: "boolean" },
    subpoena_subtype: nullableString,
    customer_details: {
      type: "object",
      nullable: true,
      properties: {
        name: nullableString, company: nullableString, ssn: nullableString,
        tax_id: nullableString, bank_account_number: nullableString,
        bank_account_type: nullableString, dob: nullableString,
        phone: nullableString, email: nullableString,
      },
    },
    requestor_information: {
      type: "object",
      nullable: true,
      properties: {
        name: nullableString, company: nullableString, address: nullableString,
        email: nullableString, state_code: nullableString, requestor_entity_type: nullableString,
      },
    },
    case_details: {
      type: "object",
      nullable: true,
      properties: {
        case_number: nullableString, date_from: nullableString,
        date_to: nullableString, due_date: nullableString,
      },
    },
    extraction_metadata: {
      type: "object",
      properties: {
        confidence_score: { type: "number" },
        notes: nullableString,
      },
      required: ["confidence_score"],
    },
    document_summary: nullableString,
  },
  required: ["is_subpoena", "extraction_metadata"],
};

const state = {
  apiKey: localStorage.getItem("gemini_api_key") || "",
  files: [],
  results: [], // { id, name, status: 'pending' | 'success' | 'error', data, error }
  isAnalyzing: false,
  theme: localStorage.getItem("theme") || "dark",
  expandedId: null,
};

const $ = (id) => document.getElementById(id);

const el = {
  apiKeyInput: $("api-key"),
  toggleKeyBtn: $("toggle-key"),
  dropZone: $("drop-zone"),
  fileInput: $("file-input"),
  fileInfo: $("file-info"),
  fileList: $("file-list"),
  analyzeBtn: $("analyze-btn"),
  statusSection: $("status-section"),
  statusText: $("status-text"),
  resultSection: $("result-section"),
  tableBody: document.querySelector("#results-table tbody"),
  copyBtn: $("copy-json"),
  downloadJsonBtn: $("download-json"),
  downloadCsvBtn: $("download-csv"),
  progressBar: document.querySelector(".loader-progress"),
  themeToggle: $("theme-toggle"),
  themeIcon: $("theme-icon"),
  themeText: $("theme-text"),
  chatSection: $("chat-section"),
  chatHistory: $("chat-history"),
  chatInput: $("chat-input"),
  sendChatBtn: $("send-chat"),
  chatFloatBtn: $("chat-float-btn"),
  errorSection: $("error-section"),
  errorTitle: $("error-title"),
  errorMessage: $("error-message"),
  retryBtn: $("retry-btn"),
};

init();

function init() {
  if (state.apiKey) el.apiKeyInput.value = state.apiKey;
  applyTheme(state.theme);
  bindEvents();
  updateAnalyzeButton();
}

function bindEvents() {
  el.apiKeyInput.addEventListener("input", (e) => {
    state.apiKey = e.target.value.trim();
    localStorage.setItem("gemini_api_key", state.apiKey);
    updateAnalyzeButton();
  });

  if (el.toggleKeyBtn) {
    el.toggleKeyBtn.addEventListener("click", () => {
      const isPassword = el.apiKeyInput.type === "password";
      el.apiKeyInput.type = isPassword ? "text" : "password";
      el.toggleKeyBtn.textContent = isPassword ? "🙈" : "👁";
    });
  }

  el.dropZone.addEventListener("click", () => el.fileInput.click());

  el.fileInput.addEventListener("change", (e) => {
    handleFileSelect(Array.from(e.target.files));
    el.fileInput.value = "";
  });

  ["dragenter", "dragover", "dragleave", "drop"].forEach((evt) => {
    el.dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
    });
  });
  el.dropZone.addEventListener("dragover", () => el.dropZone.classList.add("drag-over"));
  el.dropZone.addEventListener("dragleave", () => el.dropZone.classList.remove("drag-over"));
  el.dropZone.addEventListener("drop", (e) => {
    el.dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length) handleFileSelect(Array.from(e.dataTransfer.files));
  });

  el.analyzeBtn.addEventListener("click", () => analyzeDocuments());

  if (el.copyBtn) {
    el.copyBtn.addEventListener("click", () => {
      const successResults = state.results.filter((r) => r.status === "success");
      navigator.clipboard.writeText(JSON.stringify(successResults.map((r) => ({ name: r.name, ...r.data })), null, 2));
      flashButtonText(el.copyBtn, "Copied!");
    });
  }
  if (el.downloadJsonBtn) el.downloadJsonBtn.addEventListener("click", () => downloadData("json"));
  if (el.downloadCsvBtn) el.downloadCsvBtn.addEventListener("click", () => downloadData("csv"));

  el.themeToggle.addEventListener("click", () => {
    state.theme = state.theme === "dark" ? "light" : "dark";
    applyTheme(state.theme);
    localStorage.setItem("theme", state.theme);
  });

  el.sendChatBtn.addEventListener("click", () => handleChat());
  el.chatInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") handleChat();
  });

  if (el.chatFloatBtn) {
    el.chatFloatBtn.addEventListener("click", () => {
      el.chatSection.scrollIntoView({ behavior: "smooth" });
      el.chatInput.focus();
    });
  }

  if (el.retryBtn) {
    el.retryBtn.addEventListener("click", () => {
      hideError();
      if (state.files.length) analyzeDocuments();
    });
  }

  // Event delegation for row expand/collapse and per-file remove/download
  el.tableBody.addEventListener("click", (e) => {
    const revealBtn = e.target.closest("[data-reveal]");
    if (revealBtn) {
      e.stopPropagation();
      const span = revealBtn.previousElementSibling;
      const masked = span.dataset.masked === "true";
      span.textContent = masked ? span.dataset.full : span.dataset.mask;
      span.dataset.masked = masked ? "false" : "true";
      revealBtn.textContent = masked ? "Hide" : "Show";
      return;
    }
    const downloadLink = e.target.closest("[data-download]");
    if (downloadLink) {
      e.preventDefault();
      e.stopPropagation();
      downloadSingle(downloadLink.dataset.download);
      return;
    }
    const row = e.target.closest("tr[data-id]");
    if (row) toggleRowDetails(row.dataset.id);
  });
}

function applyTheme(theme) {
  document.body.classList.toggle("light-theme", theme === "light");
  el.themeIcon.textContent = theme === "light" ? "☀️" : "🌙";
  el.themeText.textContent = theme === "light" ? "Light Mode" : "Dark Mode";
}

function flashButtonText(btn, text) {
  const original = btn.textContent;
  btn.textContent = text;
  setTimeout(() => (btn.textContent = original), 1800);
}

// ---------- File handling ----------

function handleFileSelect(files) {
  const errors = [];
  for (const file of files) {
    const ext = "." + file.name.split(".").pop().toLowerCase();
    if (!ACCEPTED_EXTENSIONS.includes(ext)) {
      errors.push(`${file.name}: unsupported file type`);
      continue;
    }
    if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
      errors.push(`${file.name}: exceeds ${MAX_FILE_SIZE_MB}MB limit`);
      continue;
    }
    const isDuplicate = state.files.some((f) => f.name === file.name && f.size === file.size);
    if (!isDuplicate) state.files.push(file);
  }

  if (errors.length) {
    showError("Some files were skipped", errors.join("\n"));
  }

  renderFileList();
  updateAnalyzeButton();
}

function renderFileList() {
  if (!state.files.length) {
    el.fileInfo.classList.add("hidden");
    el.dropZone.classList.remove("hidden");
    return;
  }
  el.dropZone.classList.add("hidden");
  el.fileInfo.classList.remove("hidden");
  el.fileList.innerHTML = state.files
    .map(
      (f, i) => `
      <div class="file-chip">
        <span class="file-chip-name">${escapeHtml(f.name)}</span>
        <span class="file-chip-size">${formatBytes(f.size)}</span>
        <button type="button" class="file-chip-remove" data-index="${i}" title="Remove">✕</button>
      </div>`
    )
    .join("");

  el.fileList.querySelectorAll(".file-chip-remove").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.files.splice(Number(btn.dataset.index), 1);
      renderFileList();
      updateAnalyzeButton();
    });
  });
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function updateAnalyzeButton() {
  el.analyzeBtn.disabled = !state.apiKey || state.files.length === 0 || state.isAnalyzing;
  el.analyzeBtn.textContent = state.files.length > 1 ? `Analyze ${state.files.length} Documents` : "Analyze Document";
}

// ---------- Analysis ----------

async function analyzeDocuments() {
  if (state.isAnalyzing || !state.files.length) return;

  hideError();
  state.isAnalyzing = true;
  state.results = state.files.map((f, i) => ({ id: String(i), name: f.name, status: "pending", data: null, error: null }));
  updateAnalyzeButton();

  el.statusSection.classList.remove("hidden");
  el.resultSection.classList.remove("hidden");
  el.chatSection.classList.add("hidden");
  renderResults();

  const genAI = new GoogleGenerativeAI(state.apiKey);
  const model = genAI.getGenerativeModel({
    model: MODEL_NAME,
    generationConfig: { responseMimeType: "application/json", responseSchema: EXTRACTION_SCHEMA },
  });

  for (let i = 0; i < state.files.length; i++) {
    const file = state.files[i];
    updateProgress(Math.round((i / state.files.length) * 100), "parse");
    el.statusText.textContent = `Analyzing ${i + 1} of ${state.files.length}: ${file.name}`;

    try {
      const filePart = await fileToGenerativePart(file);
      const result = await model.generateContent([EXTRACTION_PROMPT, filePart]);
      const text = result.response.text();
      const data = parseModelJson(text);
      state.results[i] = { id: String(i), name: file.name, status: "success", data, error: null };
    } catch (err) {
      console.error(`Failed to analyze ${file.name}:`, err);
      state.results[i] = { id: String(i), name: file.name, status: "error", data: null, error: classifyError(err) };
    }
    renderResults();
  }

  updateProgress(100, "done");
  state.isAnalyzing = false;
  updateAnalyzeButton();

  const successCount = state.results.filter((r) => r.status === "success").length;
  const failCount = state.results.length - successCount;

  if (successCount > 0) {
    el.chatSection.classList.remove("hidden");
    let msg = `Finished analyzing ${successCount} document${successCount === 1 ? "" : "s"}.`;
    if (failCount) msg += ` ${failCount} failed — expand the row for details.`;
    addChatMessage("ai", msg);
    el.resultSection.scrollIntoView({ behavior: "smooth" });
  } else if (failCount) {
    showError("Analysis failed", state.results[0].error || "All documents failed to analyze. Check your API key and try again.");
  }

  setTimeout(() => el.statusSection.classList.add("hidden"), 2500);
}

function parseModelJson(rawText) {
  let cleaned = rawText.trim();
  cleaned = cleaned.replace(/^```(?:json)?\s*\n?/i, "").replace(/\n?```\s*$/, "");
  let data;
  try {
    data = JSON.parse(cleaned);
  } catch (e) {
    throw new Error("The AI returned a response that couldn't be parsed as JSON. Try again, or try a clearer scan of the document.");
  }
  const metadata = data.extraction_metadata || {};
  const confidence = typeof metadata.confidence_score === "number" ? metadata.confidence_score : 0;
  data.needs_review = data.is_subpoena !== false && (confidence < 0.7 || /unclear/i.test(metadata.notes || ""));
  return data;
}

function classifyError(err) {
  const msg = (err && err.message ? err.message : String(err)).toLowerCase();
  if (msg.includes("api key") || msg.includes("api_key_invalid") || msg.includes("401") || msg.includes("unauthenticated")) {
    return "Invalid API key. Double-check the key you entered, or generate a new one.";
  }
  if (msg.includes("quota") || msg.includes("429") || msg.includes("resource exhausted") || msg.includes("rate")) {
    return "Rate limit or quota exceeded on your Gemini API key. Wait a moment and try again.";
  }
  if (msg.includes("permission") || msg.includes("403")) {
    return "Your API key doesn't have permission for this model. Check its restrictions in Google AI Studio.";
  }
  if (msg.includes("failed to fetch") || msg.includes("networkerror") || msg.includes("network")) {
    return "Network error reaching Google's API. Check your connection and try again.";
  }
  if (msg.includes("couldn't be parsed")) return err.message;
  return err && err.message ? err.message : "Something went wrong analyzing this document.";
}

async function fileToGenerativePart(file) {
  const data = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result.split(",")[1]);
    reader.onerror = () => reject(new Error(`Failed to read ${file.name}`));
    reader.readAsDataURL(file);
  });
  return { inlineData: { data, mimeType: file.type || "application/octet-stream" } };
}

// ---------- Rendering ----------

function renderResults() {
  if (!state.results.length) {
    el.tableBody.innerHTML = "";
    return;
  }

  el.tableBody.innerHTML = state.results
    .map((res) => {
      const row = buildRowHtml(res);
      const detail = state.expandedId === res.id ? buildDetailRowHtml(res) : "";
      return row + detail;
    })
    .join("");
}

function buildRowHtml(res) {
  const name = escapeHtml(res.name);

  if (res.status === "pending") {
    return `<tr data-id="${res.id}">
      <td>${name}</td><td colspan="5"><span class="tag tag-pending">Analyzing…</span></td>
    </tr>`;
  }

  if (res.status === "error") {
    return `<tr data-id="${res.id}" class="clickable" title="Click for details">
      <td>${name}</td>
      <td colspan="4">${escapeHtml(res.error || "Failed")}</td>
      <td><span class="tag tag-error">Failed ⚠</span></td>
    </tr>`;
  }

  const data = res.data;
  const caseInfo = data.case_details || {};
  const customer = data.customer_details || {};
  const metadata = data.extraction_metadata || {};
  const confidence = typeof metadata.confidence_score === "number" ? `${Math.round(metadata.confidence_score * 100)}%` : "N/A";

  let customerLabel = "N/A";
  if (customer.company) {
    customerLabel = customer.company + (customer.name ? ` (${customer.name})` : "");
  } else if (customer.name) {
    customerLabel = customer.name;
  }

  let statusHtml;
  if (data.is_subpoena === false) {
    statusHtml = `<span class="tag tag-neutral" title="Not identified as a subpoena or legal request">Not Applicable</span>`;
  } else if (data.needs_review) {
    statusHtml = `<span class="tag tag-review" title="${escapeAttr(metadata.notes || "Low confidence extraction")}">Needs Review ⓘ</span>`;
  } else {
    statusHtml = `<span class="tag tag-verified" title="High-confidence extraction">Verified ✓</span>`;
  }

  return `<tr data-id="${res.id}" class="clickable" title="Click to expand details">
    <td>${name}</td>
    <td>${escapeHtml(data.subpoena_subtype || "Unknown")}</td>
    <td>${escapeHtml(customerLabel)}</td>
    <td>${escapeHtml(caseInfo.case_number || "N/A")}</td>
    <td>${escapeHtml(caseInfo.due_date || "N/A")}</td>
    <td>${confidence}</td>
    <td>${statusHtml}</td>
    <td>
      <a href="#" class="download-link" data-download="${res.id}" title="Download this result as JSON">📥</a>
    </td>
  </tr>`;
}

function buildDetailRowHtml(res) {
  if (res.status !== "success") return "";
  const data = res.data;
  const customer = data.customer_details || {};
  const requestor = data.requestor_information || {};
  const caseInfo = data.case_details || {};
  const metadata = data.extraction_metadata || {};

  const maskedField = (label, value) => {
    if (!value) return detailItem(label, "N/A");
    const masked = maskValue(value);
    return `<div class="detail-item">
      <span class="detail-label">${escapeHtml(label)}</span>
      <span class="detail-value">
        <span class="masked-value" data-masked="true" data-mask="${escapeAttr(masked)}" data-full="${escapeAttr(value)}">${escapeHtml(masked)}</span>
        <button type="button" class="reveal-btn" data-reveal>Show</button>
      </span>
    </div>`;
  };
  const detailItem = (label, value) =>
    `<div class="detail-item"><span class="detail-label">${escapeHtml(label)}</span><span class="detail-value">${escapeHtml(value || "N/A")}</span></div>`;

  return `<tr class="row-details-row"><td colspan="8">
    <div class="row-details">
      ${data.document_summary ? `<p>${escapeHtml(data.document_summary)}</p>` : ""}
      <h4>👤 Customer / Addressee</h4>
      <div class="detail-grid">
        ${detailItem("Name", customer.name)}
        ${detailItem("Company", customer.company)}
        ${maskedField("SSN", customer.ssn)}
        ${maskedField("Bank Account #", customer.bank_account_number)}
        ${detailItem("DOB", customer.dob)}
        ${detailItem("Phone", customer.phone)}
        ${detailItem("Email", customer.email)}
      </div>
      <h4>📨 Requestor</h4>
      <div class="detail-grid">
        ${detailItem("Name", requestor.name)}
        ${detailItem("Company", requestor.company)}
        ${detailItem("Email", requestor.email)}
        ${detailItem("Address", requestor.address)}
        ${detailItem("State", requestor.state_code)}
        ${detailItem("Entity Type", requestor.requestor_entity_type)}
      </div>
      <h4>📅 Case Details</h4>
      <div class="detail-grid">
        ${detailItem("Case Number", caseInfo.case_number)}
        ${detailItem("Records From", caseInfo.date_from)}
        ${detailItem("Records To", caseInfo.date_to)}
        ${detailItem("Due Date", caseInfo.due_date)}
      </div>
      ${metadata.notes ? `<h4>📝 Notes</h4><p>${escapeHtml(metadata.notes)}</p>` : ""}
    </div>
  </td></tr>`;
}

function maskValue(value) {
  const str = String(value);
  if (str.length <= 4) return "•".repeat(str.length);
  return "•".repeat(Math.max(0, str.length - 4)) + str.slice(-4);
}

function toggleRowDetails(id) {
  state.expandedId = state.expandedId === id ? null : id;
  renderResults();
}

function downloadSingle(id) {
  const result = state.results.find((r) => r.id === id);
  if (!result || result.status !== "success") return;
  triggerDownload(`${result.name}_analysis.json`, JSON.stringify(result.data, null, 2), "application/json");
}

function downloadData(format) {
  const successResults = state.results.filter((r) => r.status === "success");
  if (!successResults.length) {
    showError("Nothing to download", "Analyze at least one document successfully first.");
    return;
  }
  if (format === "json") {
    triggerDownload("legal_analysis.json", JSON.stringify(successResults.map((r) => ({ name: r.name, ...r.data })), null, 2), "application/json");
  } else {
    triggerDownload("legal_analysis.csv", convertToCSV(successResults), "text/csv");
  }
}

function convertToCSV(results) {
  const headers = ["File", "Type", "Subtype", "Customer", "SSN", "Bank Account", "DOB", "Case Number", "Due Date", "Confidence", "Status"];
  const rows = results.map((r) => {
    const d = r.data;
    const customer = d.customer_details || {};
    const caseInfo = d.case_details || {};
    const metadata = d.extraction_metadata || {};
    return [
      r.name,
      d.is_subpoena ? "Subpoena" : "Other",
      d.subpoena_subtype || "",
      customer.name || customer.company || "",
      customer.ssn || "",
      customer.bank_account_number || "",
      customer.dob || "",
      caseInfo.case_number || "",
      caseInfo.due_date || "",
      metadata.confidence_score ?? "",
      d.needs_review ? "Needs Review" : "Verified",
    ].map(csvEscape);
  });
  return [headers.map(csvEscape), ...rows].map((row) => row.join(",")).join("\n");
}

function csvEscape(value) {
  const str = String(value ?? "");
  if (/[",\n]/.test(str)) return `"${str.replace(/"/g, '""')}"`;
  return str;
}

function triggerDownload(filename, content, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ---------- Chat ----------

async function handleChat() {
  const query = el.chatInput.value.trim();
  if (!query || state.isAnalyzing) return;

  const successResults = state.results.filter((r) => r.status === "success");
  if (!successResults.length) {
    addChatMessage("ai", "Analyze a document first — then I can answer questions about it.");
    return;
  }
  if (!state.apiKey) {
    addChatMessage("ai", "Add your Gemini API key above first.");
    return;
  }

  addChatMessage("user", query);
  el.chatInput.value = "";
  el.sendChatBtn.disabled = true;

  const context = JSON.stringify(successResults.map((r) => ({ name: r.name, ...r.data })));
  const prompt = `You are a legal document assistant. Based on these parsed results: ${context}\n\nAnswer the following question concisely and professionally: ${query}`;

  try {
    const genAI = new GoogleGenerativeAI(state.apiKey);
    const model = genAI.getGenerativeModel({ model: MODEL_NAME });
    const result = await model.generateContent(prompt);
    addChatMessage("ai", result.response.text());
  } catch (err) {
    addChatMessage("ai", `Sorry, I hit an error: ${classifyError(err)}`);
  } finally {
    el.sendChatBtn.disabled = false;
  }
}

function addChatMessage(role, text) {
  const msg = document.createElement("div");
  msg.className = `chat-message ${role}`;
  msg.innerHTML = escapeHtml(text).replace(/\n/g, "<br>");
  el.chatHistory.appendChild(msg);
  el.chatHistory.scrollTop = el.chatHistory.scrollHeight;
}

// ---------- Errors & progress ----------

function showError(title, message) {
  if (!el.errorSection) return;
  el.errorTitle.textContent = title;
  el.errorMessage.textContent = message;
  el.errorSection.classList.remove("hidden");
}

function hideError() {
  if (el.errorSection) el.errorSection.classList.add("hidden");
}

function updateProgress(percent, step) {
  el.progressBar.style.width = `${percent}%`;
  document.querySelectorAll(".step").forEach((s) => s.classList.toggle("active", s.dataset.step === step));
}

// ---------- Utils ----------

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str == null ? "" : String(str);
  return div.innerHTML;
}

function escapeAttr(str) {
  return escapeHtml(str).replace(/"/g, "&quot;");
}
