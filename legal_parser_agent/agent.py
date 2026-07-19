"""
Legal document extraction agent, built on Google's Agent Development Kit (ADK).

This is the local/CLI backend: it reads GOOGLE_API_KEY from the environment
(or a .env file) and runs a Gemini-powered agent against a document, returning
structured JSON. It's independent of the static web app (index.html/app.js),
which calls the Gemini API directly from the browser instead.

Usage (see main.py for the CLI wrapper):
    from legal_parser_agent.agent import process_and_validate_document
    data = await process_and_validate_document(file_bytes, "application/pdf")
"""

import asyncio
import json

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field

from legal_parser_agent.validation import clean_json_text, flag_needs_review

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv is optional; if it's not installed, GOOGLE_API_KEY must
    # already be set in the environment.
    pass

APP_NAME = "legal_document_analyser_agent"

EXTRACTION_INSTRUCTION = """System Role: You are a specialized legal document parser. Your objective is to identify any legal subpoena, summons, or court order and extract pertinent details into a structured format.

Task: Analyze the provided document. If it is a subpoena of any kind, extract the data using the schema below. If the document is not a subpoena or a similar legal request for information, set "is_subpoena" to false and leave the other fields null.

Instructions:

1. Universal Detection: Identify the document type regardless of the issuing agency (e.g., Federal, State, Criminal, Civil, or Administrative).

2. Subtype Labeling: Use the subpoena_subtype field to describe the specific nature of the document (e.g. IRS, Medicaid, Adult Protective Services, or something else).

3. Customer Details (Party to Whom Subpoena is Addressed):
   - 'name': The individual's name to whom the subpoena is addressed (if applicable).
   - 'company': The company/organization name to whom the subpoena is addressed (if applicable).
   - Note: This is the party being asked to provide information, NOT the requestor.
   - Extract all other customer identifiers (SSN, account numbers, DOB, etc.) if present.

4. Case Details:
   - 'case_number': The unique identifier assigned by the court/agency.
   - 'date_from': The start date of the period for which records are being requested.
   - 'date_to': The end date of the period for which records are being requested.
   - 'due_date': The deadline by which the documents/information must be submitted.

5. Handling Missing Data: If a field is not found in the text, return null. Do not guess or hallucinate values.

6. Translation: Translate any non-English legal terms or entity names into English.

7. Output Format: Return only JSON matching the schema below — the output is schema-enforced.

8. Confidence & Notes: Provide a 'confidence_score' (0.0 to 1.0) based on text clarity and a 'notes' field to explain any ambiguities or missing critical fields.

JSON Output Schema:

{
  "is_subpoena": true,
  "subpoena_subtype": "string",
  "customer_details": {
    "name": "string",
    "company": "string",
    "ssn": "string",
    "tax_id": "string",
    "bank_account_number": "string",
    "bank_account_type": "string",
    "dob": "string",
    "phone": "string",
    "email": "string"
  },
  "additional_customer_details": [
    {
      "name": "string",
      "details": "string"
    }
  ],
  "requestor_information": {
    "name": "string",
    "company": "string",
    "address": "string",
    "email": "string",
    "state_code": "string",
    "requestor_entity_type": "string"
  },
  "alternate_requestor": {
    "name": "string",
    "details": "string"
  },
  "case_details": {
    "case_number": "string",
    "date_from": "YYYY-MM-DD",
    "date_to": "YYYY-MM-DD",
    "due_date": "YYYY-MM-DD"
  },
  "extraction_metadata": {
    "confidence_score": 0.0,
    "notes": "string"
  },
  "document_summary": "A brief 2-sentence summary of the document and its purpose"
}"""

# Pydantic models mirroring the JSON schema in EXTRACTION_INSTRUCTION.
# Passed as output_schema so Gemini's structured-output mode guarantees the
# response is valid JSON in exactly this shape — no markdown fences, no
# missing keys, no free-text commentary.


class CustomerDetails(BaseModel):
    name: str | None = None
    company: str | None = None
    ssn: str | None = None
    tax_id: str | None = None
    bank_account_number: str | None = None
    bank_account_type: str | None = None
    dob: str | None = None
    phone: str | None = None
    email: str | None = None


class NamedDetail(BaseModel):
    name: str | None = None
    details: str | None = None


class RequestorInformation(BaseModel):
    name: str | None = None
    company: str | None = None
    address: str | None = None
    email: str | None = None
    state_code: str | None = None
    requestor_entity_type: str | None = None


class CaseDetails(BaseModel):
    case_number: str | None = None
    date_from: str | None = Field(None, description="YYYY-MM-DD")
    date_to: str | None = Field(None, description="YYYY-MM-DD")
    due_date: str | None = Field(None, description="YYYY-MM-DD")


class ExtractionMetadata(BaseModel):
    confidence_score: float = 0.0
    notes: str | None = None


class SubpoenaExtraction(BaseModel):
    is_subpoena: bool
    subpoena_subtype: str | None = None
    customer_details: CustomerDetails | None = None
    additional_customer_details: list[NamedDetail] | None = None
    requestor_information: RequestorInformation | None = None
    alternate_requestor: NamedDetail | None = None
    case_details: CaseDetails | None = None
    extraction_metadata: ExtractionMetadata
    document_summary: str | None = None


root_agent = Agent(
    name="legal_document_agent",
    model="gemini-2.5-flash",
    description="Agent to extract and summarize legal documents.",
    instruction=EXTRACTION_INSTRUCTION,
    # Schema-enforced structured output (requires tools=[], which is also
    # least-privilege: this agent only reads the document it's given).
    output_schema=SubpoenaExtraction,
    tools=[],
)


async def setup_session_and_runner(user_id, session_id):
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)
    return session, runner


async def call_agent_async(query, file_data=None, mime_type=None, user_id="anonymous", session_id="default"):
    """Run the agent once and return its final text response."""
    parts = [types.Part(text=query)]
    if file_data and mime_type:
        parts.append(types.Part(inline_data=types.Blob(data=file_data, mime_type=mime_type)))

    content = types.Content(role="user", parts=parts)
    session, runner = await setup_session_and_runner(user_id, session_id)

    try:
        events = runner.run_async(user_id=user_id, session_id=session_id, new_message=content)

        async for event in events:
            if event.is_final_response():
                return event.content.parts[0].text

        raise ValueError("The agent finished without returning a response. Check your API key and try again.")

    except Exception as e:
        msg = str(e).lower()
        if "api key" in msg or "unauthenticated" in msg or "401" in msg:
            raise ValueError("Invalid API key: check that GOOGLE_API_KEY is correct and active.") from e
        if "quota" in msg or "429" in msg or "resource exhausted" in msg:
            raise ValueError("Quota exceeded: your API quota limit has been reached.") from e
        if "permission" in msg or "403" in msg:
            raise ValueError("Permission denied: your API key doesn't have access to this model.") from e
        if "not found" in msg or "404" in msg:
            raise ValueError("Resource not found: verify the model name and configuration.") from e
        raise


async def process_and_validate_document(file_bytes, mime_type):
    """Extract structured data from a document and flag low-confidence results."""
    raw_response = await call_agent_async("Extract details from this document.", file_data=file_bytes, mime_type=mime_type)

    if not raw_response:
        raise ValueError("The agent returned an empty response. Check your API key and try again.")

    # Output is schema-enforced, but keep the fence-stripping as a cheap
    # belt-and-suspenders fallback.
    cleaned = clean_json_text(raw_response)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"The agent returned invalid JSON: {e}. Response preview: {cleaned[:200]}") from e

    data["needs_review"] = flag_needs_review(data)

    return data


if __name__ == "__main__":
    # Quick manual smoke test: `python -m legal_parser_agent.agent` with no file.
    asyncio.run(call_agent_async("Say hello and confirm you're ready to parse legal documents."))
