
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types
import asyncio


# Avoid hardcoding sensitive identifiers in production.
# These should ideally be passed in dynamically or managed via a secure session service.
import os
APP_NAME="legal_document_analyser_agent"


root_agent = Agent(
    name="legal_document_agent",
    model="gemini-2.5-flash",
    description="Agent to extract and summarize legal documents.",
    instruction=""" System Role: You are a specialized legal document parser. Your objective is to identify any legal subpoena, summons, or court order and extract pertinent details into a structured format.

Task: Analyze the provided document. If it is a subpoena of any kind, extract the data using the schema below. If the document is not a subpoena or a similar legal request for information, return exactly {}.

Instructions:

1. Universal Detection: Identify the document type regardless of the issuing agency (e.g., Federal, State, Criminal, Civil, or Administrative).

2. Subtype Labeling: Use the subpoena_subtype field to describe the specific nature of the document example is it from IRS, medicaid, Adult protection service or something else.

3. Case Details: 
   - 'case_number': The unique identifier assigned by the court/agency.
   - 'date_from': The start date of the period for which records are being requested.
   - 'date_to': The deadline or 'due by' date for submitting the information.

4. Handling Missing Data: If a field is not found in the text, return null. Do not guess or hallucinate values.

5. Translation: Translate any non-English legal terms or entity names into English.

6. Output Format: If the user requests CSV, return the data in CSV format with headers. Otherwise, default to the JSON schema below.

7. Confidence & Notes: Provide a 'confidence_score' (0.0 to 1.0) based on text clarity and a 'notes' field to explain any ambiguities or missing critical fields.

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
    "date_to": "YYYY-MM-DD"
  }
}""",
    # Removed google_search for security: Principle of Least Privilege.
    # If search is needed, ensure queries are strictly validated.
    tools=[]
)


async def setup_session_and_runner(user_id, session_id):
    # Initialize the session service (using InMemory for this example)
    session_service = InMemorySessionService()
    
    # Create or retrieve the session
    session = await session_service.create_session(
        app_name=APP_NAME, 
        user_id=user_id, 
        session_id=session_id
    )
    
    # Initialize the runner with the root_agent
    runner = Runner(
        agent=root_agent, 
        app_name=APP_NAME, 
        session_service=session_service
    )
    return session, runner

async def call_agent_async(query, file_data=None, mime_type=None, user_id="anonymous", session_id="default"):
    # Prepare the message parts
    parts = [types.Part(text=query)]
    
    # Attach file if provided (e.g., PDF or PNG of a subpoena)
    if file_data and mime_type:
        parts.append(types.Part(inline_data=types.Blob(data=file_data, mime_type=mime_type)))
    
    content = types.Content(role='user', parts=parts)
    
    # Setup session and runner
    session, runner = await setup_session_and_runner(user_id, session_id)
    
    # Execute the agent run
    events = runner.run_async(
        user_id=user_id, 
        session_id=session_id, 
        new_message=content
    )

    async for event in events:
        # Check for the final structured response
        if event.is_final_response():
            final_response = event.content.parts[0].text
            return final_response

async def process_and_validate_document(file_bytes, mime_type):
    # Call the agent
    raw_response = await call_agent_async(
        "Extract details from this document.", 
        file_data=file_bytes, 
        mime_type=mime_type
    )
    
    import json
    data = json.loads(raw_response)
    
    # Validation Step
    metadata = data.get("extraction_metadata", {})
    confidence = metadata.get("confidence_score", 0.0)
    
    # Flag for human review if score < 0.7 or notes contain "unclear"
    data["needs_review"] = confidence < 0.7 or "unclear" in metadata.get("notes", "").lower()
    
    return data

if __name__ ==  "__main__":
    # Example: call_agent_async("extract this as csv", file_data="base64string", mime_type="application/pdf")
    asyncio.run(call_agent_async("what's the latest ai news?"))