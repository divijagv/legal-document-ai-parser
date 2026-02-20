
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

7. Output Format: If the user requests CSV, return the data in CSV format with headers. Otherwise, default to the JSON schema below.

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
    
    print(f"Starting agent run with query: {query}, file_data size: {len(file_data) if file_data else 0}, mime_type: {mime_type}")
    
    try:
        # Execute the agent run
        events = runner.run_async(
            user_id=user_id, 
            session_id=session_id, 
            new_message=content
        )

        final_response = None
        event_count = 0
        async for event in events:
            event_count += 1
            print(f"Event {event_count}: {type(event).__name__}")
            
            # Check for the final structured response
            if event.is_final_response():
                print(f"Found final response event")
                final_response = event.content.parts[0].text
                print(f"Final response text (first 200 chars): {final_response[:200] if final_response else 'None'}")
                return final_response
        
        # If we get here, no final response was found
        print(f"No final response found after {event_count} events")
        raise ValueError(f"Agent did not return a final response after processing {event_count} events. Please check your API key and ensure the Gemini API is accessible.")
    
    except Exception as e:
        error_msg = str(e).lower()
        
        # Check for specific error types and provide user-friendly messages
        if "api key" in error_msg or "invalid" in error_msg or "unauthenticated" in error_msg or "401" in error_msg:
            raise ValueError("❌ Invalid API Key: Please check that your Gemini API key is correct and active.")
        elif "quota" in error_msg or "429" in error_msg or "resource exhausted" in error_msg:
            raise ValueError("⚠️ Quota Exceeded: Your API quota limit has been reached. Please check your Google Cloud billing or wait for the quota to reset.")
        elif "permission" in error_msg or "403" in error_msg:
            raise ValueError("🔒 Permission Denied: Your API key doesn't have permission to access this resource. Please check your API key permissions in Google Cloud Console.")
        elif "not found" in error_msg or "404" in error_msg:
            raise ValueError("❓ Resource Not Found: The requested model or resource was not found. Please verify your configuration.")
        else:
            # Re-raise the original exception if it's not a known error type
            raise

async def process_and_validate_document(file_bytes, mime_type):
    # Call the agent
    raw_response = await call_agent_async(
        "Extract details from this document.", 
        file_data=file_bytes, 
        mime_type=mime_type
    )
    
    print(f"Raw agent response: {raw_response}")
    print(f"Response type: {type(raw_response)}")
    
    # Check if response is empty or None
    if not raw_response:
        raise ValueError("Agent returned an empty response. Please check your API key and try again.")
    
    import json
    import re
    
    # Strip markdown code blocks if present
    cleaned_response = raw_response.strip()
    
    # Remove ```json and ``` markers
    if cleaned_response.startswith('```'):
        # Remove opening ```json or ```
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response)
        # Remove closing ```
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        print(f"Stripped markdown code blocks. Cleaned response (first 200 chars): {cleaned_response[:200]}")
    
    try:
        data = json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON. Cleaned response was: {cleaned_response[:500]}")
        raise ValueError(f"Agent returned invalid JSON: {str(e)}. Response preview: {cleaned_response[:200]}")
    
    # Validation Step
    metadata = data.get("extraction_metadata", {})
    confidence = metadata.get("confidence_score", 0.0)
    
    # Flag for human review if score < 0.7 or notes contain "unclear"
    data["needs_review"] = confidence < 0.7 or "unclear" in metadata.get("notes", "").lower()
    
    return data

if __name__ ==  "__main__":
    # Example: call_agent_async("extract this as csv", file_data="base64string", mime_type="application/pdf")
    asyncio.run(call_agent_async("what's the latest ai news?"))