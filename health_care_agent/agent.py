import os
from mem0 import MemoryClient
from google.genai import types
from dotenv import load_dotenv
# from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService

load_dotenv()

# Set up API keys
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MEM0_API_KEY = os.getenv('MEM0_API_KEY')

USER_ID = 'Rayan'

# Initialize Mem0 client
mem0_client = MemoryClient()


# Memory Tool Functions

def save_patient_info(information: str) -> dict:
    """Saves important patient information to memory."""

    # Store in Mem0
    response = mem0_client.add(
        [{"role": "user", "content": information}],
        user_id=USER_ID,
        run_id="healthcare_session1",
        metadata={"type": "patient_info"}
    )


def retrieve_patient_info(query: str) -> dict:
    """Retrieves relevant patient information from memory."""

    # Search Mem0
    results = mem0_client.search(
        query,
        user_id=USER_ID,
        limit=5,
        threshold=0.7,  # Higher threshold for more relevant results
        output_format="v1.1"
    )

    # Format and return the results
    if results and len(results) > 0:
        memories = [memory["memory"] for memory in results.get('results', [])]
        return {
            "status": "success",
            "memories": memories,
            "count": len(memories)
        }
    else:
        return {
            "status": "no_results",
            "memories": [],
            "count": 0
        }
    

# For healthcare assistance    
def schedule_appointment(date: str, time: str, reason: str) -> dict:
    """Schedules a doctor's appointment."""
    # In a real app, this would connect to a scheduling system
    appointment_id = f"APT-{hash(date + time) % 10000}"

    return {
        "status": "success",
        "appointment_id": appointment_id,
        "confirmation": f"Appointment scheduled for {date} at {time} for {reason}",
        "message": "Please arrive 15 minutes early to complete paperwork."
    }


# Create the agent
root_agent = LlmAgent( # For custom LLM usage
    name="healthcare_assistant",
    model=LiteLlm(model="gemini/gemini-2.0-flash"), # LiteLLM model string format
    description="AI-powered Healthcare Assistant that supports patients by recording health information, retrieving relevant history, and scheduling appointments securely and efficiently.",
    instruction="""
    You are a trusted AI Healthcare Assistant designed to support patients with non-diagnostic services.

    Your core responsibilities include:
    1. **Recording key health-related information** shared by patients (e.g., symptoms, conditions, allergies, preferences) using the `save_patient_info` tool.
    2. **Retrieving previously shared health information** to ensure personalized and consistent support using the `retrieve_patient_info` tool.
    3. **Scheduling medical appointments** based on patient needs and preferences via the `schedule_appointment` tool.

    ⚠️ GUIDELINES & BEHAVIOR:
    - Always maintain a **professional, empathetic, and supportive tone**.
    - **Do NOT provide any medical diagnosis or treatment advice**. You are not a licensed medical professional.
    - Encourage patients to consult a qualified healthcare provider for serious or unclear symptoms.
    - Respect patient privacy at all times. **Do not reveal or repeat information unnecessarily.**
    - Before asking for information, check if it’s already available in memory.
    - Be concise and helpful when confirming actions (e.g., saving data or scheduling).

    📌 COMPLIANCE & PRIVACY:
    - All data should be handled in line with standard healthcare data privacy principles.
    - Treat all shared information as confidential and secure.

    Your goal is to make patients feel supported, informed, and guided — without replacing medical professionals.
    """,
    tools=[save_patient_info, retrieve_patient_info, schedule_appointment]
)