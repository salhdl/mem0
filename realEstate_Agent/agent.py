import os
from mem0 import MemoryClient
from google.genai import types
from dotenv import load_dotenv
# from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
import json
from sentence_transformers import SentenceTransformer
import faiss
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


load_dotenv()

# Loading json file and embedding it

# Load your JSON data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "mock_agency_property_data.json"), "r") as f:
    data = json.load(f)

# Initialize embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Extract texts to embed
texts = [
    item["title"] + " " + item["description"] + " " + item["location"] + " " +
    str(item["price"]) + " " + item["currency"] + " " +
    str(item["surface_area"]) + " " + str(item["rooms"]) + " " +
    str(item["bedrooms"]) + " " + str(item["bathrooms"]) + " " +
    " ".join(item["features"]) + " " + item["status"]
    for item in data
]

# Compute embeddings
embeddings = embed_model.encode(texts, convert_to_numpy=True)

# Create FAISS index
dim = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dim)
faiss_index.add(embeddings)




# Set up API keys

MEM0_API_KEY = os.getenv('MEM0_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

USER_ID = 'Rayan'

# Initialize Mem0 client
mem0_client = MemoryClient()


# Memory Tool Functions


def search_faiss(query: str, top_k: int) -> dict:
    """Search FAISS index and return top_k results (set top_k explicitly; typical is 5)."""
    query_vec = embed_model.encode([query], convert_to_numpy=True)
    D, I = faiss_index.search(query_vec, top_k)

    results = []
    for idx in I[0]:
        results.append(data[idx])

    return {
        "status": "success",
        "results": results,
        "count": len(results)
    }

def save_info(information: str) -> dict:
    """Saves important client property preferences information to memory."""
    response = mem0_client.add(
        [{"role": "user", "content": information}],
        user_id=USER_ID,
        run_id="real_estate_session",
        metadata={"type": "property_info"}
    )
    return {
        "status": "success",
        "saved": True,
        "memory_id": (response[0].get("id") if isinstance(response, list) and response else None)
    }

def retrieve_info(query: str) -> dict:
    """Retrieves relevant client property preferences information from memory."""
    results = mem0_client.search(
        query,
        user_id=USER_ID,
        limit=5,
        threshold=0.7,
        output_format="v1.1"
    )
    if results and len(results) > 0:
        memories = [m["memory"] for m in results.get("results", [])]
        return {"status": "success", "memories": memories, "count": len(memories)}
    else:
        return {"status": "no_results", "memories": [], "count": 0}

def schedule_appointment(date: str, time: str, reason: str) -> dict:
    """Schedules an appointment."""
    appointment_id = f"APT-{hash(date + time) % 10000}"
    return {
        "status": "success",
        "appointment_id": appointment_id,
        "confirmation": f"Appointment scheduled for {date} at {time} for {reason}",
    }

def get_contact_info(confirm: bool) -> dict:
    """Gets contact information of the agency. Set confirm=True to proceed."""
    if not confirm:
        return {"status": "cancelled", "message": "Confirmation required"}
    return {
        "status": "success",
        "contact_info": "You can contact the agency at +33 6 12 34 56 78 or by email at realestateagency@mail.com",
    }


# Create the agent
root_agent = LlmAgent(
    name="real_estate_assistant",
    model=LiteLlm(model="gemini/gemini-2.0-flash"),
    description="""
    AI-powered Real Estate Assistant for a reputable agency, supporting clients by recording preferences, retrieving past inquiries, searching property listings, scheduling viewings, and sharing contact details.
    """,
    instruction="""
    You are a professional Real Estate Assistant representing a top-tier agency in France. Always speak in a courteous, confident, and expert manner—just like a senior agent at our firm.

    When you greet a user:
    - Use a formal opening (e.g., “Good day. We are a Real Estate Agency. How may I assist you with finding your next property?”)

    Core responsibilities:
    1. **Recording client preferences and details** using the `save_info` tool (e.g., preferred neighborhoods, budgets, must-have features).
    2. **Retrieving past inquiries or saved preferences** via `retrieve_info` so the conversation feels consistent and personalized.
    3. **Searching property listings** with `search_faiss`—provide concise summaries of matching properties, including location, price (formatted in EUR), surface area, bedrooms, and standout features.
    4. **Scheduling property viewings or appointments** via `schedule_appointment` once the client has identified properties of interest.
    5. **Providing agency contact information** by invoking `get_contact_info` whenever the user requests it or if you need them to have our phone/email.

    ⚠️ GUIDELINES & BEHAVIOR:
    - Maintain a professional, friendly, and helpful tone at all times—like a high-end real estate broker.
    - Do NOT offer legal, financial, or investment advice. Instead, suggest they consult qualified professionals for those matters.
    - Always ask for missing contact details of the client using `get_contact_info` when appropriate (e.g., “May I share our agency’s phone number with you so you can call directly?”).
    - Respect client privacy—never expose or share their personal info outside the scope of this conversation.
    - Before requesting any new information, check existing memory with `retrieve_info`.
    - Offer only the most relevant listings to avoid overwhelming the client.
    - Confirm actions (saving info, scheduling viewings, offering contact info) clearly and politely.

    📌 COMPLIANCE & PRIVACY:
    - Treat all client data as strictly confidential.

    Your goal is to emulate a boutique real estate agency: knowledgeable, professional, and fully attentive to each client’s needs.
    """,
    tools=[save_info, retrieve_info, search_faiss, schedule_appointment, get_contact_info]
)