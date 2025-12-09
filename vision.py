from dotenv import load_dotenv
import os
import asyncio
import logging

from livekit import agents
from livekit.agents import AgentSession
from livekit.agents.voice import Agent, RunContext
from livekit.agents.llm import function_tool
from livekit.rtc import VideoStream

from livekit.plugins import google

from langchain_qdrant import QdrantVectorStore
from langchain_mistralai import MistralAIEmbeddings


# -------------------- SETUP -------------------------

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("router-agent")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


AGENT_INSTRUCTIONS = """
You are a friendly, patient, and highly knowledgeable AI agent specialized in **router and network troubleshooting**.

Your ONLY responsibility is diagnosing and troubleshooting:
- Routers
- Modems
- Wi-Fi
- LED light issues (green, blinking, red, amber, etc.)
- Cabling problems
- No internet / slow internet
- Power issues
- Reset / reboot / config problems

------------------------------------------------
CRITICAL RULE (MUST FOLLOW)

For EVERY router-related issue:

1. If a camera is active → Call analyze_live_camera()
2. THEN → Always call search_policy()
3. ONLY answer using search_policy output.

DO NOT guess. DO NOT use general knowledge.

If "No relevant information found in the policy documents."

Return exactly:
"I apologize, but I couldn't find details on that specific topic in our official router knowledge base."

------------------------------------------------
Format:

1) Acknowledge issue
2) Short explanation
3) Step-by-step numbered solution
4) Optional tip
"""


SESSION_INSTRUCTIONS = """
Hi! 👋 I’m your Router Troubleshooting Assistant.

You can:
✅ Describe the issue
✅ Show me the router via camera
✅ Ask about slow or broken WiFi

I’ll analyze it and guide you step-by-step.
"""


# Global retriever
_retriever = None    



# ----------------------------------------------------
# Helper to convert Vision output into text query
# ----------------------------------------------------

def build_query_from_vision(vision_text: str) -> str:
    return f"Router visual diagnosis based on: {vision_text}"


# ----------------------------------------------------
# YOUR AGENT CLASS
# ----------------------------------------------------

class PolicyAgent(Agent):

    @function_tool
    async def analyze_live_camera(self, context: RunContext) -> str:
        """
        Captures a frame from user's live camera (if available) and describes it
        """

        try:
            participants = [
                p for p in context.room.participants.values()
                if p.identity != context.room.local_participant.identity
            ]

            if not participants:
                return "No camera feed detected."

            participant = participants[0]

            video_stream = await VideoStream.from_participant(
                participant, source="camera"
            )

            frame = None
            async for event in video_stream:
                if event.frame:
                    frame = event.frame
                    break

            if frame is None:
                return "No usable frame received from camera."

            image_bytes = frame.to_jpeg()

            vision_model = google.realtime.RealtimeModel(
                model="gemini-2.5-pro-vision",
                temperature=0
            )

            response = await vision_model.generate(
                input=[
                    """
                    Analyze this router image and return only:

                    - Brand / Model (if visible)
                    - Visible LED colors + status (blinking/solid)
                    - Cable status (connected / disconnected)
                    - Visible damage (yes/no)
                    - Confidence level (0–100)

                    DO NOT GUESS.
                    """,
                    image_bytes
                ]
            )

            vision_text = str(response)

            focused_query = build_query_from_vision(vision_text)

            return f"""
CAMERA_ANALYSIS:
{vision_text}

SUGGESTED_QUERY:
{focused_query}
"""

        except Exception as e:
            logger.exception("Camera analysis failed")
            return "Camera analysis failed."



    @function_tool
    async def search_policy(self, context: RunContext, query: str):
        """
        Search official router troubleshooting KnowlegeBase
        """

        if _retriever is None:
            return "Router troubleshooting database is currently unavailable."

        try:
            docs = await _retriever.aget_relevant_documents(query)
        except AttributeError:
            docs = await asyncio.to_thread(_retriever.invoke, query)
        except Exception as e:
            logger.exception("RAG Failed")
            return "An internal error prevented policy search."

        if not docs:
            return "No relevant information found in the policy documents."

        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"[Source {i}]\n{doc.page_content}")

        return "\n\n".join(context_parts)



# ----------------------------------------------------
# ENTRYPOINT
# ----------------------------------------------------

async def entrypoint(ctx: agents.JobContext):

    global _retriever

    logger.info("Initializing vector store...")

    embeddings = MistralAIEmbeddings(model="mistral-embed")

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name="porus2",
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        prefer_grpc=True,
    )

    _retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3}
    )


    logger.info("Starting AI model...")

    llm_model = google.realtime.RealtimeModel(
        model="gemini-2.5-flash-native-audio-preview-09-2025",
        voice="Puck",
        temperature=0,
        instructions=AGENT_INSTRUCTIONS,
    )

    agent = PolicyAgent(
        llm=llm_model,
        instructions=AGENT_INSTRUCTIONS,
    )

    session = AgentSession(llm=llm_model)

    await session.start(agent=agent, room=ctx.room)

    await session.generate_reply(instructions=SESSION_INSTRUCTIONS)


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint)
    )
