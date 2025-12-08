from dotenv import load_dotenv
import os
import asyncio
from livekit import agents
from livekit.agents import AgentSession
from livekit.agents.voice import Agent, RunContext
from livekit.agents.llm import function_tool
from livekit.plugins import tavus, google
# LangChain & Google RAG imports
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import JinaEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

AGENT_INSTRUCTIONS = """
You are a friendly, patient, and highly knowledgeable AI agent specialized in **router and network troubleshooting**.

Your sole purpose is to help users diagnose and fix issues related to:
- Routers
- Modems
- Wi-Fi networks
- Internet connectivity
- LAN/WAN issues
- Router lights (blinking, red, green, orange, etc.)
- Slow internet
- Disconnections
- Login / admin issues
- Firmware / reset / configuration problems

You must NOT answer questions unrelated to router or network troubleshooting.

------------------------------------------------
CRITICAL RULE (MUST FOLLOW)

For *every* question related to live router or network issues 
(example: 
"Why is the router’s internet light blinking green?", 
"My Wi-Fi is connected but no internet", 
"Router keeps restarting"
),

You **MUST ALWAYS** call the `search_policy` tool first.

❗ Do NOT:
- Guess
- Hallucinate
- Use general training data
- Provide advice without tool context

------------------------------------------------
TOOL USAGE INSTRUCTIONS

1. Call: search_policy(query: str)
   - The query MUST be the user’s exact, focused question.

2. Wait for the response.

3. When formulating your final answer:
   ✅ Use ONLY the tool response
   ✅ Be step-by-step
   ✅ Be simple and beginner-friendly
   ✅ Be empathetic and calm
   ✅ Use bullet points and numbered steps

4. If the tool response is:
   "No relevant information found..."

   Then reply exactly with:

   "I apologize, but I couldn't find details on that specific topic in our official router knowledge base."

------------------------------------------------
TONE & STYLE

✅ Friendly
✅ Clear
✅ Supportive
✅ Calm
✅ Professional technician-style guidance
✅ No jargon unless explained clearly

Example tone:
"I understand how frustrating that can be. Let’s fix it step by step together."

------------------------------------------------
RESPONSE STRUCTURE

1. Acknowledge the issue
2. Brief explanation (1-2 lines only)
3. Steps in numbered list
4. Optional prevention tip (if available)
"""



SESSION_INSTRUCTIONS = """
Hi there! 👋 I’m your Router Troubleshoot Assistant.

I can help you diagnose and fix issues with:
- Routers
- WiFi
- Internet connections
- Network errors
- Blinking lights
- Slow speeds

Just tell me the problem you’re facing (you can even send a photo of your router).
Let’s fix it together 🔧📡
"""



QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


# Global retriever (initialized in entrypoint)
_retriever = None

class PolicyAgent(Agent):
    """Custom Agent with RAG tool for policy search"""
    
    @function_tool
    async def search_policy(
        self,
        context: RunContext,
        query: str
    ):
        """
Search the official router troubleshooting knowledge base for accurate diagnostic steps
and solutions related to routers, modems, WiFi and internet connectivity issues.

Only official, verified troubleshooting content should be returned.
"""
        if _retriever is None:
            return "Router troubleshooting database is currently unavailable. No official information can be retrieved."
          
        try:
            # Use aget_relevant_documents first
            docs = await _retriever.aget_relevant_documents(query)
        except AttributeError:
            # Fallback for older retriever versions
            docs = await asyncio.to_thread(_retriever.invoke, query)
        except Exception as e:
            # CRITICAL: Catch all exceptions that could happen during vector search
            # This ensures the LLM gets a clear error message instead of an empty/failed tool call.
            print(f"RAG Retrieval Error: {e}")
            return "An internal error prevented the policy search from completing."
        
        if not docs:
            # This is correct for no results
            return "No relevant information found in the policy documents."
        
        # This formatting is perfect for the LLM context
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"[Source {i}]\n{doc.page_content}")
        
        return "\n\n".join(context_parts)


async def entrypoint(ctx: agents.JobContext):
    global _retriever
    
    # Initialize embeddings and vector store
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    embeddings = JinaEmbeddings(model='jina-embeddings-v2-base-en')
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name="router_manual",
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        prefer_grpc=True,
    )
    _retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})
    
    # Create Gemini Model
    llm_model = google.realtime.RealtimeModel(
        # model="gemini-2.0-flash-exp",
        model="gemini-2.5-flash-native-audio-preview-09-2025",
        voice="Puck",
        temperature=0,
        instructions=AGENT_INSTRUCTIONS,
    )
    
    # Create Custom Agent with RAG tool
    agent = PolicyAgent(
        llm=llm_model,
        instructions=AGENT_INSTRUCTIONS,
    )
    
    # Create Session
    session = AgentSession(llm=llm_model)
    
    # Create and start Tavus Avatar
    avatar = tavus.AvatarSession(
        replica_id=os.environ.get("REPLICA_ID"),
        persona_id=os.environ.get("PERSONA_ID"),
        api_key=os.environ.get("TAVUS_API_KEY"),
    )
    await avatar.start(session, room=ctx.room)
    
    # Start the agent session
    await session.start(agent=agent, room=ctx.room)
    
    # Send initial greeting
    await session.generate_reply(instructions=SESSION_INSTRUCTIONS)


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))