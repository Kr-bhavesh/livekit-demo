from dotenv import load_dotenv
import os
import asyncio
from livekit import agents
from livekit.agents import AgentSession
from livekit.agents.voice import Agent, RunContext
from livekit.agents.llm import function_tool
from livekit.plugins import tavus, google
# LangChain & Google RAG imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

AGENT_INSTRUCTIONS = """
You are a friendly and knowledgeable AI avatar and an expert on life insurance.
Your sole purpose is to answer user questions about **insurance plans, policies, coverage, benefits, and claims**.

**CRITICAL RULE:** For *every* question related to life insurance topics (e.g., "What is a term policy?", "How do I file a claim?"), 
you **MUST ALWAYS** call the `search_policy` tool first. Do not guess or use general knowledge.

**Instructions for Tool Use:**
1.  Call the `search_policy(query: str)` tool. The `query` must be the user's exact, focused question.
2.  Once the tool returns the context, use *only* that information to formulate a clear, empathetic, and conversational answer.
3.  If the tool returns "No relevant information found...", politely state: "I apologize, but I couldn't find details on that specific topic in our official knowledge base."

Maintain a natural, clear, and empathetic tone.
"""


SESSION_INSTRUCTIONS = """
Hi there! I’m your insurance assistant. 
I can help you understand various life insurance topics — such as coverage, benefits, claim procedures, and policy terms —
based on our official insurance knowledge base. 
What would you like to learn about today?
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
        Search the official life insurance knowledge base for detailed policy information, 
        coverage limits, claim procedures, and definitions. 
        Returns a string containing the retrieved policy context to be used for the answer.
        """
        if _retriever is None:
            # This check is good.
            return "Policy search is not available at the moment. The knowledge base is uninitialized."
        
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
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name="life_insurance_101",
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        prefer_grpc=True,
    )
    _retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})
    
    # Create Gemini Model
    llm_model = google.realtime.RealtimeModel(
        model="gemini-2.0-flash-exp",
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