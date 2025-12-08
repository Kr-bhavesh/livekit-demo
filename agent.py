from dotenv import load_dotenv
import os
from livekit import agents
from livekit.agents import AgentSession, Agent
from livekit.plugins import tavus, google

# Optional: Your prompt files
load_dotenv()

AGENT_INSTRUCTIONS = """
You are a friendly and knowledgeable AI avatar that helps users through conversation.
Be concise, natural, and emotionally expressive in your tone. Must speak in english
"""

SESSION_INSTRUCTIONS = """
Start the conversation by greeting the user warmly and asking how you can help.
Keep your replies brief and human-like.
"""

async def entrypoint(ctx: agents.JobContext):
    # Create the Gemini Realtime Model
    llm = google.realtime.RealtimeModel(
        model="gemini-2.0-flash-exp",  # Use a Gemini Realtime model
        voice="Puck",                  # Choose any available voice
        temperature=0.8,
        instructions=AGENT_INSTRUCTIONS,  # Agent's behavior
    )
    
    # Create an Agent instance with the LLM
    agent = Agent(llm=llm,instructions=AGENT_INSTRUCTIONS)
    
    # Create a session
    session = AgentSession(llm=llm)
    
    # Tavus Avatar configuration
    avatar = tavus.AvatarSession(
        replica_id=os.environ.get("REPLICA_ID"),
        persona_id=os.environ.get("PERSONA_ID"),
        api_key=os.environ.get("TAVUS_API_KEY"),
    )
    
    # Start the avatar and link it with the session
    await avatar.start(session, room=ctx.room)
    
    # Start the LiveKit session for voice conversation
    await session.start(
        agent=agent,  # Pass the configured agent
        room=ctx.room,
        # room_input_options=RoomInputOptions(
        #     noise_cancellation=noise_cancellation.BVC(),
        # ),
    )
    # Send starting instructions to the model
    await session.generate_reply(instructions=SESSION_INSTRUCTIONS)

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))