"""
## Documentation
Gemini Live API with RAG Integration for Router Troubleshooting
Based on: https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get_started_LiveAPI.py

## Setup

To install the dependencies for this script, run:

```
pip install google-genai opencv-python pyaudio pillow mss langchain langchain-community qdrant-client langchain-google-genai
```

Set environment variables:
- GEMINI_API_KEY: Your Gemini API key
- QDRANT_URL: Your Qdrant instance URL (e.g., "http://localhost:6333")
- QDRANT_API_KEY: Your Qdrant API key (optional, for cloud instances)
"""

import os
import asyncio
import base64
import io
import traceback

import cv2
import pyaudio
import PIL.Image
import mss

import argparse

from google import genai
from google.genai import types
from dotenv import load_dotenv

# RAG imports
from langchain_community.vectorstores import Qdrant

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.5-flash-native-audio-preview-09-2025"

DEFAULT_MODE = "camera"

# RAG Configuration
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY",)
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "porus2")
RAG_TOP_K = 3  # Number of relevant documents to retrieve

client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=os.environ.get("GEMINI_API_KEY"),
)


class RAGRetriever:
    """Handles RAG retrieval from Qdrant vector store"""
    
    def __init__(self):
        self.embeddings =  MistralAIEmbeddings(model="mistral-embed")
        
        self.qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
        
        self.vectorstore = Qdrant(
            client=self.qdrant_client,
            collection_name=QDRANT_COLLECTION,
            embeddings=self.embeddings,
        )
        
        print(f"✓ Connected to Qdrant at {QDRANT_URL}")
        print(f"✓ Using collection: {QDRANT_COLLECTION}")
    
    async def retrieve(self, query: str, k: int = RAG_TOP_K):
        """Retrieve relevant documents from vector store"""
        try:
            docs = await asyncio.to_thread(
                self.vectorstore.similarity_search,
                query,
                k=k
            )
            return docs
        except Exception as e:
            print(f"RAG retrieval error: {e}")
            return []
    
    def format_context(self, docs):
        """Format retrieved documents into context string"""
        if not docs:
            return ""
        
        context_parts = ["[KNOWLEDGE BASE CONTEXT]"]
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"\n--- Document {i} ---")
            context_parts.append(doc.page_content)
            if doc.metadata:
                context_parts.append(f"Metadata: {doc.metadata}")
        
        context_parts.append("\n[END KNOWLEDGE BASE CONTEXT]\n")
        return "\n".join(context_parts)


CONFIG = types.LiveConnectConfig(
    response_modalities=[
        "AUDIO",
    ],
    media_resolution="MEDIA_RESOLUTION_MEDIUM",
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
        )
    ),
    system_instruction="""You are an expert router troubleshooting assistant. 
    You have access to a knowledge base of router troubleshooting information.
    When answering questions, use the provided knowledge base context to give accurate, 
    specific troubleshooting steps. Always cite which information comes from the knowledge base 
    when relevant. If the knowledge base doesn't contain relevant information, 
    use your general knowledge but make that clear to the user.""",
    context_window_compression=types.ContextWindowCompressionConfig(
        trigger_tokens=25600,
        sliding_window=types.SlidingWindow(target_tokens=12800),
    ),
)

pya = pyaudio.PyAudio()


class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE, enable_rag=True):
        self.video_mode = video_mode
        self.enable_rag = enable_rag

        self.audio_in_queue = None
        self.out_queue = None

        self.session = None
        self.rag_retriever = None

        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None
        
        # Initialize RAG if enabled
        if self.enable_rag:
            try:
                self.rag_retriever = RAGRetriever()
            except Exception as e:
                print(f"⚠ Warning: Could not initialize RAG: {e}")
                print("⚠ Continuing without RAG support...")
                self.enable_rag = False

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            if text.lower() == "q":
                break
            
            # RAG-enhanced message sending
            if self.enable_rag and text and text.strip():
                print("🔍 Retrieving relevant information from knowledge base...")
                
                # Retrieve relevant documents
                docs = await self.rag_retriever.retrieve(text)
                
                if docs:
                    print(f"✓ Found {len(docs)} relevant documents")
                    context = self.rag_retriever.format_context(docs)
                    
                    # Send context first
                    await self.session.send(
                        input=f"{context}\n\nUser Question: {text}",
                        end_of_turn=True
                    )
                else:
                    print("ℹ No relevant documents found in knowledge base")
                    await self.session.send(input=text or ".", end_of_turn=True)
            else:
                await self.session.send(input=text or ".", end_of_turn=True)

    def _get_frame(self, cap):
        # Read the frame
        ret, frame = cap.read()
        # Check if the frame was read successfully
        if not ret:
            return None
        # Fix: Convert BGR to RGB color space
        # OpenCV captures in BGR but PIL expects RGB format
        # This prevents the blue tint in the video feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)  # Now using RGB frame
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        # This takes about a second, and will block the whole program
        # causing the audio pipeline to overflow if you don't to_thread it.
        cap = await asyncio.to_thread(
            cv2.VideoCapture, 0
        )  # 0 represents the default camera

        while True:
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            await self.out_queue.put(frame)

        # Release the VideoCapture object
        cap.release()

    def _get_screen(self):
        sct = mss.mss()
        monitor = sct.monitors[0]

        i = sct.grab(monitor)

        mime_type = "image/jpeg"
        image_bytes = mss.tools.to_png(i.rgb, i.size)
        img = PIL.Image.open(io.BytesIO(image_bytes))

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_screen(self):

        while True:
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            await self.out_queue.put(frame)

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        while True:
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    continue
                if text := response.text:
                    print(text, end="")

            # If you interrupt the model, it sends a turn_complete.
            # For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded
            # much more audio than has played yet.
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        try:
            async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
                async with asyncio.TaskGroup() as tg:
                    self.session = session

                    self.audio_in_queue = asyncio.Queue()
                    self.out_queue = asyncio.Queue(maxsize=5)

                    send_text_task = tg.create_task(self.send_text())
                    tg.create_task(self.send_realtime())
                    tg.create_task(self.listen_audio())
                    if self.video_mode == "camera":
                        tg.create_task(self.get_frames())
                    elif self.video_mode == "screen":
                        tg.create_task(self.get_screen())

                    tg.create_task(self.receive_audio())
                    tg.create_task(self.play_audio())

                    await send_text_task
                    raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            self.audio_stream.close()
            traceback.print_exception(EG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="pixels to stream from",
        choices=["camera", "screen", "none"],
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG retrieval",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("🤖 Gemini Live API with RAG for Router Troubleshooting")
    print("=" * 60)
    
    main = AudioLoop(video_mode=args.mode, enable_rag=not args.no_rag)
    asyncio.run(main.run())