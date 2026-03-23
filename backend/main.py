from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Literal
import time
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


def stream_chat_response(messages: List[Message]):
    try:
        yield "event: status\ndata: Analyzing...\n\n"
        time.sleep(0.5)

        yield "event: status\ndata: Thinking...\n\n"
        time.sleep(0.5)

        yield "event: status\ndata: Generating response...\n\n"

       
        messages_dict = [msg.dict() for msg in messages]

        stream = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages_dict,
            stream=True,
        )

        last_ping = time.time()

        for chunk in stream:
            if time.time() - last_ping > 2:
                yield "event: ping\ndata: keepalive\n\n"
                last_ping = time.time()

            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield f"event: token\ndata: {delta.content}\n\n"

        yield "event: done\ndata: completed\n\n"

    except Exception as e:
        yield f"event: error\ndata: {str(e)}\n\n"


@app.post("/chat")
async def chat(request: ChatRequest):
    return StreamingResponse(
        stream_chat_response(request.messages),
        media_type="text/event-stream"
    )