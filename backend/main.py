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

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
    return chunks

def load_data():
    with open("data.txt", "r") as f:
        return f.read()

raw_text = load_data()
documents = chunk_text(raw_text)

import re

def retrieve(query, k=5):
    results = []

    def preprocess(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()

    query_words = set(preprocess(query))

    for doc in documents:
        doc_words = set(preprocess(doc))

        intersection = query_words & doc_words
        score = len(intersection)

        if score > 0:
            score += len(intersection) / len(query_words)

        results.append((doc, score))

    results.sort(key=lambda x: x[1], reverse=True)
    filtered = [doc for doc, score in results if score > 0]

    return filtered[:k]

def stream_chat_response(messages: List[Message]):
    try:
        yield "event: status\ndata: Retrieving context...\n\n"

        query = messages[-1].content
        retrieved_chunks = retrieve(query, k=5)

        print("\n--- RETRIEVAL DEBUG ---")
        print("Query:", query)
        for i, chunk in enumerate(retrieved_chunks):
            print(f"{i+1}. {chunk[:100]}")
        print("-----------------------\n")

        use_rag = len(retrieved_chunks) > 0

        if use_rag:
            top_chunks = retrieved_chunks[:3]
            context = "\n\n".join(top_chunks)

            yield f"event: sources\ndata: {top_chunks}\n\n"

            final_messages = [
                {"role": "system", "content": "You are a helpful AI assistant. Answer ONLY using the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"}
            ]
        else:
            final_messages = [msg.dict() for msg in messages]

        yield "event: status\ndata: Generating response...\n\n"

        stream = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=final_messages,
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