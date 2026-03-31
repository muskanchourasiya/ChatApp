from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Literal
import time
import os
from groq import Groq
from dotenv import load_dotenv
import logging

load_dotenv()
metrics = {
    "total_requests": 0,
    "total_errors": 0,
    "active_sessions": 0,
    "last_response_time": 0
}

conversation_store = {}

logging.basicConfig(
    filename="app.log",          
    filemode="a",               
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

logger = logging.getLogger(__name__)

app = FastAPI()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    session_id: str
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
    
def detect_prompt_injection(query: str):
    suspicious_patterns = [
        "ignore previous instructions",
        "forget your instructions",
        "system prompt",
        "act as",
        "jailbreak",
        "override",
        "bypass",
    ]

    query_lower = query.lower()

    for pattern in suspicious_patterns:
        if pattern in query_lower:
            return True

    return False

raw_text = load_data()
documents = chunk_text(raw_text)

def validate_input(query: str):
    if len(query.strip()) == 0:
        return False, "Empty query not allowed"

    if len(query) > 500:  
        return False, "Query too long"

    return True, ""

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

def detect_pii(text: str):
    patterns = [
        r"\b\d{10}\b", 
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        r"\b\d{12}\b", 
    ]

    for pattern in patterns:
        if re.search(pattern, text):
            return True

    return False

def check_retrieval_quality(chunks):
    if not chunks or len(chunks) == 0:
        return False

    total_length = sum(len(c) for c in chunks)

    if total_length < 50:
        return False

    return True

def check_grounding(answer: str, context: str):
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())

    overlap = answer_words & context_words

    return len(overlap) > 5

def stream_chat_response(session_id: str):
    messages = conversation_store[session_id]
    query = messages[-1]["content"]
    valid, msg = validate_input(query)
    if not valid:
        yield f"event: error\ndata: {msg}\n\n"
        return

    if detect_prompt_injection(query):
        yield "event: error\ndata: Unsafe query detected. Please rephrase.\n\n"
        return
    
    try:
        yield "event: status\ndata: Retrieving context...\n\n"

        retrieved_chunks = retrieve(query, k=5)

        if not check_retrieval_quality(retrieved_chunks):
            yield "event: status\ndata: No relevant context found\n\n"

            final_messages = [{
                "role": "system",
                "content": "Say you don't know if context is missing."
            }]

        logger.info("----- RETRIEVAL DEBUG START -----")
        logger.info(f"Query: {query}")
        
        for i, chunk in enumerate(retrieved_chunks):
            logger.info(f"{i+1}. {chunk[:100]}")

        logger.info("----- RETRIEVAL DEBUG END -----")

        use_rag = len(retrieved_chunks) > 0

        if use_rag:
            top_chunks = retrieved_chunks[:3]
            context = "\n\n".join(top_chunks)

            yield f"event: sources\ndata: {top_chunks}\n\n"

            final_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an enterprise AI assistant.\n"
                        "- Answer ONLY using provided context\n"
                        "- If unsure, say 'I don't know'\n"
                        "- NEVER simulate hacking, accessing systems, or retrieving hidden data\n"
                        "- NEVER generate passwords, secrets, or sensitive info\n"      
                    ),
                }
            ]

            final_messages.extend(messages[:-1])

            final_messages.append({
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{query}"
            })
        else:
            final_messages = messages

        yield "event: status\ndata: Generating response...\n\n"

        stream = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=final_messages,
            stream=True,
        )

        last_ping = time.time()
        full_response = ""
        
        for chunk in stream:
            if time.time() - last_ping > 2:
                yield "event: ping\ndata: keepalive\n\n"
                last_ping = time.time()
            
            delta = chunk.choices[0].delta
            if delta and delta.content:
                full_response += delta.content
                yield f"event: token\ndata: {delta.content}\n\n"

        if use_rag and not check_grounding(full_response, context):
            full_response = "I don't have enough information from the provided documents."
        
        if detect_pii(full_response):
            full_response = "Response blocked due to sensitive information."

        yield "event: done\ndata: completed\n\n"

        conversation_store[session_id].append({
            "role": "assistant",
            "content": full_response
        })

    except Exception as e:
        logger.error(f"Error during streaming: {str(e)}", exc_info=True)
        yield f"event: error\ndata: {str(e)}\n\n"

@app.get("/history/{session_id}")
def get_history(session_id: str):
    if session_id not in conversation_store:
        return []

    return conversation_store[session_id]

@app.post("/chat")
async def chat(request: ChatRequest):
    start_time = time.time()

    session_id = request.session_id

    if session_id not in conversation_store:
        conversation_store[session_id] = []
        metrics["active_sessions"] += 1

    metrics["total_requests"] += 1

    try:
        for msg in request.messages:
            if msg.role != "system":
                conversation_store[session_id].append(msg.dict())

        response = StreamingResponse(
            stream_chat_response(session_id),
            media_type="text/event-stream"
        )

        metrics["last_response_time"] = round(time.time() - start_time, 2)

        return response

    except Exception:
        metrics["total_errors"] += 1
        raise

@app.get("/health")
def health():
    return {
        "status": "ok",
        "uptime": "running"
    }

@app.get("/metrics")
def get_metrics():
    return metrics

@app.get("/logs")
def get_logs():
    try:
        with open("app.log", "r") as f:
            lines = f.readlines()[-20:]  
        return {"logs": lines}
    except:
        return {"logs": []}