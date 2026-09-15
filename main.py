import io
import json
import os
import time
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from pypdf import PdfReader
import requests
from database import supabase

load_dotenv()

app = FastAPI(title="AI Knowledge Base API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set in environment variables")

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN is not set in environment variables")

gemini_client = genai.Client(api_key=gemini_api_key)

# Active Hugging Face Serverless Router Endpoint
HF_API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"
HF_HEADERS = {
    "Authorization": f"Bearer {hf_token}",
    "Content-Type": "application/json"
}

def get_embedding(text: str) -> List[float]:
    payload = {
        "inputs": text,
        "options": {"wait_for_model": True}
    }
    
    for _ in range(3):
        res = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=30)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                return data[0]
            if isinstance(data, list) and isinstance(data[0], (float, int)):
                return data
        elif res.status_code == 503:
            time.sleep(3)
            continue
        else:
            raise HTTPException(status_code=500, detail=f"HuggingFace embedding error: {res.text}")
            
    raise HTTPException(status_code=500, detail="HuggingFace model warmup timed out. Please try again.")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1800,
    chunk_overlap=300,
    separators=["\n\n", "\n", ". ", " ", ""]
)

class ChatTurn(BaseModel):
    role: str
    content: str

class KnowledgeItem(BaseModel):
    title: str
    content: str

class SearchQuery(BaseModel):
    query: str
    limit: int = 8

class AskQuery(BaseModel):
    question: str
    history: Optional[List[ChatTurn]] = []

@app.get("/")
def home():
    return {"message": "AI Knowledge Base API is operational"}

@app.post("/add-knowledge")
def add_knowledge(item: KnowledgeItem):
    try:
        text_to_embed = f"{item.title}: {item.content}"
        vector = get_embedding(text_to_embed)

        response = supabase.table("knowledge_base").insert({
            "title": item.title,
            "content": item.content,
            "embedding": vector
        }).execute()

        return {"status": "success", "data": response.data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    try:
        content_text = ""
        filename = file.filename

        if filename.endswith(".pdf"):
            pdf_bytes = await file.read()
            reader = PdfReader(io.BytesIO(pdf_bytes))
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    content_text += extracted + "\n"
        elif filename.endswith(".txt"):
            content_bytes = await file.read()
            content_text = content_bytes.decode("utf-8")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format.")

        if not content_text.strip():
            raise HTTPException(status_code=400, detail="The file contains no readable text.")

        chunks = text_splitter.split_text(content_text)
        clean_doc_name = filename.replace(".pdf", "").replace(".txt", "").replace("_", " ")

        rows_to_insert = []
        for index, chunk in enumerate(chunks):
            enriched_content = f"Document: {clean_doc_name}\nSection {index + 1}:\n{chunk}"
            vector = get_embedding(enriched_content)
            rows_to_insert.append({
                "title": f"{filename} (part {index + 1})",
                "content": enriched_content,
                "embedding": vector
            })

        supabase.table("knowledge_base").insert(rows_to_insert).execute()

        return {
            "status": "success",
            "filename": filename,
            "chunks_created": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/search")
def search_knowledge(search: SearchQuery):
    try:
        query_vector = get_embedding(search.query)

        response = supabase.rpc("match_knowledge", {
            "query_embedding": query_vector,
            "match_threshold": 0.10,
            "match_count": search.limit
        }).execute()

        return {"query": search.query, "results": response.data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/ask-stream")
def ask_ai_stream(request: AskQuery):
    try:
        query_vector = get_embedding(request.question)

        matched_chunks = supabase.rpc("match_knowledge", {
            "query_embedding": query_vector,
            "match_threshold": 0.10,
            "match_count": 8
        }).execute().data

        if not matched_chunks:
            context = "No relevant context found in the knowledge base."
        else:
            context = "\n\n".join([f"Source: {c['title']}\n{c['content']}" for c in matched_chunks])

        formatted_history = ""
        if request.history:
            recent_turns = request.history[-6:]
            formatted_history = "\n".join([f"{turn.role.capitalize()}: {turn.content}" for turn in recent_turns])

        prompt = f"""You are an intelligent portfolio and document assistant.
Answer the user's question accurately and thoroughly using the provided context and past conversation history.
Keep links, technology stacks, roles, and details exact.
If the information is not in the context or chat history, reply: "The knowledge base doesn't contain this information."

Relevant Document Context:
{context}

Previous Conversation:
{formatted_history if formatted_history else "No previous conversation."}

User Question: {request.question}
Answer:"""

        def token_generator():
            sources_payload = json.dumps({"sources": matched_chunks})
            yield f"{sources_payload}\n"

            response_stream = gemini_client.models.generate_content_stream(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text

        return StreamingResponse(token_generator(), media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/knowledge")
def get_all_knowledge():
    try:
        response = supabase.table("knowledge_base").select("id, title, content, created_at").execute()
        return {"data": response.data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))