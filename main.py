import io
import json
import os
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

app = FastAPI(title="AI Knowledge Base Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not configured in environment variables.")

gemini_client = genai.Client(api_key=gemini_api_key)

# Direct v1 embedding call (768 dimensions, universally enabled across all Google AI Studio keys)
def get_embedding(text: str) -> List[float]:
    url = f"https://generativelanguage.googleapis.com/v1/models/embedding-001:embedContent?key={gemini_api_key}"
    payload = {
        "content": {
            "parts": [{"text": text}]
        }
    }
    
    resp = requests.post(url, json=payload, timeout=30)
    
    if resp.status_code == 200:
        data = resp.json()
        return data["embedding"]["values"]
        
    # Fallback to v1beta text-embedding-004 if available
    fallback_url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={gemini_api_key}"
    fb_resp = requests.post(fallback_url, json=payload, timeout=30)
    if fb_resp.status_code == 200:
        data = fb_resp.json()
        return data["embedding"]["values"]

    raise HTTPException(status_code=500, detail=f"Embedding API error: {resp.text}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=250,
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
    return {"message": "AI Knowledge Base Assistant API is active."}

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
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

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
            raise HTTPException(status_code=400, detail="Unsupported file format. Provide a PDF or TXT file.")

        if not content_text.strip():
            raise HTTPException(status_code=400, detail="The provided document contains no parseable text.")

        chunks = text_splitter.split_text(content_text)
        clean_name = filename.replace(".pdf", "").replace(".txt", "").replace("_", " ")

        rows_to_insert = []
        for idx, chunk in enumerate(chunks):
            enriched_content = f"Document: {clean_name}\nSection {idx + 1}:\n{chunk}"
            vector = get_embedding(enriched_content)
            rows_to_insert.append({
                "title": f"{filename} (part {idx + 1})",
                "content": enriched_content,
                "embedding": vector
            })

        supabase.table("knowledge_base").insert(rows_to_insert).execute()

        return {
            "status": "success",
            "filename": filename,
            "chunks_created": len(chunks)
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

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
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

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
            context = "\n\n".join([f"Source: {chunk['title']}\n{chunk['content']}" for chunk in matched_chunks])

        formatted_history = ""
        if request.history:
            recent_turns = request.history[-6:]
            formatted_history = "\n".join([f"{turn.role.capitalize()}: {turn.content}" for turn in recent_turns])

        prompt = f"""You are an intelligent document and knowledge base assistant.
Answer the user's question clearly, thoroughly, and factually using the relevant document context and chat history below.
Preserve exact dates, skills, links, tools, and technical specifications.
If the answer is completely absent from the context and chat history, say: "The knowledge base doesn't contain this information."

Relevant Document Context:
{context}

Chat History:
{formatted_history if formatted_history else "No prior conversation."}

Question: {request.question}
Answer:"""

        def token_generator():
            sources_payload = json.dumps({"sources": matched_chunks})
            yield f"{sources_payload}\n"

            stream = gemini_client.models.generate_content_stream(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            for chunk in stream:
                if chunk.text:
                    yield chunk.text

        return StreamingResponse(token_generator(), media_type="text/plain")

    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.get("/knowledge")
def get_all_knowledge():
    try:
        response = supabase.table("knowledge_base").select("id, title, content, created_at").execute()
        return {"data": response.data}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))