import io
import json
import os
import time
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
    raise ValueError("GEMINI_API_KEY is not set in environment variables.")

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN is not set in environment variables.")

# Hugging Face Feature Extraction (384 dimensions)
HF_EMBED_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5"

def get_embedding(text: str) -> List[float]:
    clean_text = text.replace("\r", " ").strip()
    headers = {
        "Authorization": f"Bearer {hf_token.strip()}",
        "Content-Type": "application/json",
        "X-Wait-For-Model": "true",
        "X-Use-Cache": "true"
    }
    payload = {"inputs": clean_text}

    last_err = ""
    for _ in range(5):
        try:
            res = requests.post(HF_EMBED_URL, headers=headers, json=payload, timeout=60)
            if res.status_code == 200:
                data = res.json()
                if isinstance(data, list) and len(data) > 0:
                    if isinstance(data[0], list):
                        return data[0]
                    if isinstance(data[0], (float, int)):
                        return data
            elif res.status_code in (503, 504):
                time.sleep(4)
                continue
            else:
                last_err = res.text
                time.sleep(2)
        except requests.RequestException as exc:
            last_err = str(exc)
            time.sleep(2)

    raise HTTPException(status_code=500, detail=f"HF Embedding error: {last_err}")

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
            raise HTTPException(status_code=400, detail="Unsupported file format.")

        if not content_text.strip():
            raise HTTPException(status_code=400, detail="The file contains no readable text.")

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
            "match_threshold": 0.05,
            "match_count": 6
        }).execute().data

        if not matched_chunks:
            context = "No relevant context found in the knowledge base."
        else:
            context = "\n\n".join([f"Source: {chunk['title']}\n{chunk['content']}" for chunk in matched_chunks])

        formatted_history = ""
        if request.history:
            recent_turns = request.history[-6:]
            formatted_history = "\n".join([f"{turn.role.capitalize()}: {turn.content}" for turn in recent_turns])

        prompt = f"""You are a helpful, professional AI assistant answering questions based on the provided documents.

CRITICAL OUTPUT GUIDELINES:
- Provide ONLY the direct, final response for the user.
- Do NOT output your thought process, meta-analysis, steps, or planning scratchpads.
- Respond with clean Markdown (bullet points, bold key terms, concise paragraphs).
- Retain exact technical stacks, project names, metrics, links, and dates accurately.
- If the question cannot be answered using the provided context, state clearly: "I don't find that information in the uploaded documents."

Document Context:
{context}

Conversation History:
{formatted_history if formatted_history else "No prior conversation."}

User Question: {request.question}
Final Answer:"""

        def token_generator():
            # Metadata JSON line consumed by Streamlit for source citations
            sources_payload = json.dumps({"sources": matched_chunks or []})
            yield f"{sources_payload}\n"

            # Dynamic model discovery prioritizing Flash variants
            candidate_models = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-flash", "gemini-pro"]
            try:
                list_res = requests.get(
                    f"https://generativelanguage.googleapis.com/v1beta/models?key={gemini_api_key}",
                    timeout=10
                )
                if list_res.status_code == 200:
                    found = [
                        m["name"].replace("models/", "")
                        for m in list_res.json().get("models", [])
                        if "generateContent" in m.get("supportedGenerationMethods", [])
                    ]
                    if found:
                        candidate_models = sorted(found, key=lambda x: 0 if "flash" in x.lower() else 1)
            except Exception:
                pass

            body = {
                "contents": [{"parts": [{"text": prompt}]}]
            }

            streamed_successfully = False
            last_err = ""

            for model_name in candidate_models:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:streamGenerateContent?alt=sse&key={gemini_api_key}"
                try:
                    with requests.post(url, json=body, stream=True, timeout=(10, 60)) as resp:
                        if resp.status_code == 200:
                            for line in resp.iter_lines():
                                if line:
                                    decoded = line.decode("utf-8")
                                    if decoded.startswith("data: "):
                                        try:
                                            chunk_data = json.loads(decoded[6:])
                                            parts = chunk_data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                                            for p in parts:
                                                txt = p.get("text", "")
                                                if txt:
                                                    streamed_successfully = True
                                                    yield txt
                                        except Exception:
                                            continue
                            if streamed_successfully:
                                return
                        else:
                            last_err = f"{model_name}: {resp.text}"
                except Exception as exc:
                    last_err = str(exc)
                    continue

            if not streamed_successfully:
                yield f"\n\n[Generation error: {last_err}]"

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