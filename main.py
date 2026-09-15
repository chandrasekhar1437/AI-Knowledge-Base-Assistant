import io
import json
import os
import time
from typing import List, Optional
import docx
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
    chunk_size=2000,
    chunk_overlap=400,
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
    limit: int = 20

class AskQuery(BaseModel):
    question: str
    history: Optional[List[ChatTurn]] = []

def fetch_active_models(api_key: str) -> List[str]:
    """Queries Google to get the exact models enabled for this specific key."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            items = r.json().get("models", [])
            valid = [
                m["name"].replace("models/", "")
                for m in items
                if "generateContent" in m.get("supportedGenerationMethods", [])
            ]
            print(f"Dynamically discovered models for key: {valid}")
            return valid
        else:
            print(f"ListModels call returned status {r.status_code}: {r.text}")
    except Exception as err:
        print(f"ListModels call failed: {err}")
    return ["gemini-1.5-flash", "gemini-1.5-pro"]

@app.get("/")
def home():
    models = fetch_active_models(gemini_api_key.strip())
    return {"message": "AI Knowledge Base Assistant API is active.", "active_models": models}

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
                    content_text += extracted + "\n\n"
        elif filename.endswith(".docx"):
            docx_bytes = await file.read()
            doc = docx.Document(io.BytesIO(docx_bytes))

            for element in doc.element.body:
                if element.tag.endswith("p"):
                    p_text = "".join(node.text for node in element.iter() if node.text and node.tag.endswith("t")).strip()
                    if p_text:
                        content_text += p_text + "\n\n"
                elif element.tag.endswith("tbl"):
                    for row in element.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tr"):
                        cells = []
                        for cell in row.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tc"):
                            cell_text = "".join(node.text for node in cell.iter() if node.text and node.tag.endswith("t")).strip().replace("\n", " ")
                            if cell_text:
                                cells.append(cell_text)

                        seen = []
                        for c in cells:
                            if not seen or c != seen[-1]:
                                seen.append(c)
                        if seen:
                            content_text += " | ".join(seen) + "\n\n"
        elif filename.endswith(".txt"):
            content_bytes = await file.read()
            content_text = content_bytes.decode("utf-8")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format.")

        if not content_text.strip():
            raise HTTPException(status_code=400, detail="The file contains no readable text.")

        chunks = text_splitter.split_text(content_text)
        clean_name = filename.replace(".pdf", "").replace(".docx", "").replace(".txt", "").replace("_", " ")

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
            "match_threshold": 0.0,
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
            "match_threshold": 0.0,
            "match_count": 20
        }).execute().data

        if not matched_chunks:
            context = "No relevant context found in the knowledge base."
        else:
            context = "\n\n".join([f"Source: {chunk['title']}\n{chunk['content']}" for chunk in matched_chunks])

        valid_turns = [
            turn for turn in (request.history or [])
            if "I don't find that information" not in turn.content
        ]
        recent_turns = valid_turns[-4:]
        formatted_history = "\n".join([f"{turn.role.capitalize()}: {turn.content}" for turn in recent_turns]) if recent_turns else "None."

        user_content = f"""Document Context:
{context}

Prior Conversation:
{formatted_history}

Question: {request.question}"""

        system_instruction = (
            "You are a professional documentation assistant. Answer the user's question clearly and accurately using ONLY the provided Document Context. "
            "Output ONLY the final answer formatted in clean Markdown bullets. Do not output instructions, constraints, or thinking steps. "
            "If the answer is not present in the Document Context, reply exactly: 'I don't find that information in the uploaded documents.'"
        )

        def token_generator():
            # Send sources payload immediately so the HTTP stream begins instantly
            sources_payload = json.dumps({"sources": matched_chunks or []})
            yield f"__SOURCES__{sources_payload}__ENDSOURCES__\n"

            api_key = gemini_api_key.strip()
            active_models = fetch_active_models(api_key)

            body = {
                "system_instruction": {
                    "parts": [{"text": system_instruction}]
                },
                "contents": [{"parts": [{"text": user_content}]}],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 1024
                }
            }
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": api_key
            }

            for model_name in active_models:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:streamGenerateContent?alt=sse&key={api_key}"
                try:
                    with requests.post(url, headers=headers, json=body, stream=True, timeout=(10, 45)) as resp:
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
                                                    yield txt
                                        except Exception:
                                            continue
                            return
                        else:
                            print(f"Model {model_name} returned status {resp.status_code}: {resp.text[:120]}")
                except Exception as exc:
                    print(f"Connection error with {model_name}: {exc}")
                    continue

            yield "I don't find that information in the uploaded documents."

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