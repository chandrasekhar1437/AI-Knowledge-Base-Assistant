import json
import os
import requests
import streamlit as st

st.set_page_config(
    page_title="AI Knowledge Base Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Backend Render API Base URL
API_URL = os.getenv("API_URL", "https://ai-knowledge-base-assistant-omgm.onrender.com")

# Custom CSS for dark-themed UI polish
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .stChatMessage {
        border-radius: 10px;
        margin-bottom: 0.8rem;
    }
    .source-box {
        background-color: #1e232f;
        border-left: 4px solid #4f8bf9;
        padding: 0.8rem 1rem;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State for Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR: Ingestion & Controls ---
with st.sidebar:
    st.title("📄 Ingestion Hub")

    # 1. Document Uploader (supports PDF, DOCX, TXT)
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader(
        "Upload a PDF, DOCX, or TXT file",
        type=["pdf", "docx", "txt"],
        help="Embeds your document directly into the vector database."
    )

    if st.button("Process & Ingest", use_container_width=True):
        if uploaded_file is not None:
            with st.spinner("Chunking, embedding, and storing in Supabase..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{API_URL}/upload-document", files=files, timeout=90)

                    if response.status_code == 200:
                        data = response.json()
                        chunks = data.get("chunks_created", 0)
                        st.success(f"Ingested {chunks} chunks from '{uploaded_file.name}'!")
                    else:
                        st.error(f"Upload failed: {response.text}")
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")
        else:
            st.warning("Please choose a file first.")

    st.markdown("---")

    # 2. Add Knowledge Manually
    st.subheader("✍️ Add Knowledge Manually")
    with st.form("manual_entry_form", clear_on_submit=True):
        entry_title = st.text_input("Title", placeholder="e.g. Availability & Preferences")
        entry_content = st.text_area("Content", placeholder="Enter specific facts, project details, or notes...", height=120)
        submit_manual = st.form_submit_button("Save Manual Entry", use_container_width=True)

        if submit_manual:
            if entry_title.strip() and entry_content.strip():
                with st.spinner("Embedding and storing entry..."):
                    try:
                        payload = {"title": entry_title.strip(), "content": entry_content.strip()}
                        res = requests.post(f"{API_URL}/add-knowledge", json=payload, timeout=30)
                        if res.status_code == 200:
                            st.success("Entry added successfully!")
                        else:
                            st.error(f"Failed to add entry: {res.text}")
                    except Exception as err:
                        st.error(f"Request failed: {str(err)}")
            else:
                st.warning("Both Title and Content are required.")

    st.markdown("---")

    # 3. Chat Control
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- MAIN PANEL: Chat Interface ---
st.title("🤖 AI Knowledge Base Assistant")
st.caption("Context-aware retrieval powered by Supabase pgvector & Gemini Flash")

# Render previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("View Retrieved Sources"):
                for idx, src in enumerate(msg["sources"]):
                    similarity = src.get("similarity", 0.0)
                    st.markdown(f"**{src.get('title', 'Unknown')}** (Similarity: `{similarity:.2f}`)")
                    st.caption(src.get("content", ""))
                    if idx < len(msg["sources"]) - 1:
                        st.divider()

# Input for new user question
if prompt := st.chat_input("Ask a question about your knowledge base..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history_payload = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        retrieved_sources = []

        try:
            req_body = {
                "question": prompt,
                "history": history_payload
            }

            with requests.post(f"{API_URL}/ask-stream", json=req_body, stream=True, timeout=60) as resp:
                if resp.status_code == 200:
                    stream_buffer = ""
                    sources_extracted = False

                    for raw_chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                        if raw_chunk:
                            stream_buffer += raw_chunk

                            # Extract metadata header wrapped in delimiter tokens
                            if not sources_extracted and "__SOURCES__" in stream_buffer:
                                if "__ENDSOURCES__\n" in stream_buffer:
                                    parts = stream_buffer.split("__ENDSOURCES__\n", 1)
                                    meta_str = parts[0].replace("__SOURCES__", "")
                                    try:
                                        meta = json.loads(meta_str)
                                        retrieved_sources = meta.get("sources", [])
                                    except Exception:
                                        pass
                                    full_response = parts[1]
                                    sources_extracted = True
                                    response_placeholder.markdown(full_response + "▌")
                                continue

                            if sources_extracted:
                                full_response += raw_chunk
                                response_placeholder.markdown(full_response + "▌")
                            else:
                                full_response = stream_buffer
                                response_placeholder.markdown(full_response + "▌")

                    response_placeholder.markdown(full_response)

                    if retrieved_sources:
                        with st.expander("View Retrieved Sources"):
                            for idx, src in enumerate(retrieved_sources):
                                similarity = src.get("similarity", 0.0)
                                st.markdown(f"**{src.get('title', 'Unknown')}** (Similarity: `{similarity:.2f}`)")
                                st.caption(src.get("content", ""))
                                if idx < len(retrieved_sources) - 1:
                                    st.divider()

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": retrieved_sources
                    })
                else:
                    err_msg = f"Server error ({resp.status_code}): {resp.text}"
                    response_placeholder.error(err_msg)
        except Exception as e:
            response_placeholder.error(f"Error fetching response: {str(e)}")