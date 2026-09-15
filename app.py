import json
import os
import requests
import streamlit as st

st.set_page_config(
    page_title="AI Knowledge Base Assistant",
    page_icon="🧠",
    layout="wide"
)

# Fetch backend URL from Streamlit secrets or default to local/render
BACKEND_URL = os.getenv("BACKEND_URL", "https://ai-knowledge-base-assistant-omgm.onrender.com")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar - Ingestion Hub
with st.sidebar:
    st.title("📄 Ingestion Hub")
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        if st.button("Process & Ingest", use_container_width=True):
            with st.spinner("Processing and vectorizing document..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    res = requests.post(f"{BACKEND_URL}/upload-document", files=files, timeout=120)
                    if res.status_code == 200:
                        data = res.json()
                        st.success(f"Ingested {data.get('chunks_created', 0)} chunks from '{uploaded_file.name}'!")
                    else:
                        st.error(f"Error {res.status_code}: {res.text}")
                except Exception as e:
                    st.error(f"Connection failed: {e}")

    st.markdown("---")
    st.subheader("✍️ Add Knowledge Manually")
    manual_title = st.text_input("Title", placeholder="e.g. Availability & Preferences")
    manual_content = st.text_area("Content", placeholder="Enter specific facts, project details, or notes...")

    if st.button("Save Knowledge", use_container_width=True):
        if manual_title.strip() and manual_content.strip():
            with st.spinner("Saving to knowledge base..."):
                try:
                    payload = {"title": manual_title.strip(), "content": manual_content.strip()}
                    res = requests.post(f"{BACKEND_URL}/add-knowledge", json=payload, timeout=60)
                    if res.status_code == 200:
                        st.success("Knowledge successfully saved!")
                    else:
                        st.error(f"Error {res.status_code}: {res.text}")
                except Exception as e:
                    st.error(f"Failed to save: {e}")
        else:
            st.warning("Please fill out both the title and content fields.")

    st.markdown("---")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main Chat Interface
st.title("🧠 AI Knowledge Base Assistant")
st.caption("Context-aware retrieval powered by Supabase pgvector & Gemini Flash")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("View Retrieved Sources"):
                for idx, src in enumerate(msg["sources"]):
                    st.markdown(f"**Source {idx + 1}: {src.get('title', 'Unknown')}**")
                    st.text(src.get("content", "")[:300] + "...")

# Chat Input & Response Handling
prompt = st.chat_input("Ask a question about your knowledge base...")

if prompt:
    # Display user turn immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        retrieved_sources = []

        # Prepare history for context
        chat_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]

        payload = {
            "question": prompt,
            "history": chat_history
        }

        try:
            # Extended connect and read timeouts (15s connect, 180s read)
            with requests.post(
                f"{BACKEND_URL}/ask-stream",
                json=payload,
                stream=True,
                timeout=(15, 180)
            ) as response:
                if response.status_code == 200:
                    raw_buffer = ""
                    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            raw_buffer += chunk

                            # Extract sources metadata header if present
                            if "__SOURCES__" in raw_buffer and "__ENDSOURCES__\n" in raw_buffer:
                                start = raw_buffer.find("__SOURCES__") + len("__SOURCES__")
                                end = raw_buffer.find("__ENDSOURCES__\n")
                                try:
                                    sources_data = json.loads(raw_buffer[start:end])
                                    retrieved_sources = sources_data.get("sources", [])
                                except Exception:
                                    pass
                                raw_buffer = raw_buffer[end + len("__ENDSOURCES__\n"):]

                            full_response = raw_buffer
                            response_placeholder.markdown(full_response + "▌")

                    response_placeholder.markdown(full_response)

                    if retrieved_sources:
                        with st.expander("View Retrieved Sources"):
                            for idx, src in enumerate(retrieved_sources):
                                st.markdown(f"**Source {idx + 1}: {src.get('title', 'Unknown')}**")
                                st.text(src.get("content", "")[:300] + "...")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": retrieved_sources
                    })
                else:
                    err_msg = f"Error fetching response: {response.status_code} - {response.text}"
                    response_placeholder.error(err_msg)
        except Exception as e:
            response_placeholder.error(f"Error fetching response: {e}")