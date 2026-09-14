import json
import streamlit as st
import requests

# Live Render backend URL with fallback for local development
DEFAULT_API_URL = "https://ai-knowledge-base-assistant-omgm.onrender.com"
API_URL = st.secrets.get("API_URL", DEFAULT_API_URL)

st.set_page_config(page_title="AI Knowledge Base Assistant", page_icon="🤖", layout="wide")
st.title("🤖 AI Knowledge Base Assistant")

# Sidebar Controls
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
    
    if uploaded_file and st.button("Process & Ingest", use_container_width=True):
        with st.spinner("Chunking, embedding, and saving to Supabase..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                res = requests.post(f"{API_URL}/upload-document", files=files, timeout=60)
                if res.status_code == 200:
                    data = res.json()
                    st.success(f"Ingested {data['chunks_created']} chunks from '{data['filename']}'!")
                else:
                    st.error(f"Upload failed: {res.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")

    st.markdown("---")
    st.header("✏️ Add Knowledge Manually")
    title = st.text_input("Title")
    content = st.text_area("Content", height=100)
    
    if st.button("Save Manual Entry", use_container_width=True):
        if title.strip() and content.strip():
            try:
                res = requests.post(f"{API_URL}/add-knowledge", json={"title": title, "content": content}, timeout=30)
                if res.status_code == 200:
                    st.success("Entry added successfully!")
                else:
                    st.error("Failed to add entry.")
            except Exception as e:
                st.error(f"Connection error: {e}")
        else:
            st.warning("Please provide both a title and content.")

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render Previous Chat Turns
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("View Retrieved Sources"):
                for src in msg["sources"]:
                    similarity_val = src.get("similarity", 0)
                    st.markdown(f"**{src.get('title', 'Unknown')}** (Similarity: `{similarity_val:.2f}`)")
                    st.caption(src.get("content", ""))

# Chat Interaction
user_query = st.chat_input("Ask a question about your knowledge base...")
if user_query:
    history_payload = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        def stream_generator():
            try:
                with requests.post(
                    f"{API_URL}/ask-stream",
                    json={"question": user_query, "history": history_payload},
                    stream=True,
                    timeout=60
                ) as res:
                    if res.status_code != 200:
                        yield f"API Error {res.status_code}: {res.text}"
                        return

                    iterator = res.iter_lines(decode_unicode=True)
                    first_line = next(iterator, None)
                    sources = []
                    if first_line:
                        try:
                            meta = json.loads(first_line)
                            sources = meta.get("sources", [])
                            st.session_state._current_sources = sources
                        except json.JSONDecodeError:
                            yield first_line

                    for chunk in iterator:
                        if chunk:
                            yield chunk + "\n"

            except Exception as err:
                yield f"Connection failed: {err}"

        st.session_state._current_sources = []
        full_response = st.write_stream(stream_generator())

        current_sources = st.session_state.get("_current_sources", [])
        if current_sources:
            with st.expander("View Retrieved Sources"):
                for src in current_sources:
                    similarity_val = src.get("similarity", 0)
                    st.markdown(f"**{src.get('title', 'Unknown')}** (Similarity: `{similarity_val:.2f}`)")
                    st.caption(src.get("content", ""))

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": current_sources
        })