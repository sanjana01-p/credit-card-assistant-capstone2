import streamlit as st
import sys
import os
import re
import json
import requests
import base64

sys.path.append(os.path.abspath("."))

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def parse_pages(page_no: str) -> list:
    if not page_no or page_no in ("N/A", "NA", ""):
        return []
    return [p.strip() for p in str(page_no).split(",") if p.strip()]


def split_answer_by_pages(answer: str, num_pages: int) -> list:
    raw = re.split(r'(?<=[.!?])\s+', answer.strip())
    sentences = [s.strip() for s in raw if s.strip()]
    if not sentences or num_pages == 0:
        return [answer] * num_pages
    base, remainder = divmod(len(sentences), num_pages)
    groups, idx = [], 0
    for i in range(num_pages):
        count = base + (1 if i < remainder else 0)
        chunk = " ".join(sentences[idx: idx + count])
        groups.append(chunk if chunk else answer)
        idx += count
    return groups


    
def format_answer(text: str) -> str:
    if not text:
        return ""

    # remove weird symbols
    text = re.sub(r"[∗•·▪■□●]+", "", text)

    # fix broken words
    text = re.sub(r"(?<=\w)\s(?=\w)", "", text)

    # normalize spaces
    text = re.sub(r"\s+", " ", text)

    # fix punctuation spacing
    text = re.sub(r"\s*([,:])\s*", r"\1 ", text)

    # fix numbers
    text = re.sub(r"(\d)\s*,\s*(\d)", r"\1,\2", text)
    text = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", text)

    # add line breaks for readability
    text = re.sub(r"\.\s+", ".\n\n", text)

    return text.strip()

# ─────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NorthStar Bank – Credit Card Assistant",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #f0f4ff 0%, #fafbff 60%, #f5f0ff 100%);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1f5c 0%, #1a3080 60%, #0d1a4a 100%);
    border-right: none;
}
[data-testid="stSidebar"] * {
    color: #e8eeff !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.18);
    color: #e8eeff !important;
    border-radius: 8px;
    font-size: 13px;
    padding: 8px 12px;
    text-align: left;
    width: 100%;
    transition: all 0.2s ease;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.16);
    border-color: rgba(255,255,255,0.35);
    transform: translateX(3px);
}

[data-testid="stChatMessage"] {
    background: #ffffff;
    border: 1px solid #e8ecf8;
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 12px;
    box-shadow: 0 2px 8px rgba(15,31,92,0.06);
}

[data-testid="stChatInput"] {
    border-radius: 12px !important;
    border: 1.5px solid #c5cef0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #3B5BDB !important;
    box-shadow: 0 0 0 3px rgba(59,91,219,0.12) !important;
}

[data-testid="stExpander"] {
    border: 1px solid #dde3f5;
    border-radius: 10px;
    background: #ffffff;
}

[data-testid="stFileUploader"] {
    border: 2px dashed #8fa3e8;
    border-radius: 12px;
    background: rgba(59,91,219,0.03);
    padding: 10px;
}

.stTextInput > div > div > input, .stTextArea > div > textarea {
    border-radius: 10px;
    border: 1.5px solid #c5cef0;
    font-family: 'DM Sans', sans-serif;
}
.stTextInput > div > div > input:focus, .stTextArea > div > textarea:focus {
    border-color: #3B5BDB;
    box-shadow: 0 0 0 3px rgba(59,91,219,0.10);
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #3B5BDB, #2344c7);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    padding: 10px 24px;
    transition: all 0.2s;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #2344c7, #1a35b0);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(59,91,219,0.35);
}

hr { border-color: #e4e9f7; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #f0f4ff; }
::-webkit-scrollbar-thumb { background: #b0beea; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────────────────

if "role" not in st.session_state:
    st.session_state.role = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0


# ─────────────────────────────────────────────────────────────
# LOGIN PAGE
# ─────────────────────────────────────────────────────────────

if st.session_state.role is None:
    col_l, col_c, col_r = st.columns([1, 1.2, 1])
    with col_c:
        st.markdown("""
        <div style="text-align:center; padding: 48px 0 28px 0;">
            <div style="font-size: 52px; margin-bottom: 10px;">💳</div>
            <div style="font-size: 28px; font-weight: 700; color: #0f1f5c; letter-spacing: -0.5px;">
                NorthStar Bank
            </div>
            <div style="font-size: 15px; color: #6b7cb0; margin-top: 6px;">
                Credit Card Intelligence Portal
            </div>
        </div>
        """, unsafe_allow_html=True)

        # st.markdown("""
        # <div style="background:#ffffff; border:1px solid #e0e7f5; border-radius:18px;
        # padding:36px 36px 30px 36px; box-shadow: 0 8px 32px rgba(15,31,92,0.10);">
        # """, unsafe_allow_html=True)

        st.markdown("#### Sign In to Continue")
        role = st.selectbox("Role", ["User", "Admin"], label_visibility="collapsed")

        if role == "Admin":
            password = st.text_input("Password", type="password", placeholder="Enter admin password",
                                     label_visibility="collapsed")
            if st.button("Login as Admin", use_container_width=True, type="primary"):
                if password == "admin123":
                    st.session_state.role = "admin"
                    st.rerun()
                else:
                    st.error("Invalid password. Please try again.")
        else:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            if st.button("Continue as User", use_container_width=True, type="primary"):
                st.session_state.role = "user"
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center; margin-top:20px; font-size:12px; color:#9aaac8;">
            NorthStar Bank · Credit Card Intelligence Portal · v2.1
        </div>
        """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────
# SIDEBAR (post-login)
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 16px 0;">
        <div style="font-size: 20px; font-weight: 700; letter-spacing: -0.3px;">💳 NorthStar Bank</div>
        <div style="font-size: 12px; opacity: 0.65; margin-top: 2px;">Credit Card Assistant</div>
    </div>
    """, unsafe_allow_html=True)

    role_color = "#e8a020" if st.session_state.role == "admin" else "#1D9E75"
    role_label = "Admin" if st.session_state.role == "admin" else "User"
    st.markdown(f"""
    <div style="display:inline-block; background:{role_color}22; border:1px solid {role_color}55;
    color:{role_color}; padding:3px 12px; border-radius:20px; font-size:12px;
    font-weight:600; margin-bottom:16px;">{role_label}</div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if st.session_state.role == "user":
        st.markdown("### 💡 Try asking...")
        suggestions = [
            "Summarise CC-881001 spending for March 2026",
            "What are the reward points earn rates?",
            "Explain the fee waiver threshold for Classic card",
            "Show international transactions for CC-881001",
            "What is the billing cycle structure?",
            "Compare CC-881001 spending vs last month",
        ]
        for s in suggestions:
            if st.button(s, key=f"suggest_{s}"):
                st.session_state["prefill_query"] = s
                st.rerun()

        st.markdown("---")
        st.markdown(f"""
        <div style="font-size:12px; opacity:0.7; line-height:1.8;">
            📊 Queries this session: <b>{st.session_state.total_queries}</b>
        </div>
        """, unsafe_allow_html=True)

    elif st.session_state.role == "admin":
        st.markdown("### 🛠 Admin Panel")
        st.markdown("""
        <div style="font-size:13px; opacity:0.75; line-height:1.7;">
            Upload PDF documents to the knowledge base for RAG retrieval.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🚪 Logout", key="logout_btn"):
        st.session_state.role = None
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.rerun()

    st.caption("Powered by Streaming Multimodal RAG")


# ═════════════════════════════════════════════════════════════
# ADMIN VIEW
# ═════════════════════════════════════════════════════════════

if st.session_state.role == "admin":

    st.markdown("""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;">
        <div style="font-size:28px;">🛠</div>
        <div>
            <div style="font-size:24px; font-weight:700; color:#0f1f5c; letter-spacing:-0.5px;">
                Document Management
            </div>
            <div style="font-size:14px; color:#6b7cb0; margin-top:2px;">
                Upload and ingest PDF documents into the NorthStar knowledge base
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.markdown("""
        <div style="background:#ffffff; border:1px solid #e0e7f5; border-radius:14px;
        padding:28px; box-shadow: 0 2px 12px rgba(15,31,92,0.07);">
            <div style="font-size:16px; font-weight:600; color:#0f1f5c; margin-bottom:18px;">
                📤 Upload PDF Document
            </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Supported format: PDF. Max size: 50MB."
        )

        if uploaded_file:
            size_kb = uploaded_file.size / 1024
            st.markdown(f"""
            <div style="background:#f0f4ff; border:1px solid #c5d0f5; border-radius:10px;
            padding:12px 16px; margin:12px 0; display:flex; align-items:center; gap:10px;">
                <span style="font-size:22px;">📄</span>
                <div>
                    <div style="font-size:14px; font-weight:600; color:#0f1f5c;">{uploaded_file.name}</div>
                    <div style="font-size:12px; color:#6b7cb0;">{size_kb:.1f} KB · PDF Document</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("⚡ Upload & Ingest", type="primary", use_container_width=True):
                with st.spinner("Uploading and embedding document…"):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                        response = requests.post(
                            "http://localhost:8000/api/v1/admin/upload",
                            files=files,
                            timeout=300
                        )
                        if response.status_code == 200:
                            resp_data = response.json()
                            st.success(f"✅ **{resp_data.get('file', uploaded_file.name)}** uploaded and embedded successfully!")
                            st.markdown(f"""
                            <div style="background:#f0fff8; border:1px solid #a8e6cc;
                            border-radius:10px; padding:14px 18px; margin-top:8px;">
                                <div style="font-size:13px; color:#1D9E75; font-weight:600;">
                                    Document is now available for RAG retrieval
                                </div>
                                <div style="font-size:12px; color:#555; margin-top:4px;">
                                    {resp_data.get('message', 'Ingestion complete.')}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error(f"Upload failed: {response.text}")
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background:#ffffff; border:1px solid #e0e7f5; border-radius:14px;
        padding:26px; box-shadow: 0 2px 12px rgba(15,31,92,0.07);">
            <div style="font-size:16px; font-weight:600; color:#0f1f5c; margin-bottom:16px;">
                📋 Ingestion Pipeline
            </div>
        """, unsafe_allow_html=True)

        steps = [
            ("1", "#3B5BDB", "PDF uploaded to server"),
            ("2", "#3B5BDB", "Document parsed & chunked"),
            ("3", "#3B5BDB", "Embeddings generated"),
            ("4", "#3B5BDB", "Stored in vector database"),
            ("✓", "#1D9E75", "Ready for retrieval"),
        ]
        steps_html = ""
        for num, color, label in steps:
            steps_html += f"""
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
                <span style="width:26px; height:26px; background:{color}; color:white;
                border-radius:50%; display:inline-flex; align-items:center; justify-content:center;
                font-size:11px; font-weight:700; flex-shrink:0;">{num}</span>
                <span style="font-size:13px; color:#4a5580;">{label}</span>
            </div>"""

        st.markdown(steps_html + "</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#fff8ec; border:1px solid #f5d98a; border-radius:14px;
        padding:20px 22px; margin-top:16px;">
            <div style="font-size:13px; font-weight:600; color:#b07a10; margin-bottom:8px;">
                ⚠️ Admin Notes
            </div>
            <div style="font-size:12px; color:#7a5a10; line-height:1.9;">
                • Only PDF format is supported<br/>
                • Large files may take up to 2 minutes<br/>
                • Duplicate filenames overwrite existing<br/>
                • Verify document before uploading
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.stop()


# ═════════════════════════════════════════════════════════════
# USER CHAT VIEW
# ═════════════════════════════════════════════════════════════

st.markdown("""
<div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;">
    <div style="font-size:28px;">💳</div>
    <div>
        <div style="font-size:24px; font-weight:700; color:#0f1f5c; letter-spacing:-0.5px;">
            NorthStar Bank – Credit Card Assistant
        </div>
        <div style="font-size:14px; color:#6b7cb0; margin-top:2px;">
            Ask about card variants, spend summaries, rewards, billing cycles, and transaction data.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown("---")


# ─────────────────────────────────────────────────────────────
# Chat History Replay
# ─────────────────────────────────────────────────────────────

def render_chunks_ui(chunks_list):
    """Render metadata card + snippets expander for a chunks_list."""
    if not chunks_list:
        return

    unique_pages = sorted(
        set(c["page_no"] for c in chunks_list if c["page_no"] not in ("N/A", "", None)),
        key=lambda x: int(x) if str(x).isdigit() else x
    )
    unique_docs  = sorted(set(c["document_name"] for c in chunks_list if c["document_name"] not in ("Unknown", "", None)))
    unique_cites = sorted(set(c["citation"] for c in chunks_list if c["citation"] not in ("N/A", "", None)))

    pages_str = ", ".join(unique_pages) if unique_pages else "N/A"
    doc_str   = "<br/>".join(f"📄 {d}" for d in unique_docs) if unique_docs else "📄 Unknown"
    cite_str  = "<br/>".join(f"🔖 {c}" for c in unique_cites) if unique_cites else "🔖 N/A"

    st.markdown(f"""
    <div style="background:#f0f4ff;border:1px solid #d0dbf5;border-radius:10px;
    padding:16px 20px;margin:16px 0 8px 0;">
        <div style="display:flex;flex-wrap:wrap;gap:24px;">
            <div style="flex:1;min-width:180px;">
                <div style="font-size:11px;font-weight:600;color:#888;
                text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Document</div>
                <div style="font-size:14px;color:#222;">{doc_str}</div>
            </div>
            <div style="flex:1;min-width:120px;">
                <div style="font-size:11px;font-weight:600;color:#888;
                text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Pages Referenced</div>
                <div style="font-size:14px;color:#222;">📌 {pages_str}</div>
            </div>
            <div style="flex:1;min-width:180px;">
                <div style="font-size:11px;font-weight:600;color:#888;
                text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Citation</div>
                <div style="font-size:14px;color:#222;">{cite_str}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander(f"📚 Retrieved Snippets ({len(chunks_list)})", expanded=False):
        for i, chunk in enumerate(chunks_list, 1):
            safe_text = str(chunk["text"]).replace("{", "&#123;").replace("}", "&#125;")
            st.markdown(f"""
            <div style="background:#f9fafb;border:1px solid #eaecf0;
            border-left:3px solid #3B5BDB;border-radius:8px;
            padding:14px 16px;margin-bottom:10px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                <b>Snippet {i}</b>
                <span style="background:#e8f0fe;color:#3B5BDB;padding:2px 8px;
                border-radius:12px;font-size:12px;">Page {chunk['page_no']}</span>
            </div>
            📄 <b>{chunk['document_name']}</b><br/><br/>
            <span style="color:#333;line-height:1.6;">{safe_text}</span>
            <hr style="margin:10px 0;border-color:#eaecf0;"/>
            <small style="color:#888;">📎 Source: {chunk['citation']}</small>
            </div>
            """, unsafe_allow_html=True)


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)
        if msg["role"] == "assistant" and msg.get("chunks_list"):
            render_chunks_ui(msg["chunks_list"])


# ─────────────────────────────────────────────────────────────
# Chat Input & Streaming
# ─────────────────────────────────────────────────────────────

prefill = st.session_state.pop("prefill_query", None)
user_query = st.chat_input("Ask about NorthStar credit cards, spend summaries, rewards…") or prefill

if user_query:
    st.session_state.total_queries += 1

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_answer = ""
        chunks_list = []

        try:
            response = requests.post(
                "http://localhost:8000/api/v1/query/stream",
                json={"query": user_query},
                stream=True,
                timeout=300
            )

            for line in response.iter_lines():
                if not line:
                    continue

                decoded = line.decode("utf-8").strip()

                try:
                    data = json.loads(decoded)
                except:
                    continue

                if not isinstance(data, dict):
                    continue

                # ── Flat JSON response (non-streaming) ──────────────
                if "answer" in data:
                    full_answer = data["answer"]  # ← ONLY the answer text
                    placeholder.markdown(full_answer, unsafe_allow_html=True)

                    # ── Also extract citations from the same flat response ──
                    doc_name = data.get("document_name", "")
                    page_no  = data.get("page_no", "")
                    citation = data.get("policy_citations", "")

                    if doc_name and doc_name not in ("N/A", "agentic_rag_db", ""):
                        for pg in str(page_no).split(","):
                            pg = pg.strip()
                            if pg and pg != "N/A":
                                chunks_list.append({
                                    "document_name": doc_name,
                                    "page_no":       pg,
                                    "citation":      citation if citation and citation != "N/A" else doc_name,
                                    "text":          "",   # no raw chunk text in flat response
                                })
                    continue  # ← skip msg_type processing for this line

                msg_type = data.get("type")

                # ── Streamed text chunk ──────────────────────────────
                if msg_type == "text":
                    content = data.get("content", "")
                    if isinstance(content, dict):
                        content = content.get("text", "")
                    elif isinstance(content, list):
                        content = "".join(
                            c.get("text", "") if isinstance(c, dict) else str(c)
                            for c in content
                        )
                    full_answer += content
                    placeholder.markdown(full_answer, unsafe_allow_html=True)

                # ── Image ────────────────────────────────────────────
                elif msg_type == "image":
                    image_rendered = False

                    image_path = data.get("image_path") or data.get("path")
                    if image_path:
                        norm_path = image_path.replace("\\", "/")
                        if os.path.exists(norm_path):
                            st.image(norm_path,
                                    caption=f"{data.get('source', 'Document')} | Page {data.get('page', 'N/A')}",
                                    use_container_width=True)
                            image_rendered = True
                        else:
                            # try resolving just the filename in data/images/
                            filename = os.path.basename(norm_path)
                            for base_dir in ["data/images", "./data/images", "."]:
                                candidate = os.path.join(base_dir, filename)
                                if os.path.exists(candidate):
                                    st.image(candidate,
                                            caption=f"{data.get('source', 'Document')} | Page {data.get('page', 'N/A')}",
                                            use_container_width=True)
                                    image_rendered = True
                                    break

                    if not image_rendered:
                        img_b64 = data.get("content") or data.get("image") or data.get("data")
                        if isinstance(img_b64, str) and img_b64:
                            try:
                                if "," in img_b64:
                                    img_b64 = img_b64.split(",", 1)[1]
                                st.image(base64.b64decode(img_b64),
                                        caption=f"{data.get('source', 'Document')} | Page {data.get('page', 'N/A')}",
                                        use_container_width=True)
                                image_rendered = True
                            except Exception as e:
                                st.warning(f"⚠️ Could not render image: {e}")

                    if not image_rendered:
                        st.info("🖼️ Image referenced but could not be displayed.")

                # ── Chunks ───────────────────────────────────────────
                elif msg_type == "chunks":
                    raw = data.get("text") or data.get("content") or data.get("chunks") or data

                    # ✅ CASE 1: SQL / structured response (dict)
                    if isinstance(raw, dict):

                        # ✅ 1️⃣ Extract ANSWER once (TOP display)
                        if raw.get("answer") and not full_answer:
                            full_answer = raw["answer"]
                            placeholder.markdown(full_answer, unsafe_allow_html=True)

                        # ✅ 2️⃣ Extract SQL/snippet separately (BOTTOM display)
                        sql_query_executed = raw.get("sql_query_executed")
                        doc_name = raw.get("document_name", "agentic_rag_db")
                        page_no = raw.get("page_no", "N/A")

                        sql_text = ""
                        if sql_query_executed:
                            sql_text = f"SQL executed: {sql_query_executed}"

                        chunks_list.append({
                            "document_name": doc_name,
                            "page_no": page_no,
                            "citation": doc_name,
                            "text": sql_text or "SQL query executed.",
                        })

                    # ✅ CASE 2: Normal RAG chunks (list)
                    else:
                        if isinstance(raw, dict):
                            raw = [raw]

                        for item in raw:
                            if isinstance(item, dict):
                                chunks_list.append({
                                    "document_name": item.get("source", "Unknown"),
                                    "page_no": str(item.get("page", "N/A")),
                                    "citation": item.get("source", "Unknown"),
                                    "text": item.get("content", ""),
                                })
                        # elif isinstance(item, str) and item.strip():
                        #     chunks_list.append({
                        #         "document_name": data.get("document_name", "Unknown"),
                        #         "page_no":       data.get("page_no", "N/A"),
                        #         "citation":      data.get("citation", "N/A"),
                        #         "text":          item,
                        #     })

        except Exception as e:
            full_answer = f"⚠️ Connection error: {str(e)}"
            placeholder.markdown(full_answer)

        # ── Metadata card + snippets (below answer) ──────────────
        render_chunks_ui(chunks_list)

        # ── Persist ───────────────────────────────────────────────
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_answer,
            "chunks_list": chunks_list,
        })