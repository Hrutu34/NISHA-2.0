import os
import time
import glob
import base64
import streamlit as st
from src.rag_engine import get_rag_chain, clean_chunk
from src.ingestion import load_and_index_documents

st.set_page_config(
    page_title="NISHA 2.0 | Internal Policy Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, Surgical CSS 
st.markdown("""
<style>
    /* Modern Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.8) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        padding: 28px 32px;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.2);
    }
    .hero-title {
        font-size: 28px;
        font-weight: 700;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #60A5FA 0%, #A78BFA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero-subtitle {
        color: #E2E8F0;
        font-size: 15px;
        margin-top: 8px;
        margin-bottom: 14px;
    }

    /* Badge Pills */
    .chip {
        display: inline-block;
        padding: 4px 12px;
        font-size: 11px;
        font-weight: 600;
        border-radius: 20px;
        background: rgba(59, 130, 246, 0.15);
        color: #93C5FD;
        border: 1px solid rgba(59, 130, 246, 0.4);
        margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to embed PDFs directly in Streamlit
def display_pdf(file_path, filename):
    try:
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()
            base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
        
        # Foolproof Fallback: Add a native download button
        st.download_button(
            label="📥 Download Policy Document",
            data=pdf_bytes,
            file_name=filename,
            mime="application/pdf"
        )
        
        st.write("") # Small spacer
        
        # Switch from <iframe> to <embed> to bypass Edge/Chrome security blocks
        pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="100%" height="650" type="application/pdf">'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Could not load PDF: {e}")

# Helper function to list indexed documents (Supports both PDF and MD)
def get_available_policies(data_dir: str = "./data/sample_policies"):
    files = glob.glob(f"{data_dir}/*.pdf") + glob.glob(f"{data_dir}/*.md")
    policy_list = []
    for f in sorted(files):
        filename = os.path.basename(f)
        clean_name = filename.replace("_", " ").replace(".md", "").replace(".pdf", "").title()
        policy_list.append({"filename": filename, "name": clean_name, "path": f})
    return policy_list

# Auto-index policy documents on first boot if DB doesn't exist
if not os.path.exists("./chroma_db"):
    with st.spinner("⚡ Initializing knowledge base and indexing PDFs..."):
        load_and_index_documents()

policies = get_available_policies()

# Sidebar: Controls & Quick Prompts
with st.sidebar:
    st.markdown("### 📚 Knowledge Base")
    st.caption(f"Currently indexed: **{len(policies)} verified policies**")
    
    with st.expander("📄 View Active Directory", expanded=False):
        for p in policies:
            st.markdown(f"• **{p['name']}**  \n`<small>{p['filename']}</small>`", unsafe_allow_html=True)
    
    st.divider()
    st.markdown("### 💡 Quick Queries")
    sample_queries = [
        "What are the eligibility criteria for an internal transfer?",
        "How many days of sick leave can I take without a medical certificate?",
        "What is the WFH ergonomics allowance amount?",
        "What are the night shift allowance and on-call rates?",
        "How do I add my parents to my health insurance?"
    ]
    
    for query in sample_queries:
        if st.button(query, use_container_width=True):
            st.session_state.preset_prompt = query

    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Hero Header Banner
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">💼 NISHA 2.0</div>
    <div class="hero-subtitle">Newcomers' Integration, Support, and Help Assistant — Instant, grounded guidance with inline citations.</div>
    <div>
        <span class="chip">PDF RAG Verified</span>
        <span class="chip">OpenAI GPT-OSS 20b</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Layout Tabs
tab_chat, tab_docs = st.tabs(["💬 Assistant Chat", "📑 Active Policy Viewer (PDFs)"])

# TAB 1: Chat Interface
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "👋 Hi there! I'm **NISHA**, your company policy companion. Ask me any question about leave allocations, travel per diems, medical coverage, hybrid work rules, or IT hardware allowances.",
                "sources": []
            }
        ]

    # Render Conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📌 Source Document Citations"):
                    for s in msg["sources"]:
                        st.markdown(f"**Document:** `{s['source']}`")
                        st.caption(s["snippet"])

# TAB 2: PDF Document Viewer
with tab_docs:
    st.markdown("### 📑 Corporate Policy Library")
    st.caption("Expand a policy below to read the original source PDF document directly in your browser.")
    st.write("")
    
    for p in policies:
        with st.expander(f"📄 {p['name']} ({p['filename']})"):
            if p["path"].endswith(".pdf"):
                # Pass the filename so the download button knows what to name the file!
                display_pdf(p["path"], p["filename"])
            elif p["path"].endswith(".md"):
                with open(p["path"], "r", encoding="utf-8") as f:
                    st.markdown(f.read())

# Chat Input
prompt = st.chat_input("Ask a question about any company policy...")

if "preset_prompt" in st.session_state and st.session_state.preset_prompt:
    prompt = st.session_state.preset_prompt
    st.session_state.preset_prompt = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    
    with tab_chat:
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(prompt)

        try:
            rag_chain, retriever = get_rag_chain()
            
            source_docs = retriever.invoke(prompt)
            formatted_sources = [
                {
                    "source": os.path.basename(d.metadata.get("source", "Policy Doc")),
                    "snippet": d.page_content
                }
                for d in source_docs
            ]

            with st.chat_message("assistant", avatar="🤖"):
                def response_generator():
                    for chunk in rag_chain.stream(prompt):
                        cleaned = clean_chunk(chunk)
                        time.sleep(0.012)
                        yield cleaned

                response_text = st.write_stream(response_generator)
                
                if formatted_sources:
                    with st.expander("📌 Source Document Citations", expanded=False):
                        for s in formatted_sources:
                            st.markdown(f"**Document:** `{s['source']}`")
                            st.caption(s["snippet"])
                            st.divider()

            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "sources": formatted_sources
            })

        except Exception as e:
            st.error(f"Error querying NISHA engine: {e}")