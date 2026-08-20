import os
import time
import glob
import streamlit as st
from src.rag_engine import get_rag_chain, clean_chunk
from src.ingestion import load_and_index_documents

st.set_page_config(
    page_title="NISHA 2.0 | Internal Policy Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, Surgical CSS (No destructive global overrides)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    /* Apply custom font without breaking Streamlit's layout */
    html, body, [class*="st-"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    /* Style the Sidebar Buttons properly so text is always visible */
    div.stButton > button {
        background-color: #1E293B !important; 
        color: #F8FAFC !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        white-space: normal !important; /* Allows long text to wrap nicely */
        height: auto !important;
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:hover {
        border-color: #3B82F6 !important;
        color: #60A5FA !important;
        background-color: #0F172A !important;
    }

    /* Modern Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.5) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(59, 130, 246, 0.2);
        padding: 28px 32px;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 4px 24px -4px rgba(0, 0, 0, 0.4);
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
        color: #CBD5E1;
        font-size: 15px;
        margin-top: 8px;
    }

    /* Glass Cards for Policy Viewer */
    .glass-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
        transition: all 0.2s ease-in-out;
    }
    .glass-card:hover {
        border-color: rgba(96, 165, 250, 0.5);
        background: rgba(30, 41, 59, 0.8);
    }

    /* Badge Pills */
    .chip {
        display: inline-block;
        padding: 4px 12px;
        font-size: 11px;
        font-weight: 600;
        border-radius: 20px;
        background: rgba(59, 130, 246, 0.1);
        color: #93C5FD;
        border: 1px solid rgba(59, 130, 246, 0.3);
        margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to list indexed documents
def get_available_policies(data_dir: str = "./data/sample_policies"):
    files = glob.glob(f"{data_dir}/*.md") + glob.glob(f"{data_dir}/*.pdf")
    policy_list = []
    for f in sorted(files):
        filename = os.path.basename(f)
        clean_name = filename.replace("_", " ").replace(".md", "").replace(".pdf", "").title()
        policy_list.append({"filename": filename, "name": clean_name, "path": f})
    return policy_list

# Auto-index policy documents on first boot if DB doesn't exist
if not os.path.exists("./chroma_db"):
    with st.spinner("⚡ Initializing knowledge base and indexing policies..."):
        load_and_index_documents()

policies = get_available_policies()

# Sidebar: Controls & Quick Prompts
with st.sidebar:
    st.markdown("### 📚 Knowledge Base")
    st.caption(f"Currently indexed: **{len(policies)} verified policies**")
    
    with st.expander("📄 View Available Policies", expanded=False):
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
    <div style="margin-top: 14px;">
        <span class="chip">RAG Verified</span>
        <span class="chip">OpenAI GPT-OSS 20b</span>
        <span class="chip">Zero-Cost Stack</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Layout Tabs
tab_chat, tab_docs = st.tabs(["💬 Assistant Chat", "📑 Active Policy Documents"])

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

# TAB 2: Knowledge Base Inspector
with tab_docs:
    st.markdown("### 📑 Indexed Corporate Policies")
    st.caption("All responses are strictly derived from the official markdown policy files below:")
    st.write("")
    
    col1, col2 = st.columns(2)
    for idx, p in enumerate(policies):
        target_col = col1 if idx % 2 == 0 else col2
        with target_col:
            st.markdown(f"""
            <div class="glass-card">
                <h4 style="margin: 0 0 4px 0; color: #60A5FA;">📄 {p['name']}</h4>
                <div style="color: #94A3B8; font-size: 13px; margin-bottom: 12px;">File: <code>{p['filename']}</code></div>
            </div>
            """, unsafe_allow_html=True)
            
            if os.path.exists(p["path"]):
                try:
                    with open(p["path"], "r", encoding="utf-8") as f:
                        preview = f.read()[:250].strip() + "..."
                        st.markdown(f"```text\n{preview}\n```")
                except Exception:
                    st.caption("Preview unavailable.")
            st.write("")

# Chat Input placed at Root Level
prompt = st.chat_input("Ask a question about any company policy...")

if "preset_prompt" in st.session_state and st.session_state.preset_prompt:
    prompt = st.session_state.preset_prompt
    st.session_state.preset_prompt = None

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    
    with tab_chat:
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(prompt)

        try:
            rag_chain, retriever = get_rag_chain()
            
            # Retrieve source chunks
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