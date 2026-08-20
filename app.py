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

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        background: linear-gradient(90deg, #0F172A 0%, #1E3A8A 50%, #2563EB 100%);
        padding: 24px 28px;
        border-radius: 14px;
        color: white;
        margin-bottom: 20px;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 26px;
        font-weight: 700;
    }
    .main-header p {
        color: #93C5FD;
        margin: 6px 0 0 0;
        font-size: 14px;
    }
    .policy-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 10px;
    }
    .badge {
        display: inline-block;
        padding: 3px 8px;
        font-size: 11px;
        font-weight: 600;
        border-radius: 12px;
        background-color: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to get list of indexed documents
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

# Header
st.markdown("""
<div class="main-header">
    <h1>💼 NISHA 2.0</h1>
    <p>Newcomers' Integration, Support, and Help Assistant — Instant, grounded answers to all company policies & benefits.</p>
</div>
""", unsafe_allow_html=True)

policies = get_available_policies()

# Sidebar: Controls & Active Policy Directory
with st.sidebar:
    st.header("📚 Knowledge Base")
    st.caption(f"Currently indexed: **{len(policies)} active policy documents**")
    
    with st.expander("📄 View Available Policies", expanded=False):
        for p in policies:
            st.markdown(f"• **{p['name']}**  \n`{p['filename']}`")
    
    st.divider()
    st.subheader("💡 Suggested Prompts")
    sample_queries = [
        "What are the eligibility criteria for an internal transfer?",
        "How many days of sick leave can I take without a doctor's certificate?",
        "What is the WFH ergonomics allowance amount?",
        "What are the night shift allowance and on-call rates?",
        "How do I add my parents to my group health insurance?"
    ]
    
    for query in sample_queries:
        if st.button(query, use_container_width=True):
            st.session_state.preset_prompt = query

    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Layout: Main Chat vs Knowledge Base Overview Tabs
tab_chat, tab_docs = st.tabs(["💬 Assistant Chat", "📑 Active Policy Documents"])

# TAB 1: Chatbot Interface
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "👋 Hi there! I'm **NISHA**, your company policy companion. Ask me any question about leave rules, travel allowances, health insurance, hybrid work guidelines, or equipment policies.",
                "sources": []
            }
        ]

    # Render Conversation History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📌 Source Document Citations"):
                    for s in msg["sources"]:
                        st.markdown(f"**Document:** `{s['source']}`")
                        st.caption(s["snippet"])

    # Handle Input
    prompt = st.chat_input("Ask a question about any company policy...")

    if "preset_prompt" in st.session_state and st.session_state.preset_prompt:
        prompt = st.session_state.preset_prompt
        st.session_state.preset_prompt = None

    if prompt:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(prompt)

        try:
            rag_chain, retriever = get_rag_chain()
            
            # Fetch source documents
            source_docs = retriever.invoke(prompt)
            formatted_sources = [
                {
                    "source": os.path.basename(d.metadata.get("source", "Policy Doc")),
                    "snippet": d.page_content
                }
                for d in source_docs
            ]

            with st.chat_message("assistant", avatar="🤖"):
                # Stream response chunk by chunk
                def response_generator():
                    for chunk in rag_chain.stream(prompt):
                        cleaned = clean_chunk(chunk)
                        time.sleep(0.015)  # Smooth, readable typing cadence
                        yield cleaned

                response_text = st.write_stream(response_generator)
                
                # Render source citations
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

# TAB 2: Knowledge Base Inspector
with tab_docs:
    st.subheader("📑 Indexed Corporate Policies")
    st.info("NISHA answers all employee queries based exclusively on the policy documents below:")
    
    col1, col2 = st.columns(2)
    for idx, p in enumerate(policies):
        target_col = col1 if idx % 2 == 0 else col2
        with target_col:
            with st.container():
                st.markdown(f"#### 📄 {p['name']}")
                st.caption(f"Filename: `{p['filename']}`")
                
                # Show policy snippet
                if os.path.exists(p["path"]):
                    try:
                        with open(p["path"], "r", encoding="utf-8") as f:
                            content = f.read()
                            # Show first 300 characters as preview
                            preview = content[:300].strip() + "..."
                            st.markdown(f"```text\n{preview}\n```")
                    except Exception:
                        st.caption("Document preview unavailable.")
                st.divider()