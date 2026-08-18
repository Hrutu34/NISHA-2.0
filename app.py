import os
import streamlit as st
from src.rag_engine import get_rag_chain
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
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 28px;
        font-weight: 700;
    }
    .main-header p {
        color: #E0E7FF;
        margin: 5px 0 0 0;
        font-size: 14px;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        font-size: 12px;
        font-weight: 600;
        border-radius: 20px;
        background-color: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        margin-right: 6px;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 12px 18px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

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

# Sidebar
with st.sidebar:
    st.header("⚙️ Assistant Controls")
    st.markdown("<div><span class='badge'>v2.0 Active</span><span class='badge'>RAG Powered</span></div>", unsafe_allow_html=True)
    st.write("")
    
    st.subheader("💡 Quick Questions")
    sample_queries = [
        "What are the eligibility criteria for an internal transfer?",
        "How many days of sick leave can I take without a medical certificate?",
        "What is the WFH ergonomics allowance and how do I claim it?",
        "What is the night shift allowance rate?",
        "How do I add my parents to my health insurance?"
    ]
    
    for query in sample_queries:
        if st.button(query, use_container_width=True):
            st.session_state.preset_prompt = query

    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Initialize Chat Memory
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hi there! I'm **NISHA**, your company policy companion. Ask me anything about leave allowances, travel reimbursements, medical benefits, probation timelines, or equipment guidelines."}
    ]

# Render Conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

# Handle Input
prompt = st.chat_input("Ask a question about any company policy...")

# Check if a preset query was clicked
if "preset_prompt" in st.session_state and st.session_state.preset_prompt:
    prompt = st.session_state.preset_prompt
    st.session_state.preset_prompt = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(prompt)

    try:
        rag_chain, retriever = get_rag_chain()
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Searching company handbook..."):
                response = rag_chain.invoke(prompt)
                st.markdown(response)
                
                # Show retrieved chunks for transparency
                with st.expander("🔍 View Retrieved Source Policy Chunks"):
                    source_docs = retriever.invoke(prompt)
                    for i, doc in enumerate(source_docs):
                        st.markdown(f"**Chunk {i+1}** — *{doc.metadata.get('source', 'Unknown Doc')}*")
                        st.caption(doc.page_content)
                        st.divider()

        st.session_state.messages.append({"role": "assistant", "content": response})
    except Exception as e:
        st.error(f"Error querying NISHA engine: {e}")