import os
import streamlit as st
from src.rag_engine import get_rag_chain
from src.ingestion import load_and_index_documents

st.set_page_config(page_title="NISHA 2.0 | Policy Assistant", page_icon="💼", layout="wide")
st.title("NISHA 2.0 — Newcomers' Integration, Support and Help Assistant")

# Auto-index policy documents on first boot if DB doesn't exist
if not os.path.exists("./chroma_db"):
    with st.spinner("Initializing policy knowledge base for the first time..."):
        load_and_index_documents()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about leaves, travel allowances, hybrid work, insurance, etc."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        rag_chain, _ = get_rag_chain()
        with st.chat_message("assistant"):
            with st.spinner("Checking policies..."):
                response = rag_chain.invoke(prompt)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    except Exception as e:
        st.error(f"Error querying NISHA engine: {e}")