import streamlit as st
from src.rag_engine import get_rag_chain

st.set_page_config(page_title="NISHA 2.0 | Policy Assistant", layout="wide")
st.title("💼 NISHA 2.0 — Policy & Onboarding Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User query input
if prompt := st.chat_input("Ask any question about company policies, benefits, leave, etc."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        rag_chain, _ = get_rag_chain()
        with st.chat_message("assistant"):
            with st.spinner("Reviewing company policies..."):
                response = rag_chain.invoke(prompt)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    except Exception as e:
        st.error(f"Error querying NISHA engine: {e}")