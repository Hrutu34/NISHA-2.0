import os
import time
import glob
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import streamlit.components.v1 as components
from src.rag_engine import get_rag_chain, clean_chunk
from src.ingestion import load_and_index_documents

st.set_page_config(
    page_title="NISHA 2.0 | Internal Policy Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Surgically scoped CSS
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

def voice_input_widget():
    """Injects a microphone icon button directly next to Streamlit's send button."""
    voice_html = """
    <script>
    (function initFloatingVoiceWidget() {
        const parentDoc = window.parent.document;

        // Clean up previous instances to prevent duplicates on Streamlit reruns
        const existingBtn = parentDoc.getElementById('nisha-embedded-mic-btn');
        if (existingBtn) {
            existingBtn.remove();
        }

        let recognition;
        let isListening = false;

        function setNativeValue(element, value) {
            const valueSetter = Object.getOwnPropertyDescriptor(element, 'value').set;
            const prototype = Object.getPrototypeOf(element);
            const prototypeValueSetter = Object.getOwnPropertyDescriptor(prototype, 'value').set;
            
            if (valueSetter && valueSetter !== prototypeValueSetter) {
                prototypeValueSetter.call(element, value);
            } else {
                valueSetter.call(element, value);
            }
            element.dispatchEvent(new Event('input', { bubbles: true }));
            element.dispatchEvent(new Event('change', { bubbles: true }));
        }

        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.lang = 'en-US';

            recognition.onresult = function(event) {
                let interimTranscript = '';
                let finalTranscript = '';

                for (let i = event.resultIndex; i < event.results.length; ++i) {
                    if (event.results[i].isFinal) {
                        finalTranscript += event.results[i][0].transcript + ' ';
                    } else {
                        interimTranscript += event.results[i][0].transcript;
                    }
                }

                const chatTextarea = parentDoc.querySelector('[data-testid="stChatInput"] textarea');
                if (chatTextarea) {
                    const combined = (finalTranscript + interimTranscript).trim();
                    if (combined) {
                        setNativeValue(chatTextarea, combined);
                    }
                }
            };

            recognition.onerror = function(event) {
                console.error("Speech error: ", event.error);
                stopDictation();
            };

            recognition.onend = function() {
                if (isListening) stopDictation();
            };
        }

        function toggleDictation(btn) {
            if (!isListening) {
                startDictation(btn);
            } else {
                stopDictation(btn);
            }
        }

        function startDictation(btn) {
            if (!recognition) return;
            try {
                recognition.start();
                isListening = true;
                btn.style.backgroundColor = 'rgba(239, 68, 68, 0.2)';
                btn.style.borderColor = '#EF4444';
                btn.innerText = '🔴';
            } catch(e) {
                console.warn(e);
            }
        }

        function stopDictation(btn) {
            if (!recognition) return;
            try {
                recognition.stop();
            } catch(e) {}
            isListening = false;
            if (btn) {
                btn.style.backgroundColor = 'transparent';
                btn.style.borderColor = 'transparent';
                btn.innerText = '🎙️';
            }
        }

        // Mount the mic button into Streamlit's chat input action container
        function attachMicButton() {
            const chatContainer = parentDoc.querySelector('[data-testid="stChatInput"]');
            if (!chatContainer) {
                setTimeout(attachMicButton, 300);
                return;
            }

            const sendButton = chatContainer.querySelector('button');
            if (!sendButton || !sendButton.parentElement) {
                setTimeout(attachMicButton, 300);
                return;
            }

            // Create mic button
            const micBtn = parentDoc.createElement('button');
            micBtn.id = 'nisha-embedded-mic-btn';
            micBtn.type = 'button';
            micBtn.innerText = '🎙️';
            micBtn.title = 'Click to dictate (Live STT)';
            
            // Inline icon styling matching chat input height
            Object.assign(micBtn.style, {
                background: 'transparent',
                border: '1px solid transparent',
                borderRadius: '8px',
                cursor: 'pointer',
                fontSize: '18px',
                padding: '4px 8px',
                marginRight: '4px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                transition: 'all 0.2s ease'
            });

            micBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                toggleDictation(micBtn);
            });

            // Insert directly to the left of the send button
            sendButton.parentElement.insertBefore(micBtn, sendButton);
        }

        attachMicButton();
    })();
    </script>
    """
    components.html(voice_html, height=0, width=0)

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
            st.markdown(f"• **{p['name']}**")
            st.caption(f"`{p['filename']}`")
            st.write("")
    
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
                # Open the file once to serve both the download button and the viewer
                with open(p["path"], "rb") as f:
                    pdf_bytes = f.read()
                
                # Keep the Download Button
                st.download_button(
                    label="📥 Download Policy Document",
                    data=pdf_bytes,
                    file_name=p["filename"],
                    mime="application/pdf"
                )
                
                st.write("") # Spacer
                
                # Render the PDF safely using PDF.js via the new library
                pdf_viewer(pdf_bytes, width=800)
                
            elif p["path"].endswith(".md"):
                with open(p["path"], "r", encoding="utf-8") as f:
                    st.markdown(f.read())

# Voice Dictation Trigger
voice_input_widget()
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