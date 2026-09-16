"""
app.py — Phase 6: Streamlit Interactive UI

This is the interactive front-end for the Apple Support AI Agent.
It provides a premium chat interface and an "Under the Hood" sidebar
to show reviewers exactly how the intent classification, RAG retrieval,
and escalation logic work in real-time.

To run:
    pip install streamlit
    streamlit run app.py
"""

import streamlit as st
import logging
import os
import sys

# Ensure we can import from src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agent import run_agent

# ──────────────────────────────────────────────
# PAGE CONFIG & PREMIUM STYLING
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Apple Support AI Agent",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Apple-inspired dark mode styling with gradients
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0a0c 0%, #1a1a24 100%);
        color: #f5f5f7;
    }
    
    /* Header typography */
    h1 {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #fff, #a1a1a6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }
    
    /* Subtle glowing divider */
    hr {
        border: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    }
    
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# APP STATE INITIALIZATION
# ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am the AI Support Agent prototype. Send me a simulated customer tweet, and I will classify the intent, retrieve historical solutions, and draft a response."}
    ]

if "last_agent_state" not in st.session_state:
    st.session_state.last_agent_state = None

# ──────────────────────────────────────────────
# SIDEBAR: UNDER THE HOOD
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Under the Hood")
    st.markdown("Transparency into the Agent's decision making.")
    
    state = st.session_state.last_agent_state
    if state:
        # Intent Classification
        st.markdown("### 🎯 Intent Classification")
        st.code(state["intent"], language=None)
        st.metric("Confidence", f"{state['intent_confidence']:.0%}")
        st.caption(state["intent_reasoning"])
        
        # Escalation Decision
        text = "ESCALATE TO HUMAN" if state['escalate'] else "AUTO-HANDLE"
        st.markdown("### ⚠️ Escalation Decision")
        # A bare conditional expression is auto-rendered by Streamlit's magic
        # handler, exposing the returned DeltaGenerator object's internals.
        if state["escalate"]:
            st.error(text)
        else:
            st.success(text)
        st.write(state["escalation_reason"])
        st.caption(f"Method: {state['escalation_method']}")
        
        # RAG Retrieval
        st.markdown("### 📚 Retrieved Context (RAG)")
        with st.expander("View Top Similar Historical Cases"):
            for i, ex in enumerate(state['retrieved_examples'][:3], 1):
                st.markdown(f"**Example {i}** (Similarity: `{ex['similarity']:.3f}`)")
                st.markdown(f"**Customer:** _{ex['customer_message']}_")
                st.markdown(f"**Apple Reply:** {ex['apple_reply']}")
                st.divider()
    else:
        st.info("Send a message to see the agent's internal state.")

# ──────────────────────────────────────────────
# MAIN CHAT INTERFACE
# ──────────────────────────────────────────────
st.title("🍎 Apple Support AI Prototype")
st.markdown("An end-to-end agentic RAG pipeline built for Hiver.")
st.markdown("<hr>", unsafe_allow_html=True)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new user input
if prompt := st.chat_input("Type a simulated customer tweet (e.g. 'My battery is dying fast')"):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent Processing
    with st.chat_message("assistant"):
        with st.spinner("Classifying intent, retrieving context, and drafting reply..."):
            try:
                # Run the pipeline!
                result = run_agent(prompt, verbose=False)
                
                # Save state for sidebar
                st.session_state.last_agent_state = result
                
                # Format final reply
                reply = result["draft_reply"]
                if result["escalate"]:
                    reply = f"🚨 **[SYSTEM: ESCALATED TO HUMAN]**\n\n**Suggested Draft:**\n{reply}"
                
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
                
                # Force a rerun to update the sidebar instantly
                st.rerun()
                
            except Exception:
                logging.exception("Agent UI request failed")
                st.error("The agent could not process this request. Check the terminal logs and setup instructions.")
