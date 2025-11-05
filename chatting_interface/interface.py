import streamlit as st
import json
from datetime import datetime
from pathlib import Path
import sys

# Import the RAG script
try:
    from Rag_completion import complete_rag
except ImportError:
    st.error("rag.py file not found. Please ensure it exists in the same directory.")
    sys.exit(1)

# Initialize session state variables
if 'school' not in st.session_state:
    st.session_state.school = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'conversation_ended' not in st.session_state:
    st.session_state.conversation_ended = False
if 'rating_submitted' not in st.session_state:
    st.session_state.rating_submitted = False
if 'session_id' not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

def save_conversation_log(rating=None):
    """Save conversation history and rating to a JSON file"""
    log_dir = Path("conversation_logs")
    log_dir.mkdir(exist_ok=True)
    
    log_data = {
        "session_id": st.session_state.session_id,
        "timestamp": datetime.now().isoformat(),
        "school": st.session_state.school,
        "messages": st.session_state.messages,
        "rating": rating
    }
    
    log_file = log_dir / f"conversation_{st.session_state.session_id}.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

def reset_conversation():
    """Reset the conversation to start a new session"""
    st.session_state.school = None
    st.session_state.messages = []
    st.session_state.conversation_ended = False
    st.session_state.rating_submitted = False
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# Main UI
st.title("🎓 School Chatbot Assistant")

# School Selection Phase
if st.session_state.school is None:
    st.markdown("### Welcome! Please select your school to begin:")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🏫 ESILV", use_container_width=True, type="primary"):
            st.session_state.school = "ESILV"
            st.rerun()
    with col2:
        if st.button("🏢 EMLV", use_container_width=True, type="primary"):
            st.session_state.school = "EMLV"
            st.rerun()

# Chat Phase
elif not st.session_state.conversation_ended:
    st.markdown(f"**Selected School:** {st.session_state.school}")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask your question..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get RAG response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = complete_rag(prompt, st.session_state.school)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # End conversation button
    st.markdown("---")
    if st.button("End Conversation", type="secondary"):
        st.session_state.conversation_ended = True
        st.rerun()

# Rating Phase
elif not st.session_state.rating_submitted:
    st.markdown("### Thank you for using our chatbot!")
    st.markdown("Please rate your experience:")
    
    rating = st.radio(
        "How satisfied are you with this interaction?",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: "⭐" * x,
        horizontal=True
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Submit Rating", type="primary", use_container_width=True):
            save_conversation_log(rating)
            st.session_state.rating_submitted = True
            st.rerun()
    with col2:
        if st.button("Skip", use_container_width=True):
            save_conversation_log(None)
            st.session_state.rating_submitted = True
            st.rerun()

# Thank You Phase
else:
    st.success("✅ Thank you for your feedback!")
    st.markdown("Your conversation has been saved.")
    
    if st.button("Start New Conversation", type="primary"):
        reset_conversation()
        st.rerun()

# Sidebar
with st.sidebar:
    st.markdown("### About")
    st.markdown("This chatbot uses RAG to answer your questions about ESILV and EMLV.")
    
    if st.session_state.school:
        st.markdown(f"**Current School:** {st.session_state.school}")
        st.markdown(f"**Messages:** {len(st.session_state.messages)}")
    
    st.markdown("---")
    st.markdown("**Session ID:**")
    st.code(st.session_state.session_id, language=None)