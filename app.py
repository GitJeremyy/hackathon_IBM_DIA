import streamlit as st
from datetime import datetime
from source.assistant import school_assistant

# Mock function - replace with actual import: from assistant import school_assistant

# Initialize session state
if 'school_selected' not in st.session_state:
    st.session_state.school_selected = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'show_review' not in st.session_state:
    st.session_state.show_review = False
if 'conversation_closed' not in st.session_state:
    st.session_state.conversation_closed = False
# NEW: flag pour clear l'input au prochain rerun
if 'clear_user_input' not in st.session_state:
    st.session_state.clear_user_input = False

# Page config
st.set_page_config(page_title="HelpAI", page_icon="🎓", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
    }
    .bot-message {
        background-color: #f5f5f5;
        border-left: 5px solid #4caf50;
    }
    .message-label {
        font-weight: bold;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    .message-content {
        font-size: 1rem;
        line-height: 1.6;
    }
    .school-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .esilv-badge {
        background-color: #1976d2;
        color: white;
    }
    .emlv-badge {
        background-color: #d32f2f;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🎓 HelpAI")

# School Selection Phase
if st.session_state.school_selected is None:
    st.markdown("## Welcome! Please select your school:")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🏫 ESILV", use_container_width=True, type="primary"):
            st.session_state.school_selected = "esilv"
            st.rerun()
    
    with col2:
        if st.button("🏢 EMLV", use_container_width=True, type="primary"):
            st.session_state.school_selected = "emlv"
            st.rerun()

    with col3:
        if st.button("🏫 IIM", use_container_width=True, type="primary"):
            st.session_state.school_selected = "iim"
            st.rerun()
    
    with col4:
        if st.button("🏢 Executive", use_container_width=True, type="secondary"):
            st.session_state.school_selected = "executive"
            st.rerun()

# Chat Phase
elif not st.session_state.show_review:
    # Display selected school
    school_name = st.session_state.school_selected.upper()
    badge_class = f"{st.session_state.school_selected}-badge"
    st.markdown(f'<div class="school-badge {badge_class}">{school_name}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Display chat history
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.markdown(f"""
                <div class="chat-message user-message">
                    <div class="message-label">👤 You</div>
                    <div class="message-content">{msg['content']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <div class="message-label">🤖 Assistant</div>
                    <div class="message-content">{msg['content']}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("👋 Hello! How can I help you today?")
    
    # Chat input area (form => Enter triggers submit)
    st.markdown("---")

    # NEW: clear avant de créer le widget (sinon Streamlit lève une exception)
    if st.session_state.pop("clear_user_input", False):
        # soit on supprime la clé...
        if "user_input" in st.session_state:
            del st.session_state["user_input"]
        # ...ou on peut pré-initialiser: st.session_state["user_input"] = ""

    with st.form(key="chat_form"):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.text_input(
                "Type your question:",
                key="user_input",
                label_visibility="collapsed",
                placeholder="Ask me anything about your school..."
            )
        with col2:
            submitted = st.form_submit_button("Send 📤", use_container_width=True, type="primary")
    
    user_input = (st.session_state.get("user_input") or "").strip()
    
    if submitted and user_input:
        # Add user message
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        })

        # Limit context (dernier 8 messages)
        context_slice = st.session_state.chat_history[-2:]

        # Spinner while generating
        with st.spinner("⏳ L'assistant réfléchit..."):
            bot_response = school_assistant(
                question=user_input,
                school=st.session_state.school_selected,
                chat_history=context_slice
            )

        # Add assistant response
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': bot_response,
            'timestamp': datetime.now().isoformat()
        })

        # Clear input au prochain rerun (ne pas modifier le widget maintenant)
        st.session_state.clear_user_input = True
        st.rerun()
    
    # Close and Review button
    st.markdown("---")
    if st.button("🔚 Close Conversation and Review", use_container_width=True, type="secondary"):
        st.session_state.show_review = True
        st.session_state.conversation_closed = True
        st.rerun()

# Review Phase
else:
    st.markdown("## 📝 Review Your Experience")
    st.markdown("Thank you for using our chatbot! Please share your feedback.")
    
    # Star rating
    st.markdown("### Rate your experience:")
    rating = st.radio(
        "Select rating:",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: "⭐" * x,
        horizontal=True,
        label_visibility="collapsed"
    )
    
    # Text review
    st.markdown("### Write your review:")
    review_text = st.text_area(
        "Share your thoughts:",
        placeholder="Tell us about your experience with the assistant...",
        height=150,
        label_visibility="collapsed"
    )
    
    # Display conversation summary
    with st.expander("📋 View Conversation Summary"):
        st.markdown(f"**School:** {st.session_state.school_selected.upper()}")
        st.markdown(f"**Number of messages:** {len(st.session_state.chat_history)}")
        st.markdown(f"**Duration:** {len(st.session_state.chat_history) // 2} questions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Save Review", use_container_width=True, type="primary"):
            # Prepare review data
            review_data = {
                'school': st.session_state.school_selected,
                'rating': rating,
                'review_text': review_text,
                'chat_history': st.session_state.chat_history,
                'timestamp': datetime.now().isoformat()
            }
            
            # Mock save - replace with actual database save
            st.success("✅ Review saved successfully!")
            st.info("🔄 Review data ready for database storage (mock save completed)")
            
            # Show what would be saved
            with st.expander("📊 Review Data (for debugging)"):
                st.json(review_data)
            
            # Option to start new conversation
            if st.button("🔄 Start New Conversation", use_container_width=True):
                st.session_state.school_selected = None
                st.session_state.chat_history = []
                st.session_state.show_review = False
                st.session_state.conversation_closed = False
                st.rerun()
    
    with col2:
        if st.button("❌ Cancel", use_container_width=True, type="secondary"):
            st.session_state.show_review = False
            st.rerun()

# Sidebar with info
with st.sidebar:
    st.markdown("## ℹ️ About")
    st.markdown("""
    This chatbot assists students from:
    - **ESILV** - École Supérieure d'Ingénieurs Léonard de Vinci
    - **EMLV** - École de Management Léonard de Vinci
    
    Ask questions about your school and get instant answers!
    """)
    
    if st.session_state.school_selected:
        st.markdown("---")
        st.markdown(f"**Current School:** {st.session_state.school_selected.upper()}")
        st.markdown(f"**Messages:** {len(st.session_state.chat_history)}")
        
        if st.button("🔄 Reset Conversation", use_container_width=True):
            st.session_state.school_selected = None
            st.session_state.chat_history = []
            st.session_state.show_review = False
            st.session_state.conversation_closed = False
            st.rerun()