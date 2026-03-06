import streamlit as st
import uuid
from langchain.schema import HumanMessage
from chatbot_with_ui_backend import chatbot

# ---------------------------
# Helpers
# ---------------------------

def generate_thread_id():
    return str(uuid.uuid4())


def generate_title_from_message(message, max_words=6):
    words = message.strip().split()
    return " ".join(words[:max_words])


def clear_chat():
    new_thread_id = generate_thread_id()
    st.session_state.thread_id = new_thread_id
    st.session_state.message_history = []


def add_thread(thread_id, first_message):
    for t in st.session_state.chat_threads:
        if t["id"] == thread_id:
            return

    st.session_state.chat_threads.append({
        "id": thread_id,
        "title": generate_title_from_message(first_message)
    })


def load_conversation(thread_id):
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    return state.values.get("messages", [])


# ---------------------------
# Session State Init
# ---------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()

if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = []


# ---------------------------
# Sidebar
# ---------------------------

st.sidebar.title("💬 Chatbot")

if st.sidebar.button("➕ New Chat"):
    clear_chat()
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("My Conversations")

for thread in st.session_state.chat_threads:
    if st.sidebar.button(thread["title"], key=thread["id"]):
        st.session_state.thread_id = thread["id"]

        messages = load_conversation(thread["id"])
        temp = []

        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp.append({"role": role, "content": msg.content})

        st.session_state.message_history = temp
        st.rerun()


# ---------------------------
# Chat UI
# ---------------------------

for msg in st.session_state.message_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Type your message here...")

# ---------------------------
# Chat Logic
# ---------------------------

if user_input:
    # Create thread title on first message
    if len(st.session_state.message_history) == 0:
        add_thread(st.session_state.thread_id, user_input)

    # Show user message
    st.session_state.message_history.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # LangChain streaming
    CONFIG = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }

    stream = chatbot.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=CONFIG,
        stream_mode="messages"
    )

    with st.chat_message("assistant"):
        assistant_text = ""
        placeholder = st.empty()

        for chunk, _ in stream:
            if chunk.content:
                assistant_text += chunk.content
                placeholder.markdown(assistant_text)

    st.session_state.message_history.append({
        "role": "assistant",
        "content": assistant_text
    })
