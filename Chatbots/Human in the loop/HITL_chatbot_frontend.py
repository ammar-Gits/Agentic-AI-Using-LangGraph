import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from HITL_chatbot_backend import chatbot

# -------------------------
# 1. Session Setup
# -------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit-thread"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = None

st.set_page_config(page_title="Stock Trading Bot", layout="centered")
st.title("📈 AI Stock Trading Assistant")

# -------------------------
# 2. Display Chat History
# -------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------
# 3. Handle Interrupt (HITL)
# -------------------------
if st.session_state.pending_interrupt:
    st.warning(st.session_state.pending_interrupt)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Approve"):
            result = chatbot.invoke(
                Command(resume="yes"),
                config={"configurable": {"thread_id": st.session_state.thread_id}},
            )

            st.session_state.pending_interrupt = None

            last_msg = result["messages"][-1]
            st.session_state.messages.append(
                {"role": "assistant", "content": last_msg.content}
            )
            st.rerun()

    with col2:
        if st.button("❌ Reject"):
            result = chatbot.invoke(
                Command(resume="no"),
                config={"configurable": {"thread_id": st.session_state.thread_id}},
            )

            st.session_state.pending_interrupt = None

            last_msg = result["messages"][-1]
            st.session_state.messages.append(
                {"role": "assistant", "content": last_msg.content}
            )
            st.rerun()

# -------------------------
# 4. Chat Input
# -------------------------
if prompt := st.chat_input("Ask about a stock or buy shares..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run graph
    result = chatbot.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config={"configurable": {"thread_id": st.session_state.thread_id}},
    )

    interrupts = result.get("__interrupt__", [])

    if interrupts:
        # Save interrupt prompt and wait for approval
        st.session_state.pending_interrupt = interrupts[0].value
        st.rerun()
    else:
        # Normal assistant response
        last_msg = result["messages"][-1]

        st.session_state.messages.append(
            {"role": "assistant", "content": last_msg.content}
        )

        with st.chat_message("assistant"):
            st.markdown(last_msg.content)