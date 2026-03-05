import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
from app.rag.query_engine import hybrid_query


st.set_page_config(
    page_title="Hybrid RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Hybrid RAG AI Assistant")
st.write("Ask questions from documents, LLM knowledge, or the internet.")


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


prompt = st.chat_input("Ask something...")


if prompt:

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            answer = hybrid_query(prompt)

            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )