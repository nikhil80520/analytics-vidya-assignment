import streamlit as st
import requests
import json
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Python Q&A Assistant", page_icon="🐍", layout="wide")

st.title("🐍 Python Programming Q&A Assistant")
st.markdown("Ask any Python-related question and get answers grounded in Stack Overflow data!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources if they exist for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("View Sources"):
                for idx, source in enumerate(message["sources"]):
                    st.markdown(f"**[{idx+1}] {source.get('title', 'Unknown')}**")
                    st.markdown(f"*Score: {source.get('score', 0)} | Answer Score: {source.get('answer_score', 0)}*")
                    st.caption(f"{source.get('preview', '')}")
                    st.divider()

# React to user input
if prompt := st.chat_input("E.g., How do I reverse a list in Python?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Check API health first (optional, but good for UX)
    with st.spinner("Generating answer..."):
        try:
            response = requests.post(f"{API_URL}/ask", json={"question": prompt}, timeout=120)
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "No answer generated.")
                sources = data.get("sources", [])
                
                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    
                    if sources:
                        with st.expander("View Sources"):
                            for idx, source in enumerate(sources):
                                st.markdown(f"**[{idx+1}] {source.get('title', 'Unknown')}**")
                                st.markdown(f"*Score: {source.get('score', 0)} | Answer Score: {source.get('answer_score', 0)}*")
                                st.caption(f"{source.get('preview', '')}")
                                st.divider()
                                
                    # Metrics info
                    st.caption(f"Retrieved {data.get('retrieved_chunks', 0)} chunks in {data.get('response_time_ms', 0)}ms using {data.get('llm_used', 'unknown model')}")
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })
            else:
                error_msg = f"API Error ({response.status_code}): {response.text}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to connect to API: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
