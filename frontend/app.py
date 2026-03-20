import streamlit as st
import requests

st.title("Chat App")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ]

for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        status_placeholder = st.empty()

        full_response = ""
        event_type = None 

        response = requests.post(
            "http://127.0.0.1:8000/chat",
            json={"messages": st.session_state.messages},
            stream=True
        )

        try:
            for line in response.iter_lines():
                if line:
                    decoded = line.decode("utf-8")

                    if decoded.startswith("event:"):
                        event_type = decoded.replace("event: ", "")
                        continue

                    if decoded.startswith("data:"):
                        data = decoded.replace("data: ", "")

                        if event_type == "status":
                            status_placeholder.info(data)

                        elif event_type == "token":
                            full_response += data
                            message_placeholder.markdown(full_response)

                        elif event_type == "done":
                            status_placeholder.empty()
                            break

                        elif event_type == "error":
                            st.error(data)
                            break

        except requests.exceptions.ChunkedEncodingError:
            st.warning("Connection interrupted. Partial response shown.")

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )