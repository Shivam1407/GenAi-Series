
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

st.title("AskBuddy: AI QNA Bot")
st.markdown("My QnA Bot with LangChain and Google Gemini!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get new question
query = st.chat_input("Ask anything?")

if query:
    # Save and display user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    # Send conversation history to Gemini
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            res = llm.invoke(st.session_state.messages)
            answer = res.content
            st.markdown(answer)

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
