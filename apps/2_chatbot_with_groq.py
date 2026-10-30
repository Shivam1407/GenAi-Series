
# AkashVaani: AI Chatbot with Google Search, Memory, and Streaming

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper   #for searching on google
from langchain_core.tools import Tool       #tool that search on google
from langchain.agents import create_agent       #for creating agent for doing extra tasks 
from langgraph.checkpoint.memory import MemorySaver         # for making LLm remembers your conversation


# -------------------- LLM --------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    streaming=True,
    temperature=0.3
)


# -------------------- Google Search Tool --------------------

search = GoogleSerperAPIWrapper()    #it is a object which contains tool

search_tool = Tool(
    name="google_search",
    description=(
        "Search Google for current information, facts, "
        "news, and answers to questions that need web research."
    ),
    func=search.run,
)

tools = [search_tool]


# -------------------- Memory --------------------

if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()

if "history" not in st.session_state:
    st.session_state.history = []


# -------------------- Agent --------------------

agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=st.session_state.memory,
    system_prompt=(
        "You are AkashVaani, a helpful AI assistant. "
        "Answer clearly and accurately. "
        "Use Google Search when current or external information "
        "is needed. If you are uncertain, say so honestly."
    ),
)


# -------------------- Streamlit Interface --------------------

st.title("AkashVaani")
st.caption("Your AI chatbot with Google Search and memory")


# Display previous messages
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User input
query = st.chat_input("Ask anything...")

if query:
    # Display user message
    st.session_state.history.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("user"):
        st.markdown(query)

    # Stream the AI response
    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""

        try:
            response_stream = agent.stream(
                {
                    "messages": [
                        {"role": "user", "content": query}
                    ]
                },
                config={
                    "configurable": {
                        "thread_id": "akashvaani_user"
                    }
                },
                stream_mode="messages",
            )

            for chunk, metadata in response_stream:
                # Only display text from the assistant,
                # not intermediate tool-call chunks.
                if (
                    metadata.get("langgraph_node") == "model"
                    and chunk.content
                    and isinstance(chunk.content, str)
                ):
                    full_response += chunk.content
                    response_container.markdown(
                        full_response + "▌"
                    )

            response_container.markdown(full_response)

        except Exception as e:
            st.error(f"Something went wrong: {e}")

        # Save the final assistant response
        if full_response:
            st.session_state.history.append(
                {
                    "role": "assistant",
                    "content": full_response,
                }
            )
