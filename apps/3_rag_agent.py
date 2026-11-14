import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False
if "agent" not in st.session_state:
    st.session_state.agent = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []

def process_documents(path):
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = InMemoryVectorStore.from_documents(docs, embeddings)

    st.session_state.vector_store = vector_store

    llm = ChatGroq(model="mixtral-8x7b-32768")

    @tool
    def retrieve_context(query: str) -> str:
        """You are a helpful assistant that answers questions using retrieved context.

    My knowledge base consists of details from the uploaded document.
    Always use the retrieve_context tool for questions requiring external knowledge.
        """
        context = ""
        results = st.session_state.vector_store.similarity_search(query, k=3)
        for doc in results:
            context += doc.page_content + "\n"
        return context

    system_prompt = "You are a helpful assistant that answers questions using retrieved context. My knowledge base consists of details from the uploaded document. Always use the retrieve_context tool for questions requiring external knowledge."

    memory = InMemorySaver()
    agent = create_react_agent(
        model=llm,
        tools=[retrieve_context],
        prompt=system_prompt,
        checkpointer=memory
    )
    
    st.session_state.agent = agent
    st.session_state.document_uploaded = True

if not st.session_state.document_uploaded:
    uploaded_files = st.file_uploader("Select PDF Files", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        with st.spinner("Processing..."):
            path = "docs_files/"
            os.makedirs(path, exist_ok=True)
            for file in uploaded_files:
                with open(os.path.join(path, file.name), "wb") as f:
                    f.write(file.getvalue())
            process_documents(path)
        st.rerun()

if st.session_state.document_uploaded and st.session_state.agent:
    query = st.chat_input("Ask anything related to uploaded documents")
    
    for message in st.session_state.messages:
        role = message.get("role")
        content = message.get("content")
        with st.chat_message(role):
            st.markdown(content)
            
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        response = st.session_state.agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"configurable": {"thread_id": "1"}}
        )
        
        answer = response["messages"][-1].content
        st.session_state.messages.append({"role": "ai", "content": answer})
        
        with st.chat_message("ai"):
            st.markdown(answer)