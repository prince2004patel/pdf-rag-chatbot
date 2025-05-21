import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq

# Load environment
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Title and UI
st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")
st.title("📄 Conversational RAG Chatbot with PDF Uploads and Chat History")

# API key input
api_key = st.text_input("🔑 Enter your Groq API key:", type="password")

# Session ID
session_id = st.text_input("🆔 Session ID", value="default_session")

# In-memory chat store
if "store" not in st.session_state:
    st.session_state.store = {}

# File uploader
uploaded_files = st.file_uploader("📁 Upload PDF file(s)", type="pdf", accept_multiple_files=True)

# Continue only if API key and files are uploaded
if api_key and uploaded_files:

    # Load and parse PDFs
    documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        documents.extend(loader.load())

    # Text splitting and embedding
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(documents)

    # Embedding model and local chroma vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    persist_directory = "chroma_store"
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    retriever = vectorstore.as_retriever()

    # Load LLM
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

    # Prompt for question reformulation (history-aware)
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the chat history and user question, reformulate it into a standalone question. If unnecessary, return as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm=llm, retriever=retriever, prompt=contextualize_q_prompt
    )

    # Prompt for final answering
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant. Use the context to answer questions in 3 sentences max. If not in context, say you don't know.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Manage chat history
    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    # Combine chain with memory
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # Chat UI
    user_input = st.text_input("💬 Ask your question:")
    if user_input:
        history = get_session_history(session_id)
        with st.spinner("Generating answer..."):
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
        st.markdown("### 🤖 Assistant:")
        st.success(response["answer"])
        st.markdown("## 🧠 Chat History")

        with st.expander("Click to view full chat history", expanded=False):
            for i in range(0, len(history.messages), 2):
                # Group every human + ai message as a pair
                if i + 1 < len(history.messages):
                    user_msg = history.messages[i]
                    ai_msg = history.messages[i + 1]

                    if user_msg.type == "human" and ai_msg.type == "ai":
                        with st.container():
                            st.markdown(
                                """
                                <div style="background-color: #1e1e1e; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                                <p style="color: #ffd700;"><b>🧑 You:</b> {}</p>
                                <p><b>🤖 Assistant:</b> {}</p>
                                </div>
                                """.format(user_msg.content, ai_msg.content),
                                unsafe_allow_html=True
                            )


# Handle empty inputs
elif not api_key:
    st.warning("⚠️ Please enter your Groq API key to proceed.")
elif not uploaded_files:
    st.info("📎 Please upload at least one PDF file.")
