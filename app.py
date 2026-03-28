import streamlit as st
from PyPDF2 import PdfReader
import docx
import pandas as pd

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory


# ---------- API KEY ----------
open_ai_key = st.secrets.get("OPENAI_API_KEY", None)


# ---------- FILE TEXT EXTRACTION ----------
def get_files_text(uploaded_files):
    text = ""

    for file in uploaded_files:

        if file.name.endswith(".pdf"):
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted

        elif file.name.endswith(".docx"):
            doc = docx.Document(file)
            for para in doc.paragraphs:
                text += para.text + "\n"

        elif file.name.endswith(".csv"):
            df = pd.read_csv(file)
            text += df.to_string(index=False) + "\n"

        else:
            st.warning(f"{file.name} format not supported")

    return text


# ---------- TEXT CHUNKS ----------
def get_text_chunks(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_text(text)


# ---------- VECTOR STORE ----------
def get_vectorstore(text_chunks):

    embeddings = OpenAIEmbeddings(
        api_key=open_ai_key,
        model="text-embedding-3-small"
    )

    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

    return vectorstore


# ---------- CONVERSATION CHAIN ----------
def get_conversation_chain(vectorstore):

    llm = ChatOpenAI(
        api_key=open_ai_key,
        model="gpt-4o-mini",
        temperature=0
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return chain


# ---------- CHAT HANDLER ----------
def handle_userinput(user_question):

    response = st.session_state.conversation(
        {'question': user_question}
    )

    st.session_state.chat_history = response['chat_history']

    for i, msg in enumerate(st.session_state.chat_history):

        if i % 2 == 0:
            st.chat_message("user").write(msg.content)
        else:
            st.chat_message("assistant").write(msg.content)


# ---------- MAIN APP ----------
def main():

    st.set_page_config(page_title="Document GPT")

    # ---------- BUTTON STYLE ----------
    st.markdown("""
    <style>

    .stButton>button{
        background-color:#2563eb !important;
        color:white !important;
        border-radius:10px;
        font-weight:600;
    }

    .stButton>button:hover{
        background-color:#1d4ed8 !important;
    }

    </style>
    """, unsafe_allow_html=True)

    st.header("📄 Document GPT")

    # ---------- SESSION STATES ----------
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = False

    # ---------- SIDEBAR ----------
    with st.sidebar:

        st.subheader("Upload Documents")

        uploaded_files = st.file_uploader(
            "Upload files:",
            type=['pdf', 'docx', 'csv'],
            accept_multiple_files=True
        )

        process = st.button("⚡ Prepare Documents")

    # ---------- PROCESS FILES ----------
    if process:

        if not open_ai_key:
            st.warning("⚠️ Add OpenAI API key in secrets.toml")
            st.stop()

        if not uploaded_files:
            st.warning("⚠️ Upload files first.")
            st.stop()

        with st.spinner("Processing documents..."):

            files_text = get_files_text(uploaded_files)

            chunks = get_text_chunks(files_text)

            vectorstore = get_vectorstore(chunks)

            st.session_state.conversation = get_conversation_chain(vectorstore)

            st.session_state.processComplete = True

        st.success("✅ Documents ready. You can now chat.")

    # ---------- CHAT ----------
    user_question = st.chat_input("Ask something about your files")

    if user_question:

        if not open_ai_key:
            st.warning("⚠️ Add your OpenAI API key.")
            st.stop()

        if not st.session_state.processComplete:
            st.warning("⚠️ Upload files and click 'Prepare Documents' first.")
            st.stop()

        handle_userinput(user_question)


# ---------- RUN APP ----------
if __name__ == "__main__":
    main()