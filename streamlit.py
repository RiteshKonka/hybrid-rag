
from langchain.document_loaders import PyPDFDirectoryLoader,DirectoryLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import openai
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI

import streamlit as st
import os
import sys
import tempfile
from pathlib import Path
from streamlit_option_menu import option_menu

# Adjusted paths based on the directory structure
TEMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'temp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')

header = st.container()



def streamlit_ui():
    with st.sidebar: 
        choice = option_menu('Navigation', ['Home', 'Chat with Document/RAG'], 
                             icons=['house', 'chat'], 
                             menu_icon="cast", default_index=0) 
    
    if choice == 'Home':
        st.title("You can chat with documents by clicking on Chat with document/RAG")

    elif choice == 'Chat with Document/RAG': 
        with header:
            st.title('Simple RAG with vector')  
            st.write("""This is a simple RAG process where user will upload a document then the document     
                     will go through RecursiveCharacterSplitter and be embedded in FAISS DB""")    
            
            source_docs = st.file_uploader(label="Upload a document", type=['pdf'], accept_multiple_files=True)
            if not source_docs:
                st.warning('Please upload a document')
            else:
                query = st.chat_input()
                print(query)
                RAG(source_docs,query)
                st.success('Document(s) uploaded successfully!')


def RAG(docs):
    #load the document
    for source_docs in docs:
        with tempfile.NamedTemporaryFile(delete=False,dir=TEMP_DIR.as_posix(),suffix='.pdf') as temp_file:
            temp_file.write(source_docs.read())

    
    loader = DirectoryLoader(TEMP_DIR.as_posix(), glob='**/*.pdf', show_progress=True)
    documents = loader.load()

    #Split the document
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text = text_splitter.split_documents(documents)


    #Vector and embeddings
    DB_FAISS_PATH = 'vectorestore/faiss'
    embedding = HuggingFaceEmbeddings(model_name ='sentence-transformers/all-MiniLM-L6-v2',
                                         model_kwargs={'device':'cpu'})
    
    db = FAISS.from_documents(text,embedding)
    db.save_local(DB_FAISS_PATH)

    #Setup LLM, Fetch base url from LM Studio
    llm = ChatOpenAI(base_url="http://localhost:1234/v1",api_key='lm-studio')

    #Build a conversational chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        db.as_retriever(search_kwargs={'k':2}),
        return_source_documents=True
    )

    chat_history = []
    #Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages =[]
    
    #Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    #React to user input
    if prompt := st.chat_input("Ask question to document assistant"):
        #Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        #Add user message to chat history
        st.session_state.messages.append({"role":"user","context":prompt})

        response = f"Echo: {prompt}"
        #Display assistant response in chat message container
        response = qa_chain({'question':prompt,'chat_history':chat_history})

        with st.chat_message("assistant"):
            st.markdown(response['answer'])
        
        st.session_state.messages.append({'role':"assistant", "content":response})
        chat_history.append({prompt,response['answer']})


if __name__ == "__main__":
    streamlit_ui()


