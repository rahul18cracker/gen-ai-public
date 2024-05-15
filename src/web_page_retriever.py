import os 
import streamlit as st
import pickle 
import time
import langchain
from langchain_openai import ChatOpenAI  
from langchain import OpenAI
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import SeleniumURLLoader 
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

api_key = os.environ.get('OPENAI_API_KEY_PERSONAL')

# global vector store  pnly good for POC 

llm = ChatOpenAI(api_key=api_key,max_tokens=500, temperature=0.5)

# some basic structure of the web page
st.title("News Research Tool")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)



process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()
# once button is clicked
if process_url_clicked:
    # load the urls into the loader for processing
    loader = SeleniumURLLoader(urls=urls)
    main_placeholder.text("Data loading started ....")
    data = loader.load()
    # start splitting the urls on escape characters
    main_placeholder.text("Text splitter started ....")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        separators = ['\n\n', '\n', '.', ' ']
    )
    # as data is of type documents we can directly use split_documents over split_text 
    docs = text_splitter.split_documents(data)
    # now we have chunks split so create embeddings for vector DB 
    main_placeholder.text("Embedding vector started ....")
    embeddings = OpenAIEmbeddings(api_key=api_key)
    vector_store = FAISS.from_documents(docs, embeddings)

    # store to disk 
    vector_store.save_local("vectorstore")


query = main_placeholder.text_input("Question: ")
if query:
    vector_store = FAISS.load_local("vectorstore", OpenAIEmbeddings(api_key=api_key), allow_dangerous_deserialization=True)
    # if there is a query get the answer from vector store and feed to it the multi chunk retrival and summarization method 
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_store.as_retriever())
    # return is a dict so use key value methods {"answer": "" , "sources": []}
    answer = chain({"question": query}, return_only_outputs=True)
    st.header("Answer")
    st.write(answer["answer"])
    sources = answer.get("sources", "")
    if sources:
        st.subheader("Sources:")
        sources_list = sources.split("\n")
        for source in sources_list:
            st.write(source)
