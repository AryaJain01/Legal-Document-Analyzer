import streamlit as st 
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

## upload pdf uisng tempfile so that when loader used that file it automatically deleted
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.read())
        temp_path = f.name
    
    ## upload the file here using pypdf uploader  
    loader=PyPDFLoader(temp_path)  
    pages=loader.load()
    
    ## now split the pdf into chunks with this default parameters    
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200)
    
    chunks=splitter.split_documents(pages)
    return chunks ,len(pages),len(chunks)
    

## create vectors firstly we do embedding in which all chunks converted into vectors 
def create_vector(chunks):
    #Load embedding model
    embeddings=HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
    )    
    ## then we'll do FAISS so that it can search exponentially fast as embedding vectors are very larget without vectore store it is not possible to find the similarity between vectors
    vectore_store=FAISS.from_documents(chunks,embeddings)
    return vectore_store


## now firstly we load our dot env and then we create a llm model using groq we loaded this beacuse in my .env file there is my api key also make sure that temprature is 0.3 cause it will give mostly factual not creative
load_dotenv()
def load_llm():
    llm=ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )
    return llm
     
## created prompt template  then it will be connected to RAG 
def create_prompt():
    prompt=ChatPromptTemplate.from_template("""
    You are an expert legal document analyzer with years of experience.
    
    Use ONLY the following context from the legal document to answer the question.
    Do not use any outside knowledge.
    If you cannot find the answer in the context, say exactly:
    "I couldn't find this information in the document."
                                              
    Context:
    {context}
    
    Question: {question}
    
    Provide a clear and structured answer:
                              
    """)
    return prompt  

## create RAG chain and what it will do firstly get top 3 most similar faiss from the vector store 
## then we initialize llm , prompt and chain using LCEL chain firstly llm model need context which it will get from retriever and then question need to go untouch that's why we used runnablepassthrough
##it will not touch pass it as it is and then | this symbol work as pipeline we then move to model then we go to prompt filled by these and then with outputPraser we simply return the text or vector into string
def create_RAG_chain(vector_store):
    retriever=vector_store.as_retriever(
        search_kwargs={"k":3}
    )
    llm=load_llm()
    prompt=create_prompt()
    chain=(
        {
            "context":retriever,"question":RunnablePassthrough()
        }
        |prompt
        |llm
        |StrOutputParser()
    )
    return chain

st.set_page_config(
    page_title="Legal Document Analyzer",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ AI Powered Legal Document Analyzer")
st.markdown("Upload a legal document and ask questions about it!")

## with the help of this sliderbar i can upload pdf 
with st.sidebar:
    st.header("Upload Document")
    uploaded_file=st.file_uploader(label="Upload your legal document",type=["pdf"])
    
    if uploaded_file is not None:
        st.success(f"✅File uploaded succesfully : {uploaded_file.name}")
        ## when pdf is being processed it should not behave like frozen page it should show file is being processed and when it is completed it automatcially closed
        with st.spinner("Processing pdf..."):
            chunks,num_pages,num_chunks=process_pdf(uploaded_file)
            
            st.info(f"📃 Pages: {num_pages}")
            st.info(f"🔢 Chunks: {num_chunks}")
            
        with st.spinner("Creating vector store..."):
            vector_store=create_vector(chunks)
            st.session_state.vector_store=vector_store
            
            st.success("✅ Vector store created!")

## chat history so that it can remember chats
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ Main area is OUTSIDE sidebar
if "vector_store" not in st.session_state:
    st.info("👈 Please upload a legal document from the sidebar to get started!")      
    
else:
    st.header("💬 Ask Questions About Your Document")
    
    # Display previous chat history
    for chat in st.session_state.chat_history:
        st.chat_message("user").write(chat["question"])
    ##  st.chat_message("assistant").write(chat["answer"])
    
    question = st.text_input(
        "Ask anything about the document",
        placeholder="e.g. What are the payment terms?"
    )
    
    if question:
        with st.spinner("Analyzing the document..."):
            chain = create_RAG_chain(st.session_state.vector_store)
            answer = chain.invoke(question)
        
        # Save to chat history
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer
        })
        
        # Display latest answer
        st.chat_message("user").write(question)
        st.chat_message("assistant").write(answer)