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
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load clause weights
# Moved this up so it's accessible globally for the dictionary
try:
    clause_df = pd.read_csv("clause_weights.csv")
    clause_df.columns = ['clause_short', 'weight'] 
    # Clean clause names
    clause_df['clause_clean'] = clause_df['clause_short'].apply(
        lambda x: x.split('that')[0].strip()
    )
    clause_weights_dict = dict(zip(clause_df['clause_clean'], clause_df['weight']))
except Exception as e:
    st.error(f"Error loading clause_weights.csv: {e}")
    clause_weights_dict = {}

@st.cache_resource
def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

@st.cache_resource
def load_clause_vectors(_model, _standard_clauses):
    return {
        k: _model.encode(v) for k, v in _standard_clauses.items()
    }

## here it is the function which will calculate the risk ;{standard_clause-detected_clause=missing_clause} risk=missing_clause/total_weight_clause *100
## we initialize the missing_weight as 0 and empty list then add iteratively and then we calculate the validity if over 60 it is valid other wise risky
def calculate_risk(detected_clauses, clause_weights_dict):
    detected_names = [c["clause"] for c in detected_clauses]

    total_weight = sum(clause_weights_dict.values())
    if total_weight == 0: return {"risk_score": 0, "validity_score": 100, "missing_clauses": [], "verdict": "Valid ✅"}
    
    missing_weight = 0
    missing_clauses = []

    for clause, weight in clause_weights_dict.items():
        if clause not in detected_names:
            missing_weight += weight
            missing_clauses.append(clause)

    risk_score = (missing_weight / total_weight) * 100
    validity_score = 100 - risk_score

    return {
        "risk_score": round(risk_score, 2),
        "validity_score": round(validity_score, 2),
        "missing_clauses": missing_clauses,
        "verdict": "Valid ✅" if validity_score >= 60 else "Risky ⚠️"
    }

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
    # Cleanup temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)
    return chunks, len(pages), len(chunks)

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


st.markdown("""
    <p style='color: #888; font-size: 1.1rem;'>
    Upload any legal document and instantly get validity analysis, 
    risk scoring, and AI powered Q&A
    </p>
""", unsafe_allow_html=True)
st.set_page_config(
    page_title="Legal Document Analyzer",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ AI Powered Legal Document Analyzer")
#st.markdown("Upload a legal document and ask questions about it!")

## with the help of this sliderbar i can upload pdf 
with st.sidebar:
    st.header("Upload Document")
    uploaded_file=st.file_uploader(label="Upload your legal document",type=["pdf"])
    
    if uploaded_file is not None:
        if "file_id" not in st.session_state or st.session_state.file_id != uploaded_file.name:
            st.success(f"✅File uploaded succesfully : {uploaded_file.name}")
            ## when pdf is being processed it should not behave like frozen page it should show file is being processed and when it is completed it automatcially closed
            with st.spinner("Processing pdf..."):
                chunks,num_pages,num_chunks=process_pdf(uploaded_file)
                st.session_state.chunks = chunks
                st.session_state.num_pages = num_pages
                st.session_state.num_chunks = num_chunks
                
            with st.spinner("Creating vector store..."):
                vector_store=create_vector(chunks)
                st.session_state.vector_store=vector_store
                st.session_state.file_id = uploaded_file.name
                
            st.success("✅ Vector store created!")
        
        if "num_pages" in st.session_state:
            st.info(f"📃 Pages: {st.session_state.num_pages}")
            st.info(f"🔢 Chunks: {st.session_state.num_chunks}")

## chat history so that it can remember chats
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ Main area is OUTSIDE sidebar
if "vector_store" not in st.session_state:
    st.info("👈 Please upload a legal document from the sidebar to get started!")      
    
else:
    # this is outside the block because i want to tell the risk immidately when user upload the documnet not when if user ask the question about it
    chunks = st.session_state.get("chunks", [])
    chunk_texts = [doc.page_content for doc in chunks]

    if "chunk_vectors" not in st.session_state:
        if not chunk_texts:
            st.warning("No document data found. Please upload again.")
            st.stop()
        with st.spinner("Analyzing document risk..."):
            st.session_state.chunk_vectors = model.encode(chunk_texts)

    chunk_vectors = st.session_state.chunk_vectors

    # here we define the standard clause note we are not defining the text we are defining the meaning
    # simple clause definitions (for now)
    standard_clauses = {
    "Document Name": "service agreement contract title name",
    "Parties": "client service provider between parties company individual",
    "Agreement Date": "date agreement made signed january 2024",
    "Governing Law": "governed laws india jurisdiction court",
    "Expiration Date": "expiration end date contract expires",
    "Effective Date": "effective start date contract begins",
    "Anti-Assignment": "assignment transfer rights obligations",
    "Cap On Liability": "liability limit damages maximum cap",
    "License Grant": "license rights usage permission grant",
    "Audit Rights": "audit inspection review records",
    "Termination For Convenience": "terminate agreement notice written",
    "Post-Termination Services": "after termination obligations services",
    "Exclusivity": "exclusive restriction competition",
    "Renewal Term": "renewal extension period term",
    "Insurance": "insurance coverage liability",
    "Revenue/Profit Sharing": "revenue payment sharing profit",
    "Minimum Commitment": "minimum commitment obligation required",
    "Non-Transferable License": "non transferable license cannot assign",
    "Ip Ownership Assignment": "intellectual property ownership rights",
    "Change Of Control": "change control merger acquisition",
    "Non-Compete": "non compete restriction competition",
    "Uncapped Liability": "unlimited liability no cap damages",
    "Notice Period To Terminate Renewal": "notice period days written termination",
    "Covenant Not To Sue": "agreement not to sue legal claims",
    "Rofr/Rofo/Rofn": "right of first refusal offer negotiation",
    "Volume Restriction": "volume restriction limits quantity",
    "Competitive Restriction Exception": "exceptions competition restrictions",
    "Warranty Duration": "warranty period guarantee terms",
    "Irrevocable Or Perpetual License": "perpetual license irrevocable rights",
    "Liquidated Damages": "damages penalty compensation predefined",
    "Affiliate License-Licensee": "affiliate license granted licensee",
    "No-Solicit Of Employees": "no solicitation employees hiring",
    "Joint Ip Ownership": "joint ownership intellectual property shared",
    "Non-Disparagement": "non disparagement no negative statements",
    "No-Solicit Of Customers": "no solicitation customers restriction",
    "Third Party Beneficiary": "third party beneficiary rights",
    "Most Favored Nation": "most favored nation equal treatment",
    "Affiliate License-Licensor": "affiliate license licensor rights",
    "Unlimited/All-You-Can-Eat-License": "unlimited usage license unrestricted",
    "Price Restrictions": "pricing restrictions control limits",
    "Source Code Escrow": "source code escrow deposit release",
    "Payment Terms": "payment due advance milestone delivery amount INR",
    "Confidentiality": "confidential secret information disclose maintain",
    "Late Payment Penalty": "penalty late payment delayed interest per month",
    "Scope Of Services": "scope services software development consulting deliver",
}
    
    #convert this clause into vector so that in both side standard clause and detected clause are in vector format so that easilt get compared 
    clause_vectors = load_clause_vectors(model, standard_clauses)
    
    #now here with the help of pairwise cosine similarity it compare and the clause that have similarity more that 0.4 
    # intializing the empty clause  
    detected_clauses = []
    # now in loop cosine similarity compare between the chunk text and standard clause text is it is 1.0 it means exactly same if 0.7 it is similar and if it is 0.2 means opposite
    #Example:
    #Chunk1 vs Payment → 0.82
    #Chunk2 vs Payment → 0.21
    #Chunk3 vs Payment → 0.65

    for clause_name, clause_vector in clause_vectors.items():
        similarities = cosine_similarity(chunk_vectors, [clause_vector])
        max_sim = np.max(similarities)
        if max_sim >= 0.25:
            detected_clauses.append({
                "clause": clause_name,
                "confidence": float(max_sim)
            })  
    

    result = calculate_risk(detected_clauses, clause_weights_dict)  
   # st.write("Detected:", [c["clause"] for c in detected_clauses])
   # st.write("CSV clauses:", list(clause_weights_dict.keys())[:10])
    
    st.subheader("📊 Document Analysis")

    col1, col2, col3 = st.columns(3)
    col1.metric("✅ Validity", f"{result['validity_score']}%")
    col2.metric("⚠️ Risk", f"{result['risk_score']}%")
    col3.metric("📌 Verdict", result['verdict'])

    with st.expander("Show Missing Clauses"):
        for clause in result["missing_clauses"]:
            st.warning(f"⚠️ Missing: {clause}")

    st.divider()
    st.header("💬 Ask Questions About Your Document")
    
    # Display previous chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])
    
    question = st.chat_input("Ask anything about the document (e.g. What are the payment terms?)")

    if question:
        with st.chat_message("user"):
            st.write(question)
            
        with st.spinner("Analyzing the document..."):
            chain = create_RAG_chain(st.session_state.vector_store)
            answer = chain.invoke(question)
         
        with st.chat_message("assistant"):
            st.write(answer)

        # Save to chat history
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer
        })
        
