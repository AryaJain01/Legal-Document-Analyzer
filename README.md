---
title: AI Powered Legal Document Analyzer
emoji: ⚖️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8501
---
# ⚖️ AI Powered Legal Document Analyzer

![Python](https://img.shields.io/badge/Python-3.10-blue)
![LangChain](https://img.shields.io/badge/LangChain-1.x-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![Groq](https://img.shields.io/badge/Groq-LLaMA3.3-orange)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

An end-to-end AI powered legal document analysis tool that combines **Retrieval Augmented Generation (RAG)**, **NLP based clause detection**, and **data driven risk scoring** to instantly analyze any legal contract.

---

## 🎯 Problem Statement

Legal contracts are complex, lengthy, and difficult to understand for non-lawyers. Missing a single important clause can lead to significant financial or legal risks. This tool automates legal document analysis by:

- Detecting which standard legal clauses are present or missing
- Calculating a data-driven risk score based on real legal contracts
- Allowing users to ask natural language questions about their document

---

## 🚀 Live Demo

> 🔗 Coming soon on Hugging Face Spaces

---

## 📸 Project Preview

![App Screenshot](image.png)

---

## 🧠 How It Works

```
User Uploads PDF
      ↓
PyPDFLoader → Extract Text
      ↓
RecursiveCharacterTextSplitter → Chunk Text
      ↓
┌─────────────────────┐     ┌──────────────────────────┐
│   LEGAL VALIDATION  │     │        RAG Q&A           │
│                     │     │                          │
│ SentenceTransformer │     │ HuggingFaceEmbeddings    │
│ Encode chunks       │     │ FAISS Vector Store       │
│        ↓            │     │        ↓                 │
│ Cosine Similarity   │     │ Retriever (top 3 chunks) │
│ vs CUAD clauses     │     │        ↓                 │
│        ↓            │     │ ChatPromptTemplate       │
│ Risk Score +        │     │        ↓                 │
│ Validity Score      │     │ Groq LLaMA 3.3 70B       │
│ Verdict             │     │        ↓                 │
└─────────────────────┘     │ Answer to User           │
                            └──────────────────────────┘
```

---

## ✨ Features

### 📊 Legal Validation (Data Science + ML)
- Detects **45 standard legal clause types** using Sentence Transformers + Cosine Similarity
- Calculates **Validity Score** and **Risk Score** based on clause presence rates derived from **510 real legal contracts** in the CUAD dataset
- Shows exactly which clauses are **missing** from the uploaded document
- Gives a clear **Valid ✅ or Risky ⚠️ verdict**

### 💬 AI Powered Q&A (GenAI + RAG)
- Ask any natural language question about your document
- Answers are grounded **only** in the uploaded document — no hallucinations
- Full **chat history** maintained during the session
- Powered by **Groq's LLaMA 3.3 70B** model for fast, accurate responses

### 🎯 Smart Document Processing
- Automatic PDF parsing and intelligent chunking
- Persistent vector store using **FAISS** for fast retrieval
- Prevents reprocessing if the same document is uploaded again

---

## 🗂️ Project Structure

```
Legal-Document-Analyzer/
│
├── app.py                  ← Main Streamlit application
├── legal_eda.ipynb         ← EDA notebook on CUAD dataset
├── clause_weights.csv      ← Clause presence rates from CUAD EDA
├── CUAD_v1.json            ← CUAD dataset (510 legal contracts)
├── requirements.txt        ← Project dependencies
├── .env                    ← API keys (not pushed to GitHub)
├── .gitignore              ← Ignores .env and cache files
└── README.md               ← Project documentation
```

---

## 📊 Data Science Component — CUAD EDA

The risk scoring system is **data driven**, not rule based. Here's how:

1. **Dataset:** CUAD (Contract Understanding Atticus Dataset) — 510 real legal contracts annotated by lawyers with 41 clause types
2. **EDA Findings:**
   - 20,910 total clause checks performed
   - 14,208 (68%) clauses were missing across all contracts
   - Document Name, Parties, Agreement Date appear in 95%+ contracts → high weight
   - Source Code Escrow, Price Restrictions appear in <5% contracts → low weight
3. **Clause Weights:** Presence rate of each clause across 510 contracts used as weight in risk scoring
4. **Risk Formula:**
```
Risk Score = (sum of weights of missing clauses / total weight) × 100
Validity Score = 100 - Risk Score
Verdict = Valid if Validity Score >= 60%
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Groq API (LLaMA 3.3 70B) |
| RAG Framework | LangChain (LCEL) |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Vector Store | FAISS |
| Clause Detection | Sentence Transformers + Cosine Similarity |
| PDF Processing | PyPDFLoader |
| Data Analysis | Pandas, Matplotlib, Seaborn |
| Dataset | CUAD (Contract Understanding Atticus Dataset) |
| Language | Python 3.10 |

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/AryaJain01/Legal-Document-Analyzer.git
cd Legal-Document-Analyzer
```

### 2. Create conda environment
```bash
conda create -n legal-doc-analyzer python=3.10
conda activate legal-doc-analyzer
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API key
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get your free Groq API key at: `https://console.groq.com`

### 5. Run the app
```bash
streamlit run app.py
```

---

## 📦 Requirements

```
langchain
langchain-groq
langchain-community
faiss-cpu
pypdf
streamlit
python-dotenv
sentence-transformers
scikit-learn
pandas
numpy
```

---

## 🧪 How To Test

1. Run the app using `streamlit run app.py`
2. Upload any legal PDF document (service agreement, NDA, employment contract etc.)
3. Wait for processing — you'll see validity score and risk score automatically
4. Check the **Show Missing Clauses** section
5. Ask questions like:
   - *"What are the payment terms?"*
   - *"Who are the parties involved?"*
   - *"What is the termination policy?"*
   - *"What is the governing law?"*

---

## 📈 Sample Results

For a basic Service Agreement:
- **Validity Score:** ~62%
- **Risk Score:** ~38%
- **Verdict:** Valid ✅
- **Missing Clauses:** Source Code Escrow, Non-Compete, Insurance, Audit Rights

---

## 🔑 Key Technical Decisions

**Why Groq over OpenAI?**
Groq provides free, extremely fast inference for LLaMA models — ideal for a prototype with no cost constraints.

**Why Cosine Similarity over a trained classifier?**
Cosine similarity with Sentence Transformers requires no training data labeling and generalizes better to unseen legal documents. A trained classifier would overfit to CUAD's specific language patterns.

**Why FAISS over ChromaDB?**
FAISS is lightweight, runs entirely locally, and requires no additional infrastructure — perfect for a self-contained Streamlit app.

**Why chunk_overlap=200?**
Legal clauses often span paragraph boundaries. An overlap of 200 characters ensures no clause context is lost during chunking.

---

## 🚧 Future Improvements

- [ ] Deploy on Hugging Face Spaces
- [ ] Add support for DOCX and TXT files
- [ ] Add contract comparison (upload 2 documents and compare)
- [ ] Add clause highlighting in original document
- [ ] Fine-tune embedding model specifically on legal text
- [ ] Add multilingual support for Indian regional languages

---

## 👨‍💻 Author

**Arya Jain**
- GitHub: [@AryaJain01](https://github.com/AryaJain01)
- Location: Dehradun, Uttarakhand, India

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- [CUAD Dataset](https://www.atticusprojectai.org/cuad) — The Atticus Project for providing 510 annotated legal contracts
- [Groq](https://console.groq.com) — For free, blazing fast LLM inference
- [LangChain](https://python.langchain.com) — For the RAG framework
- [Hugging Face](https://huggingface.co) — For free embedding models
