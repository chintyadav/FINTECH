# 📈 Aura — AI Investment Analysis Agent

**Aura** (Autonomous Unified RAG Agent) is an AI-powered investment analysis system that combines **Retrieval-Augmented Generation (RAG)** with multi-agent reasoning to provide explainable, evidence-driven insights on financial data.

It leverages uploaded financial PDFs, real-time market data, and sentiment analysis models to help users answer long-term investment questions in a grounded and transparent manner.

---

## 🚀 Key Features

- 📄 **Document Retrieval & RAG**  
  Ingests financial PDFs and builds a semantic vector index for contextual retrieval.

- 🧠 **FinBERT Sentiment Analysis**  
  Evaluates financial sentiment directly from document content.

- 📊 **Risk Metrics**  
  Computes annual return, volatility, and Sharpe ratio using Yahoo Finance data.

- 🤖 **Agentic Reasoning**  
  Uses a coordinated multi-agent pipeline:
  DataAgent, SentimentAgent, RiskAgent, StrategyAgent, and ReportAgent.

- 📝 **Executive Investment Memo**  
  Generates concise, professional summaries suitable for stakeholders.

---

## 📁 Repository Structure



---

## 🛠️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/chintyadav/FINTECH.git
cd FINTECH

2️⃣ Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Create a .env file
GROQ_API_KEY=<your Groq API key>
HF_TOKEN=<your HuggingFace token>

📊 Usage
Run the Streamlit app
streamlit run usegroq.py

Steps

Upload one or more financial PDF documents

Enter a stock ticker (e.g., YESBANK.NS)

Ask a question such as:
“Is this company a good investment for 2026?”

View:

Market sentiment

Risk metrics

Investment strategy

Executive investment memo

🧠 How It Works
PDF Upload & Vector Indexing

Financial PDFs are parsed and indexed using LangChain + Chroma.

Retrieval & Context Building

A RAG pipeline retrieves relevant document chunks for each query.

Multi-Agent Pipeline

DataAgent → Retrieves contextual evidence

SentimentAgent → FinBERT-based tone analysis

RiskAgent → Market risk metrics from Yahoo Finance

StrategyAgent → Evidence-driven investment stance

ReportAgent → Executive-level summary

Streamlit UI

Results are displayed in a clean, dark-themed dashboard with readable, white-highlighted outputs.

📌 Requirements

Python ≥ 3.10

Streamlit

LangChain & LangGraph

HuggingFace embeddings

FinBERT sentiment model (finbert-tone)

Yahoo Finance (yfinance)

⚠️ Disclaimer

This project is for educational and demonstration purposes only
and does not constitute financial advice.
