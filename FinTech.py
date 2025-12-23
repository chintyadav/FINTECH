import os
import streamlit as st
import numpy as np
import pathlib
from dotenv import load_dotenv
from typing_extensions import TypedDict

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

from transformers import pipeline

# ==================================================
# ENV SETUP
# ==================================================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# ==================================================
# STREAMLIT PAGE CONFIG
# ==================================================
st.set_page_config(page_title="AI Investment Agent", layout="wide")

# ==================================================
# CUSTOM UI STYLE
# ==================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}
.metric-card {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 16px;
    margin-bottom: 20px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.2);
}
.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 12px;
    padding: 10px 20px;
    border: none;
    font-weight: bold;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #0072ff, #00c6ff);
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# HEADER
# ==================================================
st.title("📈 AI Investment Analysis Agent")
st.caption("RAG + FinBERT Sentiment + Agentic Reasoning")

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.title("⚙️ Agent Pipeline")
st.sidebar.markdown("""
- 📄 PDF Financial Data (RAG)
- 🧠 FinBERT Sentiment
- 🤖 Gemini Strategy Agent
- 📊 Risk Simulation
- 📝 Executive Report
""")

st.sidebar.info("This tool provides AI-assisted insights, not financial advice.")

# ==================================================
# FILE UPLOAD
# ==================================================
uploaded_files = st.file_uploader(
    "Upload financial PDF(s)",
    type="pdf",
    accept_multiple_files=True
)

# ==================================================
# VECTOR STORE
# ==================================================
@st.cache_resource
def build_retriever(documents):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    split_docs = splitter.split_documents(documents)

    vectordb = Chroma.from_documents(
        split_docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    return vectordb.as_retriever(search_kwargs={"k": 3})

documents = []
retriever = None

if uploaded_files:
    with st.spinner("📄 Reading and indexing PDFs..."):
        for file in uploaded_files:
            temp_path = f"./temp_{file.name}"
            with open(temp_path, "wb") as f:
                f.write(file.getvalue())

            try:
                loader = PyPDFLoader(temp_path)
                documents.extend(loader.load())
            except Exception as e:
                st.error(f"Failed to read {file.name}: {e}")

            pathlib.Path(temp_path).unlink(missing_ok=True)

        retriever = build_retriever(documents)

    st.success("PDFs processed successfully.")

# ==================================================
# STATE
# ==================================================
class InvestmentState(TypedDict):
    user_query: str
    context: str
    sentiment: str
    sentiment_label: str
    strategy: str
    risk: str
    report: str
    retriever: object

# ==================================================
# LLM
# ==================================================
def init_llm():
    if not GEMINI_API_KEY:
        st.warning("⚠️ GEMINI_API_KEY missing in .env")
        return None
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=GEMINI_API_KEY
    )

llm = init_llm()

def call_llm(prompt):
    if not llm:
        return "LLM not available."
    return llm.invoke(prompt).content

# ==================================================
# SENTIMENT MODEL
# ==================================================
@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="yiyanghkust/finbert-tone",
    )

sentiment_pipe = load_sentiment_model()

# ==================================================
# AGENTS
# ==================================================
def DataAgent(state):
    docs = state["retriever"].get_relevant_documents(state["user_query"])
    context = "\n\n".join(d.page_content for d in docs)
    return {"context": context}

def SentimentAgent(state):
    results = sentiment_pipe(state["context"][:2000])
    label = results[0]["label"].capitalize()
    score = results[0]["score"]
    return {
        "sentiment": f"{label} (confidence: {score:.2f})",
        "sentiment_label": label
    }

def StrategyAgent(state):
    prompt = f"""
You are a financial analyst.

User Question:
{state['user_query']}

Context:
{state['context']}

Sentiment:
{state['sentiment']}

Give Buy/Hold/Sell recommendation for 2026.
"""
    return {"strategy": call_llm(prompt)}

def RiskAgent(state):
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252)
    annual_return = (1 + np.mean(returns)) ** 252 - 1
    volatility = np.std(returns) * np.sqrt(252)
    sharpe = annual_return / volatility
    return {
        "risk": (
            f"Annual Return: {annual_return*100:.2f}%\n"
            f"Volatility: {volatility*100:.2f}%\n"
            f"Sharpe Ratio: {sharpe:.2f}"
        )
    }

def ReportAgent(state):
    prompt = f"""
Create an executive investment memo.

Strategy:
{state['strategy']}

Risk:
{state['risk']}

Conclusion for 2026.
"""
    return {"report": call_llm(prompt)}

# ==================================================
# LANGGRAPH
# ==================================================
graph = StateGraph(InvestmentState)

graph.add_node("DataAgent", DataAgent)
graph.add_node("SentimentAgent", SentimentAgent)
graph.add_node("StrategyAgent", StrategyAgent)
graph.add_node("RiskAgent", RiskAgent)
graph.add_node("ReportAgent", ReportAgent)

graph.add_edge(START, "DataAgent")
graph.add_edge("DataAgent", "SentimentAgent")
graph.add_edge("SentimentAgent", "StrategyAgent")
graph.add_edge("StrategyAgent", "RiskAgent")
graph.add_edge("RiskAgent", "ReportAgent")
graph.add_edge("ReportAgent", END)

app = graph.compile()

# ==================================================
# RUN
# ==================================================
if retriever:
    query = st.text_input(
        "Ask your investment question",
        "Is this company a good investment for 2026?"
    )

    if st.button("🚀 Analyze Investment"):
        with st.spinner("Running AI agents..."):
            result = app.invoke({
                "user_query": query,
                "context": "",
                "sentiment": "",
                "sentiment_label": "",
                "strategy": "",
                "risk": "",
                "report": "",
                "retriever": retriever
            })

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.subheader("🧠 Market Sentiment")
            st.write(result["sentiment"])
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.subheader("⚠️ Risk Analysis")
            st.text(result["risk"])
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.subheader("💼 Investment Strategy")
            st.write(result["strategy"])
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("📄 Executive Investment Memo")
        st.write(result["report"])
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("⬆️ Upload financial PDFs to start analysis.")

# ==================================================
# FOOTER
# ==================================================
st.markdown("""
<hr>
<p style='text-align:center;color:gray;'>
Built with LangGraph • RAG • FinBERT • Gemini
</p>
""", unsafe_allow_html=True)
