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
from langchain_groq import ChatGroq

from transformers import pipeline

from typing_extensions import TypedDict, NotRequired
from typing import Dict, Any
import yfinance as yf
import numpy as np



# ==================================================
# ENV SETUP
# ==================================================
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

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
st.sidebar.subheader("📊 Market Data")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")


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

    return vectordb.as_retriever(search_kwargs={"k": 8})

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
    retriever: object
    ticker: str

    context: NotRequired[str]
    sentiment: NotRequired[str]
    sentiment_label: NotRequired[str]
    strategy: NotRequired[Dict[str, Any]]
    risk: NotRequired[str]
    report: NotRequired[str]

# ==================================================
# LLM
# ==================================================
def init_llm():
    if not groq_api_key:
        st.warning("⚠️ groq_api_key missing in .env")
        return None
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=groq_api_key
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
    retriever = state["retriever"]
    query = state["user_query"]

    if hasattr(retriever, "invoke"):
        docs = retriever.invoke(query)
    else:
        docs = retriever.get_relevant_documents(query)

    context = "\n\n".join(d.page_content for d in docs)
    return {"context": context}


def SentimentAgent(state):
    context = state.get("context", "").strip()

    # ---------- Safety ----------
    if not context:
        return {
            "sentiment": "Neutral (no document context available)",
            "sentiment_label": "Neutral"
        }

    # ---------- Step 1: Chunk the document ----------
    chunks = [
        context[i:i+500]
        for i in range(0, len(context), 500)
    ]

    # Limit chunks to avoid overload (optional but safe)
    chunks = chunks[:20]

    # ---------- Step 2: FinBERT on each chunk ----------
    finbert_results = sentiment_pipe(chunks)

    counts = {"positive": 0, "neutral": 0, "negative": 0}
    confidence_sum = 0.0

    for res in finbert_results:
        label = res["label"].lower()
        counts[label] += 1
        confidence_sum += res["score"]

    total = sum(counts.values())

    distribution = {
        "positive": counts["positive"] / total,
        "neutral": counts["neutral"] / total,
        "negative": counts["negative"] / total,
    }

    dominant_label = max(distribution, key=distribution.get).capitalize()
    avg_confidence = confidence_sum / total

    # ---------- Step 3: LLM interpretation ----------
    llm_prompt = f"""
You are analyzing the overall tone of a financial document.

FinBERT chunk-level sentiment distribution:
- Positive: {distribution['positive']:.0%}
- Neutral: {distribution['neutral']:.0%}
- Negative: {distribution['negative']:.0%}

Document type: Company disclosure / earnings-related document

TASK:
1. Classify the overall sentiment as one of:
   - Positive
   - Neutral
   - Cautious (preferred if mixed or risk-aware)
2. Explain the reasoning in 1–2 concise sentences
3. Do NOT treat cautious language as deteriorating fundamentals

OUTPUT FORMAT:
Overall Sentiment: <Positive | Neutral | Cautious>
Explanation: <brief explanation>
"""

    llm_sentiment = call_llm(llm_prompt)

    # ---------- Final output ----------
    return {
        "sentiment": (
            f"{llm_sentiment.strip()}\n\n"
            f"(FinBERT majority: {dominant_label}, "
            f"avg confidence: {avg_confidence:.2f})"
        ),
        "sentiment_label": dominant_label
    }



def StrategyAgent(state):
    context = state.get("context", "")
    sentiment = state.get("sentiment", "Neutral")
    risk = state.get("risk", "Risk data unavailable")
    ticker = state.get("ticker", "")

    prompt = f"""
SYSTEM:
You are a conservative institutional financial analyst.
You must be factual, cautious, and evidence-driven.
You MUST NOT invent missing data.
You MUST NOT exaggerate sentiment signals.

USER QUESTION:
{state['user_query']}

COMPANY IDENTIFIER:
Ticker: {ticker}

DOCUMENT CONTEXT (company PDFs):
{context}

MARKET SENTIMENT (FinBERT):
{sentiment}

RISK METRICS (market data):
{risk}

FUNDAMENTAL EVIDENCE RULE (CRITICAL):
- You may ONLY claim "deteriorating fundamentals" if the document context EXPLICITLY states:
  revenue decline, losses, margin collapse, reduced guidance, or business contraction
- If such explicit statements are NOT present, you MUST state that fundamentals appear stable or mixed
- You are NOT allowed to infer deteriorating fundamentals from sentiment or risk metrics alone

DOCUMENT QUALITY OVERRIDE (CRITICAL):
- If the document explicitly admits lack of financial data,
  you MUST downgrade confidence and default to HOLD
- A SELL recommendation cannot be issued based solely on sentiment or generic risk factors


EVALUATION PRIORITY (VERY IMPORTANT):
1. Company fundamentals from documents (highest priority)
2. Market risk metrics (return, volatility, Sharpe)
3. Sentiment signals (supporting evidence only)

SELL GUARDRAIL (CRITICAL RULE):
SELL is allowed ONLY if ALL conditions are met:
1. Document context explicitly states deteriorating fundamentals
2. Risk metrics show sustained negative performance
3. Sentiment is negative

If condition (1) is NOT clearly met → DO NOT recommend SELL
Default to HOLD


ADDITIONAL RULES:
- Do NOT claim "limited financial data" if documents or risk metrics are present
- Large, established companies should default to HOLD when signals are mixed
- If evidence is weak, incomplete, or conflicting → HOLD
- Do NOT predict future prices
- Use professional, cautious language suitable for investors

TASK:
Provide a long-term (2026) investment stance based strictly on evidence.

OUTPUT FORMAT (STRICT):
Recommendation: BUY | HOLD | SELL
Rationale:
- Bullet 1: Key fundamental insight from document evidence
- Bullet 2: How sentiment supports or contradicts fundamentals
- Bullet 3: Risk interpretation (return vs volatility)
Confidence Level: LOW | MEDIUM | HIGH
"""

    strategy_text = call_llm(prompt)

    return {"strategy": strategy_text}




def RiskAgent(state):
    ticker = state.get("ticker", "AAPL")

    data = yf.download(
        ticker,
        period="5y",
        interval="1d",
        progress=False,
        auto_adjust=False
    )

    if data.empty:
        return {"risk": f"No market data available for {ticker}."}

    # Safe price column handling
    if "Adj Close" in data.columns:
        prices = data["Adj Close"].dropna()
    elif "Close" in data.columns:
        prices = data["Close"].dropna()
    else:
        return {"risk": "Price column missing in Yahoo Finance response."}

    if len(prices) < 20:
        return {"risk": "Insufficient price history to compute risk metrics."}

    returns = prices.pct_change().dropna()

    # ✅ FORCE SCALARS
    annual_return = float((1 + returns.mean()) ** 252 - 1)
    volatility = float(returns.std() * np.sqrt(252))

    sharpe = None
    if volatility != 0:
        sharpe = annual_return / volatility

    sharpe_text = f"{sharpe:.2f}" if sharpe is not None else "N/A"

    return {
        "risk": (
            f"📌 Data Source: Yahoo Finance\n"
            f"🏷️ Ticker: {ticker}\n"
            f"📆 Period: {prices.index.min().date()} → {prices.index.max().date()}\n\n"
            f"Annual Return: {annual_return*100:.2f}%\n"
            f"Volatility (Annual): {volatility*100:.2f}%\n"
            f"Sharpe Ratio: {sharpe_text}"
        )
    }




def get_company_name(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
        return info.get("longName") or info.get("shortName") or ticker
    except Exception:
        return ticker


def ReportAgent(state):
    strategy = state.get("strategy", "Strategy unavailable")
    risk = state.get("risk", "Risk data unavailable")
    ticker = state.get("ticker", "N/A")

    company_name = get_company_name(ticker)

    prompt = f"""
SYSTEM:
You are writing for senior investment stakeholders.
Be concise, neutral, and professional.
Do NOT introduce new facts.

TASK:
Create a short executive investment memo (max 120 words).

STRUCTURE (STRICT):
Company Name
Ticker
Overview (2–3 lines)
Key Considerations (bullet points)
Final Recommendation
Disclaimer

INPUTS:
Company Name: {company_name}
Ticker: {ticker}

Investment Strategy:
{strategy}

Risk Analysis:
{risk}

RULES:
- Do NOT contradict the strategy
- Do NOT add new financial data
- Avoid promotional language
- Use cautious wording
"""

    report_text = call_llm(prompt)

    return {"report": report_text}


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
graph.add_edge("SentimentAgent", "RiskAgent")
graph.add_edge("RiskAgent","StrategyAgent")
graph.add_edge("StrategyAgent", "ReportAgent")
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
                "retriever": retriever,
                "ticker": ticker
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
