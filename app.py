import os
import tempfile
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

# ---------------------------------------------------------------------------
# Load environment variables from .env file
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LangChain Multi-Tool Assistant",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Clean CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    [data-testid="stSidebar"] h1 { font-size: 1.15rem; }

    .result-box {
        background-color: #f7f7f8;
        border-left: 3px solid #4a4a4a;
        padding: 1rem 1.25rem;
        border-radius: 4px;
        margin-top: 0.75rem;
        font-size: 0.95rem;
        line-height: 1.7;
        color: #1a1a1a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — tool selector only (API key loaded from .env)
# ---------------------------------------------------------------------------
api_key = os.getenv("OPENAI_API_KEY", "")

with st.sidebar:
    st.title("Tools")

    tool_choice = st.radio(
        "Select a tool",
        options=["Text Summarizer", "Code Explainer", "CSV Query Chatbot"],
        index=0,
    )

    if tool_choice == "CSV Query Chatbot":
        st.divider()
        if st.button("Clear conversation"):
            st.session_state["csv_chat_history"] = []
            st.rerun()

    st.divider()
    st.caption("Built with LangChain and Streamlit.")


# ---------------------------------------------------------------------------
# LLM — single instance per session
# ---------------------------------------------------------------------------
def get_llm(api_key: str) -> ChatOpenAI:
    if "llm" not in st.session_state or st.session_state.get("_api_key") != api_key:
        st.session_state["llm"] = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.3,
            openai_api_key=api_key,
            max_retries=3,
            request_timeout=60,
        )
        st.session_state["_api_key"] = api_key
    return st.session_state["llm"]


# ===================================================================
# TOOL 1 — TEXT SUMMARIZER
#
# LangChain components used:
#   - RecursiveCharacterTextSplitter (splits long text into chunks)
#   - Document (wraps each chunk)
#   - load_summarize_chain (map_reduce summarization)
# ===================================================================

def run_summarizer(llm: ChatOpenAI, text: str) -> str:
    """
    Splits long text into chunks, summarizes each chunk,
    then merges all chunk summaries into one final summary.
    """
    # Split the text into manageable chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    # Wrap each chunk as a LangChain Document
    docs = [Document(page_content=c) for c in chunks]

    # Run the map-reduce summarization chain
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    result = chain.invoke(docs)

    return result["output_text"]


# ===================================================================
# TOOL 2 — PYTHON CODE EXPLAINER
#
# LangChain components used:
#   - PromptTemplate (structures the prompt)
#   - LCEL chain: prompt | llm | StrOutputParser
# ===================================================================

def run_code_explainer(llm: ChatOpenAI, code: str) -> str:
    """
    Takes Python code and explains it in plain English
    that a non-technical person can understand.
    """
    prompt = PromptTemplate(
        input_variables=["code"],
        template=(
            "You are a senior Python developer explaining code to a "
            "non-technical manager.\n\n"
            "Rules:\n"
            "- Use plain English. Avoid jargon.\n"
            "- Walk through the code step by step.\n"
            "- Mention what the code does overall first, then break it down.\n"
            "- If there are potential issues or improvements, note them briefly.\n\n"
            "Code:\n```python\n{code}\n```\n\n"
            "Explanation:"
        ),
    )

    # LCEL chain: prompt -> LLM -> extract string
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"code": code})

    return result


# ===================================================================
# TOOL 3 — CSV QUERY CHATBOT
#
# LangChain components used:
#   - create_csv_agent (autonomous pandas agent)
#   - PromptTemplate (for the explanation step)
#   - LCEL chain: prompt | llm | StrOutputParser
# ===================================================================

def run_csv_query(llm: ChatOpenAI, file_path: str, question: str,
                  chat_history: list) -> dict:
    """
    Queries CSV data and returns a human-readable answer.

    Step 1: Build context from past conversation for follow-ups
    Step 2: CSV agent writes pandas code and gets raw result
    Step 3: LLM explains the raw result in plain English
    """

    # Build short context from chat history (last 4 messages only)
    context_lines = []
    for msg in chat_history[-4:]:
        role = "You" if msg["role"] == "user" else "Agent"
        content = msg.get("display", msg["content"])
        if len(content) > 200:
            content = content[:200] + "..."
        context_lines.append(f"{role}: {content}")
    context = "\n".join(context_lines)

    # Add context for follow-up questions
    if context:
        full_question = (
            f"Context from previous messages:\n{context}\n\n"
            f"Current question: {question}"
        )
    else:
        full_question = question

    # Step 1: CSV agent queries the data
    csv_agent = create_csv_agent(
        llm,
        file_path,
        verbose=False,
        agent_type="zero-shot-react-description",
        allow_dangerous_code=True,
        number_of_head_rows=5,
        max_iterations=5,
        early_stopping_method="generate",
    )

    try:
        agent_result = csv_agent.invoke({"input": full_question})
        raw_answer = str(agent_result["output"])
    except Exception:
        # If context makes it too long, retry without context
        try:
            agent_result = csv_agent.invoke({"input": question})
            raw_answer = str(agent_result["output"])
        except Exception as e:
            return {
                "answer": (
                    "Sorry, there was an error querying the data. "
                    "Please try again or rephrase your question."
                ),
                "raw": str(e)[:200],
            }

    # Step 2: Explain the raw result in plain English
    df = pd.read_csv(file_path)
    columns = ", ".join(df.columns.tolist())

    explain_prompt = PromptTemplate(
        input_variables=["question", "raw_answer", "columns", "row_count"],
        template=(
            "You are a data analysis assistant. You just queried a CSV file.\n\n"
            "Dataset info:\n"
            "- Columns: {columns}\n"
            "- Total rows: {row_count}\n\n"
            "User's question: {question}\n\n"
            "Raw result from your query: {raw_answer}\n\n"
            "Present the result in a clear, human-readable way.\n"
            "Do not just dump a list or a number. Write proper sentences.\n"
            "If the result is a list, present it as a readable sentence "
            "(example: 'There are 5 departments: Engineering, Marketing, "
            "Sales, HR, and Finance.').\n"
            "If the result is a number, put it in context "
            "(example: 'The average salary in Engineering is 95,400.').\n"
            "Be specific, complete, and conversational."
        ),
    )

    try:
        explain_chain = explain_prompt | llm | StrOutputParser()
        explained = explain_chain.invoke({
            "question": question,
            "raw_answer": raw_answer,
            "columns": columns,
            "row_count": str(len(df)),
        })
    except Exception:
        # If explanation fails, return raw answer directly
        explained = raw_answer

    return {"answer": explained, "raw": raw_answer}


# ===================================================================
# MAIN UI
# ===================================================================
st.header("LangChain Multi-Tool Assistant")
st.write(
    "Pick a tool from the sidebar, paste your input, and get results. "
    "Each tool runs through a dedicated LangChain chain or agent."
)

if not api_key:
    st.error(
        "OPENAI_API_KEY not found. "
        "Add your key to the .env file and restart the app."
    )
    st.stop()

llm = get_llm(api_key)

# ---- Text Summarizer ----
if tool_choice == "Text Summarizer":
    st.subheader("Text Summarizer")
    st.write(
        "Paste any long text below. The tool splits it into chunks, "
        "summarizes each chunk, then merges the summaries into one result."
    )

    input_text = st.text_area(
        "Text to summarize",
        height=250,
        placeholder="Paste an article, report, or any long-form text here...",
    )

    if st.button("Summarize"):
        if not input_text.strip():
            st.warning("Please paste some text first.")
        else:
            with st.spinner("Summarizing..."):
                try:
                    summary = run_summarizer(llm, input_text)
                except Exception as e:
                    summary = f"Error: {str(e)[:300]}"

            st.markdown("**Summary**")
            st.markdown(
                f'<div class="result-box">{summary}</div>',
                unsafe_allow_html=True,
            )

# ---- Code Explainer ----
elif tool_choice == "Code Explainer":
    st.subheader("Python Code Explainer")
    st.write(
        "Paste a Python snippet below. The tool will explain what the code "
        "does in plain English, step by step."
    )

    input_code = st.text_area(
        "Python code",
        height=250,
        placeholder="def greet(name):\n    return f'Hello, {name}!'",
    )

    if st.button("Explain"):
        if not input_code.strip():
            st.warning("Please paste some code first.")
        else:
            with st.spinner("Analyzing code..."):
                try:
                    explanation = run_code_explainer(llm, input_code)
                except Exception as e:
                    explanation = f"Error: {str(e)[:300]}"

            st.markdown("**Explanation**")
            st.markdown(
                f'<div class="result-box">{explanation}</div>',
                unsafe_allow_html=True,
            )

# ---- CSV Query Chatbot ----
elif tool_choice == "CSV Query Chatbot":
    st.subheader("CSV Query Chatbot")
    st.write(
        "Upload a CSV file and chat with your data. Ask questions, "
        "follow up on answers, and dig deeper — the chatbot remembers "
        "the conversation."
    )

    # Initialize chat history
    if "csv_chat_history" not in st.session_state:
        st.session_state["csv_chat_history"] = []
    if "csv_file_path" not in st.session_state:
        st.session_state["csv_file_path"] = None

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Save file — only re-save if it is a new file
        current_name = uploaded_file.name
        if st.session_state.get("csv_file_name") != current_name:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".csv", dir=tempfile.gettempdir()
            ) as tmp:
                tmp.write(uploaded_file.getvalue())
                st.session_state["csv_file_path"] = tmp.name
                st.session_state["csv_file_name"] = current_name
                st.session_state["csv_chat_history"] = []

        file_path = st.session_state["csv_file_path"]

        # Show data preview in a collapsible section
        df = pd.read_csv(file_path)
        with st.expander(
            f"Data preview — {len(df)} rows, {len(df.columns)} columns",
            expanded=False,
        ):
            st.dataframe(df.head(15), use_container_width=True)
            st.caption(f"Columns: {', '.join(df.columns.tolist())}")

        st.divider()

        # Display chat history
        for msg in st.session_state["csv_chat_history"]:
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.write(msg["content"])
                else:
                    st.write(msg["display"])

        # Chat input
        question = st.chat_input("Ask a question about your data...")

        if question:
            # Show user message
            with st.chat_message("user"):
                st.write(question)

            st.session_state["csv_chat_history"].append({
                "role": "user",
                "content": question,
            })

            # Run the query
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = run_csv_query(
                        llm,
                        file_path,
                        question,
                        st.session_state["csv_chat_history"],
                    )

                st.write(result["answer"])

            # Save assistant message to history
            st.session_state["csv_chat_history"].append({
                "role": "assistant",
                "content": result["answer"],
                "display": result["answer"],
            })

    else:
        st.write("Upload a CSV file to start chatting with your data.")
