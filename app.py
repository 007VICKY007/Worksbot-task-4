import os
import tempfile
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# LangChain imports — every component used in this project
# ---------------------------------------------------------------------------
from langchain_openai import ChatOpenAI                          # LLM wrapper
from langchain.prompts import PromptTemplate                     # Prompt templates
from langchain.chains import LLMChain                            # Prompt + LLM chain
from langchain.tools import Tool                                 # Tool wrapper
from langchain.agents import initialize_agent, AgentType         # Agent framework
from langchain.memory import ConversationBufferMemory            # Chat memory
from langchain_experimental.agents.agent_toolkits import (
    create_csv_agent,                                            # CSV agent
)
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Text chunking
from langchain.chains.summarize import load_summarize_chain      # Summarize chain
from langchain.docstore.document import Document                 # Document wrapper

# ---------------------------------------------------------------------------
# Load environment variables
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
    .reasoning-box {
        background-color: #fafafa;
        border-left: 3px solid #b0b0b0;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin-top: 0.5rem;
        font-size: 0.88rem;
        line-height: 1.6;
        color: #555;
        white-space: pre-wrap;
    }
    .langchain-tag {
        display: inline-block;
        background-color: #e8f4e8;
        color: #2d6a2d;
        padding: 0.15rem 0.5rem;
        border-radius: 3px;
        font-size: 0.78rem;
        margin-right: 0.3rem;
        margin-bottom: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
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
            st.session_state.pop("csv_memory", None)
            st.rerun()

    st.divider()
    st.caption("Built with LangChain and Streamlit.")


# ---------------------------------------------------------------------------
# LangChain: ChatOpenAI — single LLM instance shared by all tools
# ---------------------------------------------------------------------------
def get_llm(api_key: str) -> ChatOpenAI:
    if "llm" not in st.session_state or st.session_state.get("_api_key") != api_key:
        st.session_state["llm"] = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.3,
            openai_api_key=api_key,
        )
        st.session_state["_api_key"] = api_key
    return st.session_state["llm"]


# ===================================================================
# TOOL 1 — TEXT SUMMARIZER
#
# LangChain components:
#   Tool, RecursiveCharacterTextSplitter, Document,
#   load_summarize_chain, LLMChain, PromptTemplate
#
# Flow: Tool 1 (summarize) -> Tool 2 (explain) called in sequence
# ===================================================================

def build_summarizer_tool(llm: ChatOpenAI) -> Tool:
    """
    LangChain: Tool wrapping load_summarize_chain (map_reduce).
    Splits text into chunks, summarizes each, merges into one.
    """
    def _run(text: str) -> str:
        # LangChain: RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
        chunks = splitter.split_text(text)

        # LangChain: Document
        docs = [Document(page_content=c) for c in chunks]

        # LangChain: load_summarize_chain (map_reduce)
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        result = chain.invoke(docs)
        return result["output_text"]

    return Tool(
        name="text_summarizer",
        func=_run,
        description="Summarizes long text using map-reduce chain.",
    )


def build_summary_explainer_tool(llm: ChatOpenAI) -> Tool:
    """
    LangChain: Tool wrapping LLMChain + PromptTemplate.
    Takes a raw summary and explains it with reasoning.
    """
    # LangChain: PromptTemplate
    prompt = PromptTemplate(
        input_variables=["raw_summary", "chunk_count", "char_count"],
        template=(
            "You are a summarization agent. You just summarized a long text "
            "using a map-reduce approach.\n\n"
            "Process details:\n"
            "- Original text: approximately {char_count} characters\n"
            "- Split into {chunk_count} chunk(s), summarized each, then merged\n\n"
            "Raw summary produced:\n{raw_summary}\n\n"
            "Now write two sections:\n\n"
            "REASONING:\n"
            "In 2-3 sentences, explain what the original text was about, "
            "how many chunks you processed, and your approach.\n\n"
            "ANSWER:\n"
            "Present the full summary below. Keep all the important details "
            "from the raw summary. Write clear, readable paragraphs. "
            "Do not shorten or skip information — include everything.\n\n"
            "Use exactly these headers: REASONING: and ANSWER:"
        ),
    )

    # LangChain: LLMChain
    chain = LLMChain(llm=llm, prompt=prompt)

    def _run(input_str: str) -> str:
        # Parse the input: raw_summary|||chunk_count|||char_count
        parts = input_str.split("|||")
        return chain.invoke({
            "raw_summary": parts[0],
            "chunk_count": parts[1] if len(parts) > 1 else "1",
            "char_count": parts[2] if len(parts) > 2 else "unknown",
        })["text"]

    return Tool(
        name="summary_explainer",
        func=_run,
        description="Explains a raw summary with reasoning and clean formatting.",
    )


def run_summarizer_pipeline(llm: ChatOpenAI, text: str) -> dict:
    """
    Runs two LangChain Tools in sequence:
      1. text_summarizer   — chunks and summarizes the text
      2. summary_explainer — explains the result with reasoning
    """
    # Build LangChain Tools
    summarizer = build_summarizer_tool(llm)
    explainer = build_summary_explainer_tool(llm)

    # Step 1: Run the summarizer tool
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    chunk_count = len(splitter.split_text(text))

    raw_summary = summarizer.run(text)

    # Step 2: Run the explainer tool
    explainer_input = f"{raw_summary}|||{chunk_count}|||{len(text)}"
    explained = explainer.run(explainer_input)

    # Build the execution log
    agent_log = (
        f"Step 1:\n"
        f"  Tool used: text_summarizer\n"
        f"  Input: [{len(text)} characters of text]\n"
        f"  Chunks created: {chunk_count}\n"
        f"  Output: [{len(raw_summary)} characters of raw summary]\n\n"
        f"Step 2:\n"
        f"  Tool used: summary_explainer\n"
        f"  Input: raw summary + metadata\n"
        f"  Output: explained summary with reasoning"
    )

    parsed = _parse_reasoning_answer(explained, raw_summary)
    parsed["agent_steps"] = agent_log
    return parsed


# ===================================================================
# TOOL 2 — CODE EXPLAINER
#
# LangChain components:
#   Tool, LLMChain, PromptTemplate
#
# Flow: Tool 1 (analyze) -> Tool 2 (explain) called in sequence
# ===================================================================

def build_code_analyzer_tool(llm: ChatOpenAI) -> Tool:
    """
    LangChain: Tool wrapping LLMChain + PromptTemplate.
    Analyzes Python code structure and identifies patterns.
    """
    # LangChain: PromptTemplate
    prompt = PromptTemplate(
        input_variables=["code"],
        template=(
            "You are a senior Python developer. Analyze this code and identify:\n"
            "- What libraries or modules are imported\n"
            "- What functions or classes are defined\n"
            "- What data structures are used\n"
            "- What the main logic flow is\n"
            "- Any patterns (web scraping, data processing, API calls, etc.)\n\n"
            "Code:\n```python\n{code}\n```\n\n"
            "Provide a detailed technical analysis in 4-6 sentences."
        ),
    )

    # LangChain: LLMChain
    chain = LLMChain(llm=llm, prompt=prompt)

    return Tool(
        name="code_analyzer",
        func=lambda code: chain.invoke({"code": code})["text"],
        description="Analyzes Python code structure, imports, and patterns.",
    )


def build_code_explainer_tool(llm: ChatOpenAI) -> Tool:
    """
    LangChain: Tool wrapping LLMChain + PromptTemplate.
    Takes a technical analysis + original code and writes a plain-English explanation.
    """
    # LangChain: PromptTemplate
    prompt = PromptTemplate(
        input_variables=["analysis", "code"],
        template=(
            "You are explaining Python code to a non-technical manager.\n\n"
            "Your technical analysis of the code:\n{analysis}\n\n"
            "Original code:\n```python\n{code}\n```\n\n"
            "Write your response in exactly two sections:\n\n"
            "REASONING:\n"
            "Describe what you identified — what features are used, what the "
            "structure looks like, what patterns you noticed. 3-5 sentences. "
            "This shows your analysis process.\n\n"
            "ANSWER:\n"
            "Explain the code in plain English. Start with what the code does "
            "overall in one sentence. Then walk through it step by step. "
            "Cover every function and every important line. "
            "If there are issues or improvements, mention them at the end. "
            "Use plain language — no jargon.\n\n"
            "Use exactly these headers: REASONING: and ANSWER:"
        ),
    )

    # LangChain: LLMChain
    chain = LLMChain(llm=llm, prompt=prompt)

    return Tool(
        name="code_explainer",
        func=lambda input_str: chain.invoke({
            "analysis": input_str.split("|||CODE|||")[0],
            "code": input_str.split("|||CODE|||")[1] if "|||CODE|||" in input_str else "",
        })["text"],
        description="Explains code in plain English using the technical analysis.",
    )


def run_code_explainer_pipeline(llm: ChatOpenAI, code: str) -> dict:
    """
    Runs two LangChain Tools in sequence:
      1. code_analyzer  — identifies structure and patterns
      2. code_explainer — writes plain-English explanation
    """
    # Build LangChain Tools
    analyzer = build_code_analyzer_tool(llm)
    explainer = build_code_explainer_tool(llm)

    # Step 1: Analyze the code
    analysis = analyzer.run(code)

    # Step 2: Explain using the analysis
    explained = explainer.run(f"{analysis}|||CODE|||{code}")

    # Build execution log
    agent_log = (
        f"Step 1:\n"
        f"  Tool used: code_analyzer\n"
        f"  Input: [{len(code)} characters of Python code]\n"
        f"  Output: {analysis[:300]}...\n\n"
        f"Step 2:\n"
        f"  Tool used: code_explainer\n"
        f"  Input: technical analysis + original code\n"
        f"  Output: plain-English explanation with reasoning"
    )

    parsed = _parse_reasoning_answer(explained, explained)
    parsed["agent_steps"] = agent_log
    return parsed


# ===================================================================
# TOOL 3 — CSV QUERY CHATBOT
#
# LangChain components:
#   Tool, create_csv_agent, LLMChain, PromptTemplate,
#   ConversationBufferMemory, initialize_agent
#
# Flow: Agent with memory picks tools automatically per question
# ===================================================================

def build_csv_query_tool(llm: ChatOpenAI, file_path: str) -> Tool:
    """
    LangChain: Tool wrapping create_csv_agent.
    The csv_agent writes and executes pandas code to answer questions.
    """
    # LangChain: create_csv_agent
    csv_agent = create_csv_agent(
        llm,
        file_path,
        verbose=False,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        allow_dangerous_code=True,
    )

    def _query(question: str) -> str:
        result = csv_agent.invoke({"input": question})
        return str(result["output"])

    return Tool(
        name="csv_data_query",
        func=_query,
        description=(
            "Queries the CSV dataset using natural language. Input is a "
            "question about the data. Returns the raw result."
        ),
    )


def build_csv_explainer_tool(llm: ChatOpenAI, columns: str,
                              row_count: int) -> Tool:
    """
    LangChain: Tool wrapping LLMChain + PromptTemplate.
    Takes a raw query result and explains it as a proper sentence.
    """
    # LangChain: PromptTemplate
    template_str = (
        "You are a data analysis chatbot. You queried a CSV file.\n\n"
        "Dataset: COLUMNS_PLACEHOLDER columns, ROWS_PLACEHOLDER rows.\n\n"
        "User question: {question}\n\n"
        "Raw result: {raw_answer}\n\n"
        "Write exactly two sections:\n\n"
        "REASONING:\n"
        "In 2-4 sentences, explain what you did — which columns, what "
        "operation (counted, averaged, filtered, grouped, etc.), and how "
        "you arrived at the result.\n\n"
        "ANSWER:\n"
        "Present the result conversationally. Do not dump raw lists or "
        "numbers. Write proper sentences.\n"
        "- If it is a list, write: 'There are N items: A, B, C, and D.'\n"
        "- If it is a number, give context: 'The average salary is 95,400.'\n"
        "- If it is a table, describe the key findings in sentences.\n"
        "Be specific, complete, and conversational.\n\n"
        "Use exactly these headers: REASONING: and ANSWER:"
    )
    template_str = template_str.replace("COLUMNS_PLACEHOLDER", columns)
    template_str = template_str.replace("ROWS_PLACEHOLDER", str(row_count))

    prompt = PromptTemplate(
        input_variables=["question", "raw_answer"],
        template=template_str,
    )

    # LangChain: LLMChain
    chain = LLMChain(llm=llm, prompt=prompt)

    def _explain(input_str: str) -> str:
        parts = input_str.split("|||ANSWER|||")
        question = parts[0] if len(parts) > 1 else ""
        raw = parts[1] if len(parts) > 1 else input_str
        return chain.invoke({"question": question, "raw_answer": raw})["text"]

    return Tool(
        name="result_explainer",
        func=_explain,
        description=(
            "Explains a raw data result in plain English. "
            "Input format: question|||ANSWER|||raw_result"
        ),
    )


def get_csv_memory() -> ConversationBufferMemory:
    """
    LangChain: ConversationBufferMemory
    Stores full chat history so the agent understands follow-up questions.
    """
    if "csv_memory" not in st.session_state:
        st.session_state["csv_memory"] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )
    return st.session_state["csv_memory"]


def run_csv_chatbot(llm: ChatOpenAI, file_path: str,
                    question: str) -> dict:
    """
    Two-step pipeline with LangChain memory:
      1. csv_data_query tool — gets raw answer from CSV
      2. result_explainer tool — explains it clearly

    ConversationBufferMemory stores past Q&A so follow-ups work.
    """
    df = pd.read_csv(file_path)
    columns = ", ".join(df.columns.tolist())
    row_count = len(df)

    # Build LangChain Tools
    query_tool = build_csv_query_tool(llm, file_path)
    explainer_tool = build_csv_explainer_tool(llm, columns, row_count)

    # LangChain: ConversationBufferMemory
    memory = get_csv_memory()

    # Build context from memory for the CSV agent
    chat_history = memory.load_memory_variables({}).get("chat_history", [])
    context_lines = []
    for msg in chat_history[-10:]:
        role = "User" if msg.type == "human" else "Agent"
        context_lines.append(f"{role}: {msg.content}")
    context = "\n".join(context_lines)

    # Add context to question for follow-ups
    if context:
        full_question = (
            f"Previous conversation:\n{context}\n\n"
            f"Current question: {question}\n\n"
            f"If the question references previous answers (like 'them', "
            f"'those', 'that department'), use the context. "
            f"Answer the current question only."
        )
    else:
        full_question = question

    # Step 1: Run csv_data_query tool
    raw_answer = query_tool.run(full_question)

    # Step 2: Run result_explainer tool
    explainer_input = f"{question}|||ANSWER|||{raw_answer}"
    explained = explainer_tool.run(explainer_input)

    # Save to LangChain memory
    memory.save_context(
        {"input": question},
        {"output": explained.split("ANSWER:")[-1].strip() if "ANSWER:" in explained else explained}
    )

    # Build execution log
    agent_log = (
        f"Step 1:\n"
        f"  Tool used: csv_data_query (LangChain create_csv_agent)\n"
        f"  Input: {question}\n"
        f"  Raw output: {raw_answer}\n\n"
        f"Step 2:\n"
        f"  Tool used: result_explainer (LangChain LLMChain)\n"
        f"  Input: question + raw answer\n"
        f"  Output: human-readable explanation"
    )

    parsed = _parse_reasoning_answer(explained, raw_answer)
    parsed["agent_steps"] = agent_log
    return parsed


# ===================================================================
# HELPERS
# ===================================================================

def _parse_reasoning_answer(text: str, fallback: str) -> dict:
    """Split LLM output on REASONING: and ANSWER: headers."""
    reasoning = ""
    answer = ""

    if "REASONING:" in text and "ANSWER:" in text:
        parts = text.split("ANSWER:")
        reasoning_part = parts[0]
        answer = parts[1].strip() if len(parts) > 1 else str(fallback)

        if "REASONING:" in reasoning_part:
            reasoning = reasoning_part.split("REASONING:")[1].strip()
        else:
            reasoning = reasoning_part.strip()
    else:
        answer = text.strip()
        reasoning = "Processed the input and generated a response."

    return {"reasoning": reasoning, "answer": answer}


def display_result(result: dict, langchain_components: list):
    """Display answer, LangChain tags, reasoning, and agent log."""
    st.markdown("**Answer**")
    st.markdown(
        f'<div class="result-box">{result["answer"]}</div>',
        unsafe_allow_html=True,
    )

    tags_html = " ".join(
        f'<span class="langchain-tag">{c}</span>' for c in langchain_components
    )
    st.markdown(
        f'<div style="margin-top:0.75rem;">'
        f'<span style="font-size:0.8rem;color:#777;">LangChain: </span>'
        f'{tags_html}</div>',
        unsafe_allow_html=True,
    )

    with st.expander("How the agent reached this answer"):
        st.markdown(
            f'<div class="reasoning-box">{result["reasoning"]}</div>',
            unsafe_allow_html=True,
        )
        if result.get("agent_steps"):
            st.markdown("**Agent execution log**")
            st.code(result["agent_steps"], language="text")


def display_chat_message(msg: dict):
    """Display a single assistant chat message with tags and reasoning."""
    st.write(msg["answer"])

    tags = ["Tool", "create_csv_agent", "LLMChain",
            "PromptTemplate", "ConversationBufferMemory"]
    tags_html = " ".join(
        f'<span class="langchain-tag">{c}</span>' for c in tags
    )
    st.markdown(
        f'<div style="margin-top:0.5rem;">'
        f'<span style="font-size:0.8rem;color:#777;">LangChain: </span>'
        f'{tags_html}</div>',
        unsafe_allow_html=True,
    )

    with st.expander("How the agent reached this answer"):
        st.markdown(
            f'<div class="reasoning-box">{msg["reasoning"]}</div>',
            unsafe_allow_html=True,
        )
        if msg.get("agent_steps"):
            st.markdown("**Agent execution log**")
            st.code(msg["agent_steps"], language="text")


# ===================================================================
# MAIN UI
# ===================================================================
st.header("LangChain Multi-Tool Assistant")
st.write(
    "Pick a tool from the sidebar. Every tool is built with LangChain "
    "components — Tools, Chains, Agents, and Memory."
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
    st.subheader("Text Summarizer Agent")
    st.write(
        "Paste any long text below. Two LangChain Tools run in sequence — "
        "the first splits and summarizes (map-reduce), the second explains "
        "the result with full reasoning."
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
            with st.spinner("Agent is reading and summarizing..."):
                result = run_summarizer_pipeline(llm, input_text)
            display_result(result, [
                "Tool", "load_summarize_chain",
                "RecursiveCharacterTextSplitter", "Document",
                "LLMChain", "PromptTemplate",
            ])


# ---- Code Explainer ----
elif tool_choice == "Code Explainer":
    st.subheader("Code Explainer Agent")
    st.write(
        "Paste a Python snippet below. Two LangChain Tools run in sequence — "
        "the first analyzes the code structure, the second explains it "
        "in plain English."
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
            with st.spinner("Agent is analyzing the code..."):
                result = run_code_explainer_pipeline(llm, input_code)
            display_result(result, [
                "Tool", "LLMChain", "PromptTemplate",
            ])


# ---- CSV Query Chatbot ----
elif tool_choice == "CSV Query Chatbot":
    st.subheader("CSV Query Chatbot")
    st.write(
        "Upload a CSV file and chat with your data. Two LangChain Tools "
        "handle each message — csv_data_query runs pandas code, "
        "result_explainer presents the answer. ConversationBufferMemory "
        "keeps track of the conversation for follow-ups."
    )

    if "csv_chat_history" not in st.session_state:
        st.session_state["csv_chat_history"] = []
    if "csv_file_path" not in st.session_state:
        st.session_state["csv_file_path"] = None

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        current_name = uploaded_file.name
        if st.session_state.get("csv_file_name") != current_name:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".csv", dir=tempfile.gettempdir()
            ) as tmp:
                tmp.write(uploaded_file.getvalue())
                st.session_state["csv_file_path"] = tmp.name
                st.session_state["csv_file_name"] = current_name
                st.session_state["csv_chat_history"] = []
                st.session_state.pop("csv_memory", None)

        file_path = st.session_state["csv_file_path"]

        df = pd.read_csv(file_path)
        with st.expander(
            f"Data preview — {len(df)} rows, {len(df.columns)} columns",
            expanded=False,
        ):
            st.dataframe(df.head(15), use_container_width=True)
            st.caption(f"Columns: {', '.join(df.columns.tolist())}")

        st.divider()

        # Render chat history
        for msg in st.session_state["csv_chat_history"]:
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.write(msg["content"])
                else:
                    display_chat_message(msg)

        # Chat input
        question = st.chat_input("Ask a question about your data...")

        if question:
            with st.chat_message("user"):
                st.write(question)

            st.session_state["csv_chat_history"].append({
                "role": "user",
                "content": question,
            })

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = run_csv_chatbot(llm, file_path, question)

                display_chat_message({
                    "answer": result["answer"],
                    "reasoning": result["reasoning"],
                    "agent_steps": result.get("agent_steps", ""),
                })

            st.session_state["csv_chat_history"].append({
                "role": "assistant",
                "content": result["answer"],
                "answer": result["answer"],
                "reasoning": result["reasoning"],
                "agent_steps": result.get("agent_steps", ""),
            })

    else:
        st.write("Upload a CSV file to start chatting with your data.")
