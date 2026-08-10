import os
import tempfile
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import AgentType
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
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — tool selector only
# ---------------------------------------------------------------------------
api_key = os.getenv("OPENAI_API_KEY", "")

with st.sidebar:
    st.title("Tools")

    tool_choice = st.radio(
        "Select a tool",
        options=["Text Summarizer", "Code Explainer", "CSV Query Chatbot"],
        index=0,
    )

    # Show clear chat button only for CSV chatbot
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
        )
        st.session_state["_api_key"] = api_key
    return st.session_state["llm"]


# ===================================================================
# AGENT 1 — TEXT SUMMARIZER AGENT
# ===================================================================
def run_summarizer_agent(llm: ChatOpenAI, text: str) -> dict:
    """
    Two-step agent:
      Step 1 — Chunk and summarize (map-reduce chain)
      Step 2 — LLM explains what it did and presents the summary clearly
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=c) for c in chunks]

    chain = load_summarize_chain(llm, chain_type="map_reduce")
    raw_summary = chain.invoke(docs)["output_text"]

    explain_prompt = PromptTemplate(
        input_variables=["original_length", "chunk_count", "raw_summary"],
        template=(
            "You are a summarization agent. You just processed a text and "
            "produced a raw summary. Now explain your work to the user.\n\n"
            "Details of what you did:\n"
            "- The original text was approximately {original_length} characters long.\n"
            "- You split it into {chunk_count} chunk(s) to handle it properly.\n"
            "- You summarized each chunk, then merged them into one summary.\n\n"
            "Raw summary you produced:\n{raw_summary}\n\n"
            "Now write two sections:\n\n"
            "REASONING:\n"
            "Explain in 2-3 sentences what you did — how many chunks, "
            "what the text was about, and how you approached it.\n\n"
            "ANSWER:\n"
            "Present the final summary in clear, readable sentences. "
            "Do not use bullet points. Write it as a proper paragraph.\n\n"
            "Use exactly these headers: REASONING: and ANSWER:"
        ),
    )

    explain_chain = LLMChain(llm=llm, prompt=explain_prompt)
    result = explain_chain.invoke({
        "original_length": str(len(text)),
        "chunk_count": str(len(chunks)),
        "raw_summary": raw_summary,
    })

    return _parse_reasoning_answer(result["text"], raw_summary)


# ===================================================================
# AGENT 2 — CODE EXPLAINER AGENT
# ===================================================================
def run_code_explainer_agent(llm: ChatOpenAI, code: str) -> dict:
    """
    Single LLMChain that reasons about code structure, then explains it.
    """
    prompt = PromptTemplate(
        input_variables=["code"],
        template=(
            "You are a code analysis agent. A user has given you Python code "
            "and wants to understand what it does.\n\n"
            "Your job:\n"
            "1. First, figure out what this code is doing. Identify the key "
            "   parts — functions, loops, imports, data structures, logic.\n"
            "2. Then explain it in plain English that a non-technical manager "
            "   can understand.\n\n"
            "Code:\n```python\n{code}\n```\n\n"
            "Write your response in exactly two sections:\n\n"
            "REASONING:\n"
            "Describe what you identified in the code — what language features "
            "are used, what the structure looks like, what patterns you noticed. "
            "Keep it to 3-5 sentences.\n\n"
            "ANSWER:\n"
            "Now explain the code to the user in simple terms. "
            "Start with what the code does overall in one sentence. "
            "Then walk through it step by step. "
            "If there are any issues or improvements, mention them at the end. "
            "Use plain English throughout.\n\n"
            "Use exactly these headers: REASONING: and ANSWER:"
        ),
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.invoke({"code": code})
    return _parse_reasoning_answer(result["text"], result["text"])


# ===================================================================
# AGENT 3 — CSV QUERY CHATBOT
# ===================================================================
def run_csv_chatbot(llm: ChatOpenAI, file_path: str, question: str,
                    chat_history: list) -> dict:
    """
    Conversational CSV agent. Takes the full chat history into account
    so the user can ask follow-up questions.

    Step 1 — Build context from past conversation
    Step 2 — CSV agent queries the data
    Step 3 — LLM explains the result in plain English
    """

    # Build conversation context from history
    context_lines = []
    for msg in chat_history:
        role = "User" if msg["role"] == "user" else "Agent"
        # Only include the answer part, not reasoning, to keep context concise
        content = msg.get("answer", msg["content"]) if role == "Agent" else msg["content"]
        context_lines.append(f"{role}: {content}")
    conversation_context = "\n".join(context_lines[-10:])  # last 10 messages max

    # Build the full question with context for follow-ups
    if conversation_context:
        full_question = (
            f"Previous conversation for context:\n{conversation_context}\n\n"
            f"Current question: {question}\n\n"
            f"If the current question references something from the previous "
            f"conversation (like 'them', 'those', 'that department', 'the same', "
            f"'more details', etc.), use the context to understand what is meant. "
            f"Answer the current question only."
        )
    else:
        full_question = question

    # Step 1: CSV agent queries the data
    csv_agent = create_csv_agent(
        llm,
        file_path,
        verbose=False,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        allow_dangerous_code=True,
        return_intermediate_steps=True,
    )

    agent_result = csv_agent.invoke({"input": full_question})
    raw_answer = agent_result["output"]

    # Collect intermediate steps
    steps = agent_result.get("intermediate_steps", [])
    reasoning_log = []
    for i, (action, observation) in enumerate(steps, 1):
        reasoning_log.append(
            f"Step {i}:\n"
            f"  Thought: {action.log.strip()}\n"
            f"  Action: {action.tool}\n"
            f"  Input: {action.tool_input}\n"
            f"  Result: {str(observation).strip()}"
        )
    agent_steps = "\n\n".join(reasoning_log) if reasoning_log else ""

    # Read CSV info for the explanation step
    df = pd.read_csv(file_path)
    col_info = ", ".join(df.columns.tolist())

    # Step 2: LLM explains the result clearly
    explain_prompt = PromptTemplate(
        input_variables=["question", "raw_answer", "columns", "row_count",
                         "conversation_context"],
        template=(
            "You are a data analysis chatbot. You just queried a CSV file to "
            "answer a user's question. Now present your findings clearly.\n\n"
            "Dataset info:\n"
            "- Columns: {columns}\n"
            "- Total rows: {row_count}\n\n"
            "Previous conversation:\n{conversation_context}\n\n"
            "User's current question: {question}\n\n"
            "Raw result from your query: {raw_answer}\n\n"
            "Write your response in exactly two sections:\n\n"
            "REASONING:\n"
            "Explain in 2-4 sentences what you did to get this answer. "
            "Mention which columns you looked at, what operation you performed "
            "(counted, averaged, filtered, grouped, etc.), and how you arrived "
            "at the result. If this was a follow-up question, mention what "
            "context you used from the previous conversation.\n\n"
            "ANSWER:\n"
            "Present the result in a clear, conversational way. "
            "Do not dump raw lists or numbers. Write proper sentences. "
            "If the result is a list, write it as a readable sentence. "
            "If the result is a number, put it in context. "
            "If this is a follow-up, connect it naturally to the previous answer. "
            "Be specific and complete. Keep a conversational tone.\n\n"
            "Use exactly these headers: REASONING: and ANSWER:"
        ),
    )

    explain_chain = LLMChain(llm=llm, prompt=explain_prompt)
    result = explain_chain.invoke({
        "question": question,
        "raw_answer": str(raw_answer),
        "columns": col_info,
        "row_count": str(len(df)),
        "conversation_context": conversation_context or "(No previous messages)",
    })

    parsed = _parse_reasoning_answer(result["text"], raw_answer)
    if agent_steps:
        parsed["agent_steps"] = agent_steps

    return parsed


# ===================================================================
# HELPER — Parse REASONING: / ANSWER: from LLM output
# ===================================================================
def _parse_reasoning_answer(text: str, fallback: str) -> dict:
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


# ===================================================================
# HELPER — Display result with reasoning (for summarizer + code)
# ===================================================================
def display_result(result: dict):
    st.markdown("**Answer**")
    st.markdown(
        f'<div class="result-box">{result["answer"]}</div>',
        unsafe_allow_html=True,
    )

    with st.expander("How the agent reached this answer"):
        st.markdown(
            f'<div class="reasoning-box">{result["reasoning"]}</div>',
            unsafe_allow_html=True,
        )
        if "agent_steps" in result:
            st.markdown("**Agent execution log**")
            st.code(result["agent_steps"], language="text")


# ===================================================================
# MAIN UI
# ===================================================================
st.header("LangChain Multi-Tool Assistant")
st.write(
    "Pick a tool from the sidebar, provide your input, and the agent will "
    "process it, show its reasoning, and give you a clear answer."
)

if not api_key:
    st.error(
        "OPENAI_API_KEY not found. "
        "Add your key to the .env file and restart the app."
    )
    st.stop()

llm = get_llm(api_key)

# ---- Text Summarizer Agent ----
if tool_choice == "Text Summarizer":
    st.subheader("Text Summarizer Agent")
    st.write(
        "Paste any long text below. The agent splits it into manageable chunks, "
        "summarizes each one, merges the results, and explains what it did."
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
                result = run_summarizer_agent(llm, input_text)
            display_result(result)

# ---- Code Explainer Agent ----
elif tool_choice == "Code Explainer":
    st.subheader("Code Explainer Agent")
    st.write(
        "Paste a Python snippet below. The agent will analyze the code structure, "
        "identify what each part does, and explain it in plain English."
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
                result = run_code_explainer_agent(llm, input_code)
            display_result(result)

# ---- CSV Query Chatbot ----
elif tool_choice == "CSV Query Chatbot":
    st.subheader("CSV Query Chatbot")
    st.write(
        "Upload a CSV file and chat with your data. Ask questions, "
        "follow up on answers, and dig deeper — the agent remembers "
        "the full conversation."
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
                st.session_state["csv_chat_history"] = []  # reset chat for new file

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
                    # Assistant message — show answer + collapsible reasoning
                    st.write(msg["answer"])
                    with st.expander("How the agent reached this answer"):
                        st.markdown(
                            f'<div class="reasoning-box">{msg["reasoning"]}</div>',
                            unsafe_allow_html=True,
                        )
                        if msg.get("agent_steps"):
                            st.markdown("**Agent execution log**")
                            st.code(msg["agent_steps"], language="text")

        # Chat input
        question = st.chat_input("Ask a question about your data...")

        if question:
            # Show user message
            with st.chat_message("user"):
                st.write(question)

            # Add to history
            st.session_state["csv_chat_history"].append({
                "role": "user",
                "content": question,
            })

            # Run the agent
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = run_csv_chatbot(
                        llm,
                        file_path,
                        question,
                        st.session_state["csv_chat_history"],
                    )

                # Show the answer
                st.write(result["answer"])

                # Show reasoning in expander
                with st.expander("How the agent reached this answer"):
                    st.markdown(
                        f'<div class="reasoning-box">{result["reasoning"]}</div>',
                        unsafe_allow_html=True,
                    )
                    if result.get("agent_steps"):
                        st.markdown("**Agent execution log**")
                        st.code(result["agent_steps"], language="text")

            # Save assistant message to history
            st.session_state["csv_chat_history"].append({
                "role": "assistant",
                "content": result["answer"],
                "answer": result["answer"],
                "reasoning": result["reasoning"],
                "agent_steps": result.get("agent_steps", ""),
            })

    else:
        st.write("Upload a CSV file to start chatting with your data.")