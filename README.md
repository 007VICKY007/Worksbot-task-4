# LangChain Multi-Tool Assistant

A single Streamlit application that bundles three practical LangChain-powered utilities into one interface. Built to demonstrate how LangChain tools, chains, and agents work together in a real application.

---

## What This Project Does

Instead of building three separate apps, this project packages three common text/data tasks into one clean dashboard. A user picks a tool from the sidebar, provides input, and gets a result. Every tool runs through LangChain under the hood.

| Tool | What it does | LangChain component used |
|------|-------------|--------------------------|
| **Text Summarizer** | Condenses long articles or reports into a short summary | `load_summarize_chain` (map_reduce strategy) |
| **Code Explainer** | Explains Python code in plain English, step by step | `LLMChain` with a custom `PromptTemplate` |
| **CSV Query Chatbot** | Conversational agent — ask questions about CSV data, follow up, dig deeper | `create_csv_agent` + conversation history |

---

## Why These Three Tools

These were chosen because they cover three different LangChain patterns that matter in real projects:

1. **Summarizer** shows how to handle text that is longer than the model's context window. LangChain splits the text into chunks, summarizes each chunk separately, then merges the chunk-level summaries into one final result. This is the map-reduce pattern.

2. **Code Explainer** shows the simplest and most common LangChain pattern: a prompt template fed into an LLM chain. The prompt tells the model to behave like a senior developer explaining code to a non-technical person. This pattern is the building block for most LangChain applications.

3. **CSV Query Chatbot** shows LangChain's agent framework running as a chatbot. Unlike a single-shot tool, the chatbot maintains conversation history so the user can ask follow-up questions like "what about the Engineering department?" or "show me the top 3 from that list." The agent uses the previous messages as context to understand references like "them," "those," and "the same column."

---

## Architecture

Every tool follows a two-step agent pattern: first it does the actual work (summarize, analyze, or query), then it passes the raw result to a second LLM call that explains the reasoning and presents a clean, human-readable answer.

```
User (Browser)
    |
    v
Streamlit UI  (app.py)
    |
    |--- Sidebar: tool selector
    |
    |--- Agent 1: Text Summarizer
    |       |
    |       +-- Step 1: RecursiveCharacterTextSplitter + load_summarize_chain (map_reduce)
    |       +-- Step 2: LLMChain explains what it did + presents clean summary
    |       +-- Output: REASONING (how it chunked and merged) + ANSWER (final summary)
    |
    |--- Agent 2: Code Explainer
    |       |
    |       +-- Step 1+2 combined: Single LLMChain with structured prompt
    |       +-- The prompt forces the LLM to first reason about the code structure,
    |       |   then explain it in plain English
    |       +-- Output: REASONING (what patterns it found) + ANSWER (explanation)
    |
    |--- Agent 3: CSV Query
            |
            +-- Step 1: create_csv_agent (writes pandas code, executes it, returns raw result)
            +-- Step 2: LLMChain takes the raw result and explains it in context
            +-- Output: REASONING (what columns/operations it used) + ANSWER (readable result)
            +-- Bonus: Agent execution log (the actual pandas steps it ran)
```

All three agents share the same `ChatOpenAI` instance (GPT-3.5-turbo, temperature 0.3), created once and reused across the session.

---

## Project Structure

```
langchain-multi-tool/
    app.py              # Main application (all UI + tool logic)
    requirements.txt    # Python dependencies
    .env.example        # Template — copy to .env and add your key
    .env                # Your actual API key (git-ignored, never committed)
    sample_data.csv     # Sample CSV file for testing the CSV Query tool
    README.md           # This file
    .gitignore          # Keeps .env and other local files out of git
```

The project is intentionally kept as a single `app.py` file. For a utility this size, splitting into multiple modules would add complexity without adding clarity.

---

## How to Run

### Prerequisites

- Python 3.10 or higher
- An OpenAI API key (get one at https://platform.openai.com/api-keys)

### Steps

1. Clone or download this project:

```bash
git clone https://github.com/007VICKY007/langchain-multi-tool.git
cd langchain-multi-tool
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up your API key:

```bash
cp .env.example .env
```

Open `.env` in any editor and replace the placeholder with your real key:

```
OPENAI_API_KEY=sk-proj-abc123...
```

The app loads this file on startup using `python-dotenv`. The sidebar field will be pre-filled automatically. You can also override it in the sidebar at any time without touching the file.

4. Run the app:

```bash
streamlit run app.py
```

5. Open the browser (Streamlit will print the URL, usually http://localhost:8501).

---

## How Each Agent Works (Technical Detail)

All three agents follow the same output pattern: they produce a **REASONING** section (what the agent did and why) and an **ANSWER** section (the clean, human-readable result). The UI shows the answer prominently and puts the reasoning in a collapsible panel.

### Text Summarizer Agent

**Problem:** A user pastes a 5,000-word article. Summarizing very long text in one shot produces poor results, and the user gets no visibility into what happened.

**Solution:** A two-step agent. Step 1 uses LangChain's `load_summarize_chain` (map-reduce) to produce a raw summary. Step 2 passes that raw summary to a second LLMChain that explains the process and presents a polished result.

**Step-by-step flow:**

1. The input text is split into chunks of ~3,000 characters using `RecursiveCharacterTextSplitter`.
2. Each chunk is summarized independently (map step).
3. All chunk summaries are merged into one (reduce step).
4. A second LLMChain receives the raw summary plus metadata (character count, chunk count) and writes both the REASONING and ANSWER sections.

**What the user sees:**

- **Answer:** A clean, readable paragraph summarizing the text.
- **Reasoning (expandable):** "I split the 12,400-character text into 5 chunks. The text was about renewable energy policy in Southeast Asia. I summarized each chunk separately, then merged the results into one coherent summary."

### Code Explainer Agent

**Problem:** A non-technical person needs to understand what a Python script does, and a raw explanation without context is hard to follow.

**Solution:** A single LLMChain with a structured prompt that forces the model to first analyze the code (identify patterns, structures, imports), then explain it step by step.

**Step-by-step flow:**

1. The user pastes Python code.
2. The prompt instructs the LLM to first reason about the code structure (what it identified), then explain it in plain English.
3. The LLM returns both sections in one response.

**What the user sees:**

- **Answer:** "This code connects to a website, downloads the page, and pulls out all the headlines. It uses two external libraries — one to fetch the page and one to read its structure."
- **Reasoning (expandable):** "I identified a function using the requests library for HTTP calls and BeautifulSoup for HTML parsing. The code uses a list comprehension to extract h2 tags."

### CSV Query Agent

**Problem:** A user asks "What departments exist in this data?" and gets back `['Engineering', 'Marketing', 'Sales', 'HR', 'Finance']`. That is technically correct but not useful — the user wants a sentence, not a Python list.

**Solution:** A two-step agent. Step 1 uses LangChain's `create_csv_agent` which autonomously writes and executes pandas code. Step 2 passes the raw result to a second LLMChain that explains what happened and presents the answer as a proper sentence.

**Step-by-step flow:**

1. The user uploads a CSV file. A preview is shown.
2. The user types a question in natural language.
3. The CSV agent decides what pandas code to run, executes it, and returns a raw result.
4. The intermediate steps (what code the agent wrote, what it observed) are captured.
5. A second LLMChain receives the raw result, the column names, and the row count, then writes the REASONING and ANSWER sections.

**What the user sees:**

- **Answer:** "There are 5 departments in the dataset: Engineering, Marketing, Sales, HR, and Finance. Engineering has the most employees with 5 people."
- **Reasoning (expandable):** "I looked at the 'department' column across all 15 rows. I used a unique-values operation to find the distinct departments, then counted the occurrences of each."
- **Agent execution log (expandable):** The actual pandas operations the agent ran, step by step.

**Key code:**

```python
csv_agent = create_csv_agent(
    llm, file_path,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    allow_dangerous_code=True,
    return_intermediate_steps=True,  # captures the agent's thought process
)
agent_result = csv_agent.invoke({"input": question})

# Step 2: explain the raw result
explain_chain = LLMChain(llm=llm, prompt=explain_prompt)
result = explain_chain.invoke({
    "question": question,
    "raw_answer": agent_result["output"],
    "columns": col_info,
    "row_count": str(len(df)),
})
```

**Note on `allow_dangerous_code=True`:** The CSV agent executes Python code (pandas operations) to answer questions. LangChain requires this flag as an explicit opt-in. In a production deployment, you would run this inside a sandboxed container.

---

## LangChain Concepts Demonstrated

| Concept | Where it appears | What it means |
|---------|-----------------|---------------|
| **Agent** | CSV Query (autonomous), all three (reasoning pattern) | An Agent decides its own steps — it reasons, acts, observes, and repeats |
| **Chain** | Summarizer, Code Explainer, explanation step in all | A Chain is a fixed sequence: prompt goes in, answer comes out |
| **PromptTemplate** | All three agents | A reusable template with placeholders filled at runtime |
| **Text Splitter** | Summarizer | Breaks long text into overlapping chunks that fit the context window |
| **Intermediate Steps** | CSV Query Chatbot | The agent's internal thought process — what code it wrote and what it observed |
| **Two-step pattern** | Summarizer and CSV Chatbot | Raw result from step 1 is explained by step 2 — separates computation from presentation |
| **Conversation History** | CSV Query Chatbot | Past messages are passed as context so the agent understands follow-up questions |

---

## Sample Questions for Testing

### Text Summarizer

Paste any long article or report. For a quick test, copy a Wikipedia article.

### Code Explainer

Try pasting this:

```python
import requests
from bs4 import BeautifulSoup

def scrape_headlines(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    headlines = [h.text.strip() for h in soup.find_all("h2")]
    return headlines[:10]
```

### CSV Query Chatbot

Upload the included `sample_data.csv` and try this conversation flow:

```
You:   How many employees are in each department?
Agent: There are 5 departments. Engineering has 5 employees, Marketing has 2, ...

You:   Which one has the highest average salary?
Agent: Among those departments, Finance has the highest average salary at 104,000...

You:   List the people in that department.
Agent: The Finance department has 2 employees: Lakshmi Pillai (Financial Analyst)
       and Amit Patel (Finance Manager)...

You:   Who earns more between them?
Agent: Amit Patel earns more at 130,000 compared to Lakshmi Pillai at 78,000...
```

The chatbot understands follow-ups like "that department," "between them," and "which one" because it keeps the full conversation history.

---

## Configuration

| Parameter | Default | Where to change |
|-----------|---------|----------------|
| LLM model | `gpt-3.5-turbo` | `get_llm()` function in `app.py` |
| Temperature | `0.3` | `get_llm()` function in `app.py` |
| Chunk size (summarizer) | `3000` chars | `build_summarizer_tool()` in `app.py` |
| Chunk overlap | `200` chars | `build_summarizer_tool()` in `app.py` |

To use GPT-4 instead, change the model name in `get_llm()`:

```python
ChatOpenAI(model="gpt-4", temperature=0.3, openai_api_key=api_key)
```

---

## Limitations

- **API cost:** Every request calls the OpenAI API. The summarizer makes multiple calls for long texts (one per chunk + one for the final merge).
- **CSV agent security:** The agent executes Python code. In production, run it inside a container or sandbox.
- **No conversation memory:** Each tool call is independent. The app does not maintain chat history across interactions.
- **File size:** Very large CSV files (100MB+) may hit memory or timeout limits in Streamlit.

---

## Tech Stack

- **Python 3.10+** — runtime
- **Streamlit** — web UI framework
- **LangChain** — LLM orchestration (chains, agents, tools, text splitters)
- **OpenAI GPT-3.5-turbo** — language model
- **pandas** — data manipulation (used internally by the CSV agent)

---

## License

MIT
