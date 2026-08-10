# LangChain Multi-Tool Assistant

A single Streamlit application that bundles three practical LangChain-powered utilities into one interface. Built to demonstrate how LangChain tools, chains, and agents work together in a real application.

---

## What This Project Does

Instead of building three separate apps, this project packages three common text/data tasks into one clean dashboard. A user picks a tool from the sidebar, provides input, and gets a result. Every tool runs through LangChain under the hood.

| Tool | What it does | LangChain component used |
|------|-------------|--------------------------|
| **Text Summarizer** | Condenses long articles or reports into a short summary | `load_summarize_chain` (map_reduce strategy) |
| **Code Explainer** | Explains Python code in plain English, step by step | `LLMChain` with a custom `PromptTemplate` |
| **CSV Query** | Answers natural-language questions about uploaded CSV data | `create_csv_agent` (writes and runs pandas code internally) |

---

## Why These Three Tools

These were chosen because they cover three different LangChain patterns that matter in real projects:

1. **Summarizer** shows how to handle text that is longer than the model's context window. LangChain splits the text into chunks, summarizes each chunk separately, then merges the chunk-level summaries into one final result. This is the map-reduce pattern.

2. **Code Explainer** shows the simplest and most common LangChain pattern: a prompt template fed into an LLM chain. The prompt tells the model to behave like a senior developer explaining code to a non-technical person. This pattern is the building block for most LangChain applications.

3. **CSV Query** shows LangChain's agent framework. Unlike a simple chain where we control every step, the agent decides on its own what pandas code to write and execute to answer the user's question. This is the most autonomous pattern and demonstrates how LangChain agents reason and act.

---

## Architecture

```
User (Browser)
    |
    v
Streamlit UI  (app.py)
    |
    |--- Sidebar: API key input + tool selector
    |
    |--- Tool 1: Text Summarizer
    |       |
    |       +-- RecursiveCharacterTextSplitter (splits long text into chunks)
    |       +-- load_summarize_chain (map_reduce)
    |       +-- ChatOpenAI (LLM)
    |
    |--- Tool 2: Code Explainer
    |       |
    |       +-- PromptTemplate (structured prompt)
    |       +-- LLMChain (prompt + LLM)
    |       +-- ChatOpenAI (LLM)
    |
    |--- Tool 3: CSV Query
            |
            +-- create_csv_agent (autonomous agent)
            +-- pandas (data manipulation, executed by agent)
            +-- ChatOpenAI (LLM)
```

All three tools share the same `ChatOpenAI` instance (GPT-3.5-turbo, temperature 0.3), which is created once and reused across the session.

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

## How Each Tool Works (Technical Detail)

### Text Summarizer

**Problem:** A user pastes a 5,000-word article. GPT-3.5-turbo has a context window limit, and even if the text fits, summarizing very long text in one shot often produces poor results.

**Solution:** LangChain's `load_summarize_chain` with the `map_reduce` strategy.

**Step-by-step flow:**

1. The input text is split into chunks of ~3,000 characters each using `RecursiveCharacterTextSplitter`. Adjacent chunks overlap by 200 characters so no sentence is cut in half.
2. Each chunk is wrapped in a LangChain `Document` object.
3. The map step: each chunk is summarized independently by the LLM.
4. The reduce step: all chunk-level summaries are combined and summarized again into one final output.

**Key code:**

```python
splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
chunks = splitter.split_text(text)
docs = [Document(page_content=c) for c in chunks]
chain = load_summarize_chain(llm, chain_type="map_reduce")
result = chain.invoke(docs)
```

### Code Explainer

**Problem:** A manager or a junior developer needs to understand what a Python script does, without reading the code line by line.

**Solution:** A `PromptTemplate` + `LLMChain`. The prompt instructs the model to act as a senior developer explaining code to a non-technical audience.

**Step-by-step flow:**

1. The user pastes Python code.
2. The code is injected into a prompt template that sets the tone and format.
3. The LLMChain sends the completed prompt to the model.
4. The model returns a plain-English walkthrough.

**Key code:**

```python
prompt = PromptTemplate(
    input_variables=["code"],
    template="You are a senior Python developer explaining code to a "
             "non-technical manager. ... Code:\n{code}\nExplanation:"
)
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.invoke({"code": code})
```

### CSV Query

**Problem:** A user has a CSV file and wants answers — "What is the average salary by department?" — without writing pandas code.

**Solution:** LangChain's `create_csv_agent`. This creates an autonomous agent that reads the CSV, figures out the right pandas operations, executes them, and returns the answer.

**Step-by-step flow:**

1. The user uploads a CSV file. It is saved to a temporary file.
2. A preview (first 10 rows) is shown so the user can see the column names.
3. The user types a question in natural language.
4. The agent reads the CSV, decides what pandas code to run, executes it in a sandboxed environment, and returns the answer.

**Key code:**

```python
agent = create_csv_agent(
    llm,
    file_path,
    verbose=False,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    allow_dangerous_code=True,
)
result = agent.invoke({"input": question})
```

**Note on `allow_dangerous_code=True`:** The CSV agent needs to execute Python code (pandas operations) to answer questions. LangChain requires this flag as an explicit opt-in. In a production deployment, you would run this inside a sandboxed container.

---

## LangChain Concepts Demonstrated

| Concept | Where it appears | What it means |
|---------|-----------------|---------------|
| **Tool** | All three utilities are wrapped as `Tool` objects | A Tool is a function with a name and description that LangChain can call |
| **Chain** | Summarizer and Code Explainer | A Chain is a fixed sequence of steps: prompt goes in, answer comes out |
| **Agent** | CSV Query | An Agent decides its own steps. It reasons about what to do, acts, observes the result, and repeats until it has an answer |
| **PromptTemplate** | Code Explainer | A reusable template with placeholders that get filled at runtime |
| **Text Splitter** | Summarizer | Breaks long text into overlapping chunks that fit the model's context window |
| **Document** | Summarizer | LangChain's standard wrapper for a piece of text plus optional metadata |

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

### CSV Query

Upload the included `sample_data.csv` and try:

- "How many employees are in each department?"
- "What is the average salary in Engineering?"
- "Who has the most experience?"
- "List all employees in Chennai."

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
#   W o r k s b o t - t a s k - 4  
 