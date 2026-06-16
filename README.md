# HR Policy Chatbot

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-1C3C3C?style=flat&logo=chainlink&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6B35?style=flat)
![Groq](https://img.shields.io/badge/Groq-LLM_API-F55036?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

An end-to-end **Retrieval-Augmented Generation (RAG)** chatbot that answers
employee questions about company HR policies — grounded strictly in uploaded
policy documents, with zero hallucination of policy details.

Built with LangChain, ChromaDB, HuggingFace embeddings, and Groq (Llama 3),
this project demonstrates production-ready RAG architecture with conversational
memory, MMR-based diverse retrieval, and a clean Streamlit UI.

---

## What It Does

Employees can ask natural-language questions like:

> *"How many sick days do I get?"*  
> *"Can I carry over unused annual leave?"*  
> *"What is the process for raising a grievance?"*

The chatbot retrieves the most relevant passages from the HR policy documents
and generates a precise, cited answer — declining to answer anything not
covered in the documents.

---

## Evaluation Results

This project includes a dedicated evaluation pipeline (`evaluate.py`) that
runs a 20-question golden test set through four independent test suites.

| Metric | Result |
|---|---|
| **Retrieval Accuracy** (top-K) | 94.7% (18/19) |
| **Groundedness Score** (LLM-as-judge, 1–5) | 4.45 / 5.0 |
| **Guardrail Compliance** | 5/5 |
| **Multi-turn Memory Retention** | 2/2 |
| **Overall System Readiness** | 95.9% |

**What's measured:**
- **Retrieval Accuracy** — whether the correct policy document appears
  in the top-K retrieved chunks for each query
- **Groundedness Score** — a separate LLM judges each answer for factual
  accuracy and whether it's fully supported by retrieved context
- **Guardrail Compliance** — tests against off-topic questions, prompt
  injection, hallucination probes, and role-specificity edge cases
- **Multi-turn Memory** — confirms follow-up questions with pronoun
  references ("Can those days be carried over?") are correctly resolved
  using conversation history

Run the evaluation yourself:
```bash
python evaluate.py
```

---

## Key Technical Highlights

### Retrieval-Augmented Generation (RAG)
Rather than relying on an LLM's training knowledge (which knows nothing
about company-specific policies), this system retrieves relevant document
chunks at query time and feeds them as context to the LLM. The LLM acts
purely as a reader and writer — all knowledge comes from your documents.

### MMR Retrieval Strategy
The retriever uses **Maximal Marginal Relevance (MMR)** instead of pure
cosine similarity search. Here's why this matters:

| Strategy | Behaviour | Problem |
|---|---|---|
| Similarity search | Returns the K most similar chunks | Can return K near-identical chunks from the same paragraph, giving the LLM redundant context |
| **MMR** | Balances relevance AND diversity | Returns chunks that are relevant to the query but maximally different from each other — covering more of the policy space |

For HR documents where the same topic (e.g. leave entitlement) is
mentioned across multiple sections, MMR ensures the LLM receives
complementary context rather than five copies of the same sentence.

### Conversational Memory
Uses `ConversationBufferWindowMemory` (k=6 turns) with a custom condense
prompt that rewrites follow-up questions into standalone queries before
retrieval. This means pronoun references ("Can those days be carried over?")
are correctly resolved to their full concept before hitting the vector store.

### Guardrailed Responses
A carefully engineered system prompt constrains the LLM to:
- Answer only from retrieved context
- Match role/person specificity (CEO salary ≠ engineer salary)
- Decline off-topic questions cleanly
- Suppress source citations on "I don't know" responses

---

## Project Structure
```
hr-policy-rag-chatbot/
│
├── config.py          # Central config: model names, chunk size,
│                      # overlap, retriever K, all file paths
│
├── ingest.py          # Document ingestion pipeline:
│                      # load PDFs/DOCX → chunk → embed → store in ChromaDB
│
├── chain.py           # Core RAG chain: loads vector store, builds
│                      # MMR retriever, loads LLM, assembles
│                      # ConversationalRetrievalChain with memory
│
├── test_chain.py      # Offline test suite: smoke tests, multi-turn
│                      # memory tests, guardrail tests, interactive mode
│
├── evaluate.py        # Evaluation pipeline: retrieval accuracy,
│                      # LLM-as-judge groundedness, guardrail tests,
│                      # multi-turn memory tests
│
├── golden_test_set.json  # 20-question benchmark set used by evaluate.py
│
├── logger.py          # SQLite logging for queries and eval results
│
├── requirements.txt   # All Python dependencies
├── .env               # API keys — never committed (see .gitignore)
│
└── docs/              # HR policy PDFs/DOCX files go here
```

> ⚠️ **Disclaimer on Sample Documents**
>
> The policy documents provided in the `docs/` folder are AI-generated for
> the sole purpose of demonstrating this project's capabilities. They are
> not based on, affiliated with, or representative of the HR policies of
> any existing organization.

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/pramitbose2024/hr-policy-chatbot.git
cd hr-policy-chatbot
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Create a `.env` file in the project root:
GROQ_API_KEY=your_groq_api_key_here

Get a free Groq API key at [console.groq.com](https://console.groq.com) —
no credit card required.

---

## Running the Project

### Step 1 — Build the knowledge base
```bash
python ingest.py
```

This loads your documents, splits them into chunks, generates embeddings
using `all-MiniLM-L6-v2`, and stores them in a local ChromaDB vector store.
Run this once per document update.

### Step 2 — Test the chain (optional but recommended)
```bash
python test_chain.py
```

Runs automated smoke tests, multi-turn memory tests, and guardrail
tests in the terminal before wiring to any UI.

### Step 3 — Run the evaluation suite
```bash
python evaluate.py
```

Runs the full benchmark and prints a summary report — see
**Evaluation Results** above for what to expect.

---

## Skills Demonstrated

This project was built to demonstrate end-to-end competency across
the full RAG stack — from document processing to evaluation.

| Area | What's demonstrated |
|---|---|
| **RAG Architecture** | Full pipeline: document loading → chunking → embedding → vector storage → retrieval → generation |
| **Vector Search** | ChromaDB with MMR retrieval; understanding of similarity vs diversity trade-offs in dense retrieval |
| **LLM Orchestration** | LangChain `ConversationalRetrievalChain` with custom condense and answer prompts |
| **Prompt Engineering** | Multi-rule system prompt with specificity guards, role-matching rules, and guardrail patterns |
| **Conversational Memory** | `ConversationBufferWindowMemory` with custom condense prompt that resolves pronouns across turns |
| **Production Structure** | Separation of config, ingestion, chain logic, and UI into distinct modules; no hardcoded values |
| **Evaluation Mindset** | Dedicated 4-suite evaluation pipeline with golden test set and LLM-as-judge scoring |

### Why RAG over fine-tuning?
Fine-tuning an LLM on HR documents would bake the knowledge into model
weights — meaning every policy update requires an expensive retraining cycle.
RAG keeps the knowledge in the document store: update a PDF, re-run
`ingest.py`, and the chatbot answers from the new policy immediately.
This is the architecturally correct approach for any domain where the
source of truth changes regularly.

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq API — Llama 3.1 8B Instant |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | ChromaDB (local persistent) |
| Retrieval | MMR (Maximal Marginal Relevance) |
| Orchestration | LangChain `ConversationalRetrievalChain` |
| Document Parsing | PyPDF, docx2txt |
| Evaluation | LLM-as-judge groundedness scoring |

---

## License

MIT License — see [LICENSE](LICENSE) for details.
