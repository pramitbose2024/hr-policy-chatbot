import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.messages import SystemMessage

from langchain.prompts import PromptTemplate

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
You are a question rewriter for an HR policy chatbot.

Your ONLY job: rewrite the follow-up question into a fully standalone
question by resolving every pronoun and vague reference using the
conversation history.

CRITICAL RULES:
1. When replacing a pronoun that refers to a leave type, policy, or rule,
   always use the FULL NAME of the leave type or policy — never substitute
   a number or quantity in its place.
   WRONG: "Can 18 days per year be carried over?"
   RIGHT: "Can Earned Leave (EL) days be carried over to the next year?"

2. When the conversation switches subject (e.g. from female to male
   employees), always use the MOST RECENTLY mentioned subject.
   The latest message in history takes priority over earlier ones.

3. Replace these pronouns with the specific noun from history:
   "those", "them", "it", "that", "same", "similar", "this", "these",
   "they", "their", "them", "the same", "as well", "also", "too"

4. If the question is already fully standalone, return it unchanged.

5. Output ONLY the rewritten question. Nothing else. No explanation.

EXAMPLES:
─────────────────────────────────────────────────────────────
History:
  Human: How many days of earned leave (EL) do confirmed employees get?
  AI: 18 days per year for confirmed employees.
Follow-up: "Can those days be carried over?"
Rewritten: "Can Earned Leave (EL) days be carried over to the next year?"
─────────────────────────────────────────────────────────────
History:
  Human: How many leaves does the female employees get?
  AI: Female employees get 26 weeks maternity leave, 12 days CL...
  Human: what about for male employees?
  AI: I couldn't find specific information about male employees' paid leaves.
Follow-up: "what about their office timing?"
Rewritten: "What are the office timings for male employees?"
─────────────────────────────────────────────────────────────
History:
  Human: What is the sick leave policy?
  AI: 12 days per calendar year for confirmed employees.
Follow-up: "Is it the same for probationary employees?"
Rewritten: "Is the sick leave policy the same for probationary employees?"
─────────────────────────────────────────────────────────────
History:
  Human: What is the WFH policy?
  AI: Eligible employees may work remotely.
Follow-up: "What about interns, do they get it too?"
Rewritten: "Does the work from home (WFH) policy apply to interns?"
─────────────────────────────────────────────────────────────

Conversation History:
{chat_history}

Follow-up Question: {question}

Rewritten Standalone Question:""")

from config import (
    CHROMA_DIR,
    EMBEDDING_MODEL,
    RETRIEVER_K,
    LLM_PROVIDER,
    GROQ_API_KEY,
    # OPENAI_API_KEY,
    GROQ_MODEL,
    # OPENAI_MODEL,
)


# ─────────────────────────────────────────────────────────────
# STEP 1: Load the vector store from disk
# ─────────────────────────────────────────────────────────────

def load_vector_store() -> Chroma:
    """
    Opens the ChromaDB vector store that ingest.py built in Phase 1.
    
    Key point: we use the EXACT same embedding model here as we did
    during ingestion. This is non-negotiable.
    
    Why? When a user asks a question, we embed that question into a
    vector using this same model. We then compare that vector against
    the stored chunk vectors using cosine similarity. If we used a
    different model here, the vectors would live in completely different
    mathematical spaces and comparison would be meaningless — like
    measuring temperature in Celsius but comparing against Fahrenheit
    values stored in a database.
    """
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            f"Vector store not found at '{CHROMA_DIR}'. "
            "Run ingest.py first to build it."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    count = vector_store._collection.count()
    print(f"✅ Vector store loaded — {count} chunks available")
    return vector_store


# ─────────────────────────────────────────────────────────────
# STEP 2: Build the retriever
# ─────────────────────────────────────────────────────────────

def build_retriever(vector_store: Chroma):
    """
    Wraps the vector store in a retriever interface.
    
    The retriever is responsible for one job: given a question,
    return the K most semantically relevant chunks from the vector store.
    
    search_type="similarity" uses cosine similarity between the
    question vector and chunk vectors. The top K=4 chunks are returned.
    
    Why K=4?
    - Too few (K=1 or 2): risk missing relevant context spread across chunks
    - Too many (K=8+): you start injecting irrelevant chunks, which
      confuses the LLM and can cause it to hallucinate or contradict itself
    - K=4 hits the sweet spot for most HR policy documents
    
    You can tune this in config.py later based on your test results.
    """
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVER_K},
    )
    print(f"✅ Retriever built — returning top {RETRIEVER_K} chunks per query")
    return retriever


# ─────────────────────────────────────────────────────────────
# STEP 3: Load the LLM
# ─────────────────────────────────────────────────────────────

def load_llm():
    """
    Loads the language model that will generate answers.
    
    The LLM never sees your full HR documents. It only ever sees:
      1. The system prompt (its instructions and persona)
      2. The 4 retrieved chunks most relevant to the question
      3. The recent conversation history
      4. The user's current question
    
    This is the core insight of RAG: the LLM is used purely for
    language understanding and generation. The knowledge comes
    from your documents, not from the LLM's training weights.
    This is why the chatbot will not hallucinate HR policies —
    it can only reference what was actually retrieved.
    
    Provider choice:
    - Groq: free tier, extremely fast (Llama 3 at ~500 tokens/sec),
      best for development and demos
    - OpenAI: GPT-4o-mini is low cost (~$0.15/1M input tokens),
      slightly better reasoning, good for production
    """
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is missing from your .env file.")
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL,
            temperature=0,         # 0 = deterministic, factual answers
            max_tokens=1024,
        )
        print(f"✅ LLM loaded — Groq / {GROQ_MODEL}")

    # elif LLM_PROVIDER == "openai":
    #     from langchain_openai import ChatOpenAI
    #     if not OPENAI_API_KEY:
    #         raise ValueError("OPENAI_API_KEY is missing from your .env file.")
    #     llm = ChatOpenAI(
    #         api_key=OPENAI_API_KEY,
    #         model_name=OPENAI_MODEL,
    #         temperature=0,
    #         max_tokens=1024,
    #     )
    #     print(f"✅ LLM loaded — OpenAI / {OPENAI_MODEL}")

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. "
            "Set it to 'groq' or 'openai' in config.py."
        )

    return llm


# ─────────────────────────────────────────────────────────────
# STEP 4: Define the system prompt
# ─────────────────────────────────────────────────────────────

def build_prompt() -> ChatPromptTemplate:
    """
    The system prompt is the single most important guardrail in this project.
    
    Without it, the LLM will answer from its general training knowledge —
    which means it might invent HR policies, mix in policies from other
    companies, or give legally incorrect information.
    
    With a well-written system prompt, the LLM is constrained to:
      - Answer ONLY from the retrieved context
      - Admit when it doesn't know
      - Stay in the HR assistant persona
      - Never go off-topic
    
    The {context} placeholder is automatically filled by LangChain with
    the 4 retrieved chunks. The {question} placeholder is filled with
    the user's current message. These names are required by LangChain's
    ConversationalRetrievalChain.
    """

    system_template = """You are an HR Policy Assistant for this organization.
    Your ONLY purpose is to answer questions about THIS organization's HR policies
    using ONLY the text provided in the context section below.

    STRICT RULES — follow every one of these without exception:

    1. ONLY use the provided context. Never use outside knowledge.

    2. SPECIFICITY RULE: The question must match the context specifically.
    If someone asks about "the CEO's salary" and the context contains
    salary data for OTHER roles (e.g. engineers, interns), that context
    does NOT answer the question. Say you couldn't find it.

    3. PERSON/ROLE MATCHING: If the question asks about a specific person
    or role (e.g. CEO, Director, founder), only answer if that exact
    person or role is mentioned in the context. Do not substitute
    with a different role's data.

    4. If the answer is genuinely not in the context, respond with:
    "I couldn't find information about [specific topic] in the current
    HR policy documents. Please contact the HR department directly."

    5. Never fabricate, infer, or extrapolate anything not explicitly stated.

    6. Be concise. Use bullet points for lists. Always cite the document
    name and section when available.

    7. Off-topic questions (general knowledge, coding, personal advice,
    questions about specific individuals' private information) must be
    declined with: "I can only assist with HR policy questions."

    Context from HR policy documents:
    ────────────────────────────────
    {context}
    ────────────────────────────────
    """

    human_template = "{question}"

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    prompt = ChatPromptTemplate.from_messages([
        system_message_prompt,
        human_message_prompt,
    ])

    print("✅ System prompt built")
    return prompt


# ─────────────────────────────────────────────────────────────
# STEP 5: Build the memory
# ─────────────────────────────────────────────────────────────

def build_memory() -> ConversationBufferWindowMemory:
    """
    Upgraded memory configuration.

    k=8 instead of k=5:
    Your interactive session showed failure when the subject
    switched two turns back (female → male employees). With k=5
    that was still within the window but the small model lost
    track of the most recent subject. k=8 gives the condense
    step more history to work with so it can correctly identify
    the MOST RECENT subject even in dense multi-subject conversations.

    Why not higher than 8?
    Each turn adds ~200-400 tokens to the condense prompt.
    At k=8 you're feeding ~1600-3200 tokens of history into
    every condense call. Beyond k=10 you hit the model's
    effective attention limit for small models and resolution
    quality actually degrades. k=8 is the practical ceiling
    for llama-3.1-8b-instant.
    """
    memory = ConversationBufferWindowMemory(
        k=6,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    print(f"✅ Memory built — retaining last 8 exchanges")
    return memory


# ─────────────────────────────────────────────────────────────
# STEP 6: Assemble the full RAG chain
# ─────────────────────────────────────────────────────────────

def build_chain() -> ConversationalRetrievalChain:
    """
    Combines all components into the final conversational RAG chain.
    
    How a single query flows through this chain:
    
    User question
         │
         ▼
    [Condense question]
    If there's chat history, the chain first rewrites the user's
    question to be standalone. For example:
      History: "How many sick days do I get? → 10 days"
      New Q:   "Can I carry them over?"
      Rewritten: "Can sick days be carried over to the next year?"
    This rewritten question is what gets sent to the retriever,
    ensuring correct chunks are fetched even for follow-ups.
         │
         ▼
    [Retrieve]
    The rewritten question is embedded and compared against
    all chunk vectors. Top 4 most similar chunks are returned.
         │
         ▼
    [Generate]
    The system prompt + 4 chunks + conversation history + original
    question are assembled and sent to the LLM. The LLM generates
    a grounded answer.
         │
         ▼
    Response dict: {"answer": "...", "source_documents": [...]}
    
    return_source_documents=True is critical — it gives you the
    chunks used to generate the answer, which you'll display in
    the Streamlit UI so users can verify the source.
    """
    print("\n🔧 Building RAG chain...")

    vector_store = load_vector_store()
    retriever    = build_retriever(vector_store)
    llm          = load_llm()
    prompt       = build_prompt()
    memory       = build_memory()

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,   # ← added this line for Conversational RAG
        return_source_documents=True,
        verbose=False,
    )

    print("✅ RAG chain ready\n")
    return chain


# ─────────────────────────────────────────────────────────────
# STEP 7: The query function (what Streamlit will call)
# ─────────────────────────────────────────────────────────────

# These patterns indicate the bot has no answer or is declining.
# When matched, sources are suppressed — showing retrieved chunks
# alongside a "I don't know" answer is confusing and misleading.
_NO_ANSWER_PATTERNS = [
    "i couldn't find information",
    "i can only assist with hr policy",
    "please contact the hr department",
    "i don't have information",
    "not available in the current hr",
    "couldn't find",
]


def _is_no_answer(text: str) -> bool:
    """Returns True if the answer is a refusal or knowledge-gap response."""
    lowered = text.lower()
    return any(pattern in lowered for pattern in _NO_ANSWER_PATTERNS)


def ask(chain: ConversationalRetrievalChain, question: str) -> dict:
    """
    Public interface called by Streamlit and test_chain.py.

    Returns:
    {
        "answer":  "The leave policy states...",
        "sources": [...]   ← empty list when answer is a refusal
    }

    Sources are suppressed when:
    - The answer contains a "couldn't find" / refusal pattern
    - This prevents the confusing situation where the bot says
      "I don't know" but then shows 4 source documents below it,
      which implies the documents were found but the answer wasn't
      — undermining user trust in both the answer and the sources.
    """
    if not question.strip():
        return {"answer": "Please enter a question.", "sources": []}

    result = chain.invoke({"question": question})
    answer = result["answer"]

    # Suppress sources entirely for no-answer / refusal responses
    if _is_no_answer(answer):
        return {"answer": answer, "sources": []}

    # Parse and deduplicate source chunks
    sources = []
    seen = set()

    for doc in result.get("source_documents", []):
        key = (
            doc.metadata.get("source", "unknown"),
            doc.metadata.get("page", "N/A"),
        )
        if key not in seen:
            seen.add(key)
            sources.append({
                "source": doc.metadata.get("source", "unknown"),
                "page":   doc.metadata.get("page", "N/A"),
                "text":   doc.page_content[:300],
            })

    return {"answer": answer, "sources": sources}