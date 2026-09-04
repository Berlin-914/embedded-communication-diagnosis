from pathlib import Path
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INDEX_PATH = PROJECT_ROOT / "data" / "esp32.index"
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks.npy"


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "qwen3:4b"

TOP_K = 5


# --------------------------------------------------
# CHECK FILES
# --------------------------------------------------

if not INDEX_PATH.exists():
    raise FileNotFoundError(
        "FAISS index not found. Run build_index.py first."
    )

if not CHUNKS_PATH.exists():
    raise FileNotFoundError(
        "Chunks file not found. Run build_index.py first."
    )


# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


# --------------------------------------------------
# LOAD FAISS INDEX
# --------------------------------------------------

print("Loading FAISS index...")

index = faiss.read_index(
    str(INDEX_PATH)
)


# --------------------------------------------------
# LOAD CHUNKS
# --------------------------------------------------

chunks = np.load(
    CHUNKS_PATH,
    allow_pickle=True
)


print("\nRAG system ready.")
print("Type 'exit' to quit.")


# --------------------------------------------------
# RAG QUERY FUNCTION
# --------------------------------------------------

def ask_rag(question):

    # ----------------------------------------------
    # 1. Convert question to embedding
    # ----------------------------------------------

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    query_embedding = query_embedding.astype(
        "float32"
    )


    # ----------------------------------------------
    # 2. Search FAISS
    # ----------------------------------------------

    distances, indices = index.search(
        query_embedding,
        TOP_K
    )


    # ----------------------------------------------
    # 3. Build context from retrieved chunks
    # ----------------------------------------------

    context_parts = []

    sources = []

    for rank, index_number in enumerate(indices[0]):

        chunk = chunks[index_number]

        source_text = (
            f"[Source {rank + 1}]\n"
            f"Document: {chunk['document']}\n"
            f"Page: {chunk['page']}\n"
            f"Chunk ID: {chunk['chunk_id']}\n"
            f"Content:\n{chunk['text']}"
        )

        context_parts.append(source_text)

        sources.append({
            "document": chunk["document"],
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
            "distance": float(distances[0][rank])
        })


    context = "\n\n".join(context_parts)


    # ----------------------------------------------
    # 4. Construct grounded prompt
    # ----------------------------------------------

    prompt = f"""
You are an embedded systems technical assistant.

Your task is to answer the user's question using ONLY
the technical documentation provided below.

IMPORTANT RULES:

1. Use the provided documentation as the primary source.
2. Do not invent technical specifications.
3. Do not introduce information that is not supported
   by the provided documentation.
4. If the documentation does not contain enough
   information to answer the question, explicitly say:
   "The provided documentation does not contain enough
   information to answer this."
5. Give a concise technical explanation.
6. When making a claim, mention the relevant page number
   when possible.

TECHNICAL DOCUMENTATION:

{context}


USER QUESTION:

{question}


ANSWER:
"""


    # ----------------------------------------------
    # 5. Send context + question to Ollama
    # ----------------------------------------------

    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False
    }


    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=300
    )


    # ----------------------------------------------
    # 6. Check Ollama response
    # ----------------------------------------------

    if response.status_code != 200:

        raise RuntimeError(
            f"Ollama error: {response.text}"
        )


    result = response.json()

    answer = result["response"]


    return answer, sources


# --------------------------------------------------
# INTERACTIVE QUERY LOOP
# --------------------------------------------------

while True:

    question = input("\nQuestion: ")

    if question.lower().strip() == "exit":
        break


    print("\nGenerating answer...")

    answer, sources = ask_rag(question)


    # ----------------------------------------------
    # DISPLAY ANSWER
    # ----------------------------------------------

    print("\n" + "=" * 70)
    print("RAG ANSWER")
    print("=" * 70)

    print(answer)


    # ----------------------------------------------
    # DISPLAY SOURCES
    # ----------------------------------------------

    print("\n" + "=" * 70)
    print("SOURCES USED")
    print("=" * 70)

    for source in sources:

        print(
            f"Page {source['page']} | "
            f"Chunk {source['chunk_id']} | "
            f"Distance {source['distance']:.4f}"
        )

    print("=" * 70)