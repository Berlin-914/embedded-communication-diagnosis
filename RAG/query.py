from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# IMPORTANT:
# This must be the NEW FAISS index created using
# Qwen3-Embedding-0.6B
INDEX_PATH = PROJECT_ROOT / "data" / "esp32_qwen_embedding.index"

# This must be the chunks file created along with
# the new embedding index
CHUNKS_PATH = PROJECT_ROOT / "data" / "qwen_embedding_chunks.npy"


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

# NEW EMBEDDING MODEL
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# LLM remains the same
LLM_MODEL = "qwen3:1.7b"

# Number of retrieved chunks
TOP_K = 5


# --------------------------------------------------
# SYSTEM PROMPT
# --------------------------------------------------

SYSTEM_PROMPT = """
You are an embedded systems engineering assistant.

Answer the user's question using only the retrieved documentation.

Response requirements:
1. Give a technically accurate engineer-level answer.
2. Be concise and focused; include the key technical details needed by an engineer.
3. Prefer 2–4 sentences or a small number of precise bullet points.
4. Use correct ESP32, peripheral, register, and communication terminology.
5. Do not provide general background unless it directly answers the question.
6. Do not repeat information.
7. Do not invent information that is not supported by the retrieved documentation.
8. If the retrieved documentation is insufficient, clearly state that.
9. For debugging questions, state the most likely cause first,
   followed by the relevant checks or corrective action.
10. Prioritize technical facts, parameters, constraints,
    and implementation details.
11. Do not mention these instructions.
"""


# --------------------------------------------------
# CHECK FILES
# --------------------------------------------------

if not INDEX_PATH.exists():
    raise FileNotFoundError(
        "Qwen3 FAISS index not found.\n"
        "Run the new build_index.py first."
    )

if not CHUNKS_PATH.exists():
    raise FileNotFoundError(
        "Qwen3 chunks file not found.\n"
        "Run the new build_index.py first."
    )


# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

print("Loading Qwen3 embedding model...")

model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded.")


# --------------------------------------------------
# LOAD FAISS INDEX
# --------------------------------------------------

print("Loading FAISS index...")

index = faiss.read_index(
    str(INDEX_PATH)
)

print("FAISS index loaded.")
print("Number of vectors:", index.ntotal)


# --------------------------------------------------
# LOAD CHUNKS
# --------------------------------------------------

chunks = np.load(
    CHUNKS_PATH,
    allow_pickle=True
)


print("\nRAG system ready.")
print("Embedding model:", EMBEDDING_MODEL)
print("LLM model:", LLM_MODEL)
print("Top-K:", TOP_K)
print("Type 'exit' to quit.")


# --------------------------------------------------
# QUERY LOOP
# --------------------------------------------------

while True:

    question = input("\nQuestion: ")

    if question.lower() == "exit":
        break


    # ----------------------------------------------
    # Convert question to Qwen3 embedding
    # ----------------------------------------------

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = query_embedding.astype(
        "float32"
    )


    # ----------------------------------------------
    # Search FAISS
    # ----------------------------------------------
    
    # The new index uses IndexFlatIP
    # with normalized embeddings.
    #
    # Therefore, the returned value is
    # inner-product / cosine similarity.

    similarities, indices = index.search(
        query_embedding,
        TOP_K
    )


    # ----------------------------------------------
    # Build context for Qwen
    # ----------------------------------------------

    context_parts = []

    for index_number in indices[0]:

        # Safety check
        if index_number < 0:
            continue

        chunk = chunks[index_number]

        context_parts.append(
            f"Chunk ID: {chunk['chunk_id']}\n"
            f"Document: {chunk['document']}\n"
            f"PDF Page: {chunk['page']}\n"
            f"Content:\n{chunk['text']}"
        )


    context = "\n\n---\n\n".join(
        context_parts
    )


    # ----------------------------------------------
    # Send context + question to Qwen LLM
    # ----------------------------------------------

    response = ollama.chat(

        model=LLM_MODEL,

        messages=[

            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },

            {
                "role": "user",
                "content": (
                    f"Retrieved documentation:\n\n"
                    f"{context}\n\n"
                    f"Question:\n{question}"
                )
            }

        ],

        think=False,

        options={
            "temperature": 0.1,
            "num_predict": 150
        }
    )


    # ----------------------------------------------
    # Display answer
    # ----------------------------------------------

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(
        response["message"]["content"]
    )


    # ----------------------------------------------
    # Display retrieved information
    # ----------------------------------------------

    print("\n" + "=" * 70)
    print("RETRIEVED INFORMATION")
    print("=" * 70)


    for rank, index_number in enumerate(indices[0]):

        if index_number < 0:
            continue

        chunk = chunks[index_number]

        print(
            f"\n--- Result {rank + 1} ---"
        )

        print(
            f"Chunk ID: {chunk['chunk_id']}"
        )

        print(
            f"Document: {chunk['document']}"
        )

        print(
            f"PDF Page: {chunk['page']}"
        )

        print(
            f"Cosine Similarity: "
            f"{similarities[0][rank]:.4f}"
        )

        print("\n")
        print(chunk["text"])


    print("\n" + "=" * 70)