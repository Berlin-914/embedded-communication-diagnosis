from pathlib import Path
import pymupdf
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_PATH = PROJECT_ROOT / "data" / "esp32_technical_reference_manual_en.pdf"
INDEX_PATH = PROJECT_ROOT / "data" / "esp32.index"
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks.npy"


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# --------------------------------------------------
# CHECK PDF
# --------------------------------------------------

if not PDF_PATH.exists():
    raise FileNotFoundError(
        f"PDF not found:\n{PDF_PATH}\n\n"
        "Make sure the PDF is inside the data folder."
    )


# --------------------------------------------------
# LOAD PDF
# --------------------------------------------------

print("Loading PDF...")
print(f"File: {PDF_PATH}")

doc = pymupdf.open(PDF_PATH)

print(f"Number of pages: {len(doc)}")


# --------------------------------------------------
# EXTRACT TEXT
# --------------------------------------------------

print("\nExtracting text...")

pages = []

for page_number, page in enumerate(doc):
    text = page.get_text()

    if text.strip():
        pages.append({
            "page": page_number + 1,
            "text": text
        })

print(f"Pages containing text: {len(pages)}")


# --------------------------------------------------
# CHUNKING
# --------------------------------------------------

print("\nCreating chunks...")

chunks = []

chunk_id = 0

for page in pages:

    text = page["text"]

    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk_text = text[start:end].strip()

        if chunk_text:

            chunks.append({
                "chunk_id": chunk_id,
                "document": "ESP32 Technical Reference Manual",
                "page": page["page"],
                "text": chunk_text
            })

            chunk_id += 1

        start += CHUNK_SIZE - CHUNK_OVERLAP


print(f"Total chunks created: {len(chunks)}")


# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

print("\nLoading embedding model...")

model = SentenceTransformer(EMBEDDING_MODEL)

print("Embedding model loaded.")


# --------------------------------------------------
# CREATE EMBEDDINGS
# --------------------------------------------------

print("\nCreating embeddings...")

texts = [chunk["text"] for chunk in chunks]

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True
)

embeddings = embeddings.astype("float32")

print("Embedding shape:", embeddings.shape)


# --------------------------------------------------
# CREATE FAISS INDEX
# --------------------------------------------------

print("\nCreating FAISS index...")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("Vectors stored in FAISS:", index.ntotal)


# --------------------------------------------------
# SAVE FAISS INDEX
# --------------------------------------------------

faiss.write_index(index, str(INDEX_PATH))

print("\nFAISS index saved to:")
print(INDEX_PATH)


# --------------------------------------------------
# SAVE CHUNKS
# --------------------------------------------------

np.save(
    CHUNKS_PATH,
    np.array(chunks, dtype=object)
)

print("\nChunks saved to:")
print(CHUNKS_PATH)


# --------------------------------------------------
# COMPLETE
# --------------------------------------------------

print("\n===================================")
print("RAG INDEX BUILD COMPLETE")
print("===================================")