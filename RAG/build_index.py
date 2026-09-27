from pathlib import Path
import pymupdf
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "esp32_technical_reference_manual_en.pdf"
)


INDEX_PATH = (
    PROJECT_ROOT
    / "data"
    / "esp32_qwen_embedding.index"
)

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "qwen_embedding_chunks.npy"
)


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# NEW EMBEDDING MODEL
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


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

print(
    f"Number of pages: {len(doc)}"
)


# --------------------------------------------------
# EXTRACT TEXT
# --------------------------------------------------

print("\nExtracting text...")

pages = []


for page_number, page in enumerate(doc):

    text = page.get_text()

    if text.strip():

        pages.append(
            {
                "page":
                    page_number + 1,

                "text":
                    text
            }
        )


print(
    f"Pages containing text: "
    f"{len(pages)}"
)


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

        chunk_text = (
            text[start:end]
            .strip()
        )


        if chunk_text:

            chunks.append(
                {
                    "chunk_id":
                        chunk_id,

                    "document":
                        "ESP32 Technical Reference Manual",

                    "page":
                        page["page"],

                    "type":
                        "text",

                    "text":
                        chunk_text
                }
            )

            chunk_id += 1


        start += (
            CHUNK_SIZE
            - CHUNK_OVERLAP
        )


print(
    f"Total text chunks created: "
    f"{len(chunks)}"
)


# --------------------------------------------------
# TABLE EXTRACTION
# --------------------------------------------------

print("\nExtracting tables...")

table_count = 0


for page_number, page in enumerate(doc):

    try:

        tables = page.find_tables()

        for table_index, table in enumerate(
            tables.tables
        ):

            table_data = table.extract()

            if not table_data:
                continue


            table_lines = []

            table_lines.append(
                f"Table {table_index + 1} "
                f"from page {page_number + 1}"
            )


            for row in table_data:

                if row is None:
                    continue

                cleaned_row = []

                for cell in row:

                    if cell is None:
                        cell = ""

                    cleaned_row.append(
                        str(cell).strip()
                    )


                row_text = " | ".join(
                    cleaned_row
                )

                if row_text.strip():

                    table_lines.append(
                        row_text
                    )


            table_text = "\n".join(
                table_lines
            )


            if table_text.strip():

                chunks.append(
                    {
                        "chunk_id":
                            chunk_id,

                        "document":
                            "ESP32 Technical Reference Manual",

                        "page":
                            page_number + 1,

                        "type":
                            "table",

                        "text":
                            table_text
                    }
                )

                chunk_id += 1

                table_count += 1


    except Exception as e:

        print(
            f"Table extraction warning "
            f"on page {page_number + 1}: {e}"
        )


print(
    f"Tables extracted: {table_count}"
)


# --------------------------------------------------
# IMPORTANT
# --------------------------------------------------
#
# IMAGE DESCRIPTIONS
#
# If you have already generated image
# descriptions using Qwen2.5-VL, add them
# to the chunks in the same format:
#
# {
#     "chunk_id": ...,
#     "document": ...,
#     "page": ...,
#     "type": "image",
#     "text": "description generated by VLM"
# }
#
# This means the SAME embedding model
# Qwen3-Embedding-0.6B will embed:
#
# TEXT + TABLE + IMAGE DESCRIPTION
#
# --------------------------------------------------


# --------------------------------------------------
# LOAD NEW EMBEDDING MODEL
# --------------------------------------------------

print("\nLoading Qwen3 embedding model...")

model = SentenceTransformer(
    EMBEDDING_MODEL
)

print(
    "Embedding model loaded."
)


# --------------------------------------------------
# CREATE EMBEDDINGS
# --------------------------------------------------

print("\nCreating Qwen3 embeddings...")

texts = [
    chunk["text"]
    for chunk in chunks
]


embeddings = model.encode(
    texts,

    show_progress_bar=True,

    convert_to_numpy=True,

    normalize_embeddings=True
)


embeddings = embeddings.astype(
    "float32"
)


print(
    "Embedding shape:",
    embeddings.shape
)


# --------------------------------------------------
# CREATE FAISS INDEX
# --------------------------------------------------

print("\nCreating FAISS index...")

dimension = embeddings.shape[1]


# Because embeddings are normalized,
# Inner Product is suitable for cosine similarity.

index = faiss.IndexFlatIP(
    dimension
)


index.add(
    embeddings
)


print(
    "Vectors stored in FAISS:",
    index.ntotal
)


# --------------------------------------------------
# SAVE FAISS INDEX
# --------------------------------------------------

faiss.write_index(
    index,
    str(INDEX_PATH)
)


print(
    "\nFAISS index saved to:"
)

print(
    INDEX_PATH
)


# --------------------------------------------------
# SAVE CHUNKS
# --------------------------------------------------

np.save(
    CHUNKS_PATH,

    np.array(
        chunks,
        dtype=object
    )
)


print(
    "\nChunks saved to:"
)

print(
    CHUNKS_PATH
)


# --------------------------------------------------
# COMPLETE
# --------------------------------------------------

print("\n===================================")

print(
    "QWEN3 MULTIMODAL RAG INDEX COMPLETE"
)

print("===================================")

print(
    f"Embedding model : {EMBEDDING_MODEL}"
)

print(
    f"Total vectors   : {index.ntotal}"
)

print(
    f"Vector dimension: {dimension}"
)

print(
    "Content types   : Text + Tables + Images"
)

print(
    "===================================")