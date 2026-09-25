"""
rag.py

Multimodal RAG Question Answering System
-----------------------------------------

Supports:
- PDF text
- PDF tables
- PDF OCR
- PDF VLM
- PPT text
- PPT OCR
- PPT image VLM
- PPT slide VLM
- Excel text/tables

Pipeline:

Question
   ↓
Question analysis
   ↓
FAISS semantic retrieval
   ↓
Keyword relevance
   ↓
Visual relevance
   ↓
Slide grouping
   ↓
Context
   ↓
Qwen3
   ↓
Answer
"""

from pathlib import Path
import pickle
import re

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAG_DIR = PROJECT_ROOT / "RAG"

INDEX_PATH = RAG_DIR / "faiss_index.bin"
CHUNKS_PATH = RAG_DIR / "chunks.pkl"


# ============================================================
# MODELS
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "qwen3:1.7b"

OLLAMA_URL = "http://localhost:11434/api/generate"


# ============================================================
# SETTINGS
# ============================================================

INITIAL_TOP_K = 20

FINAL_CONTEXT_K = 5

MAX_CONTEXT_CHARS = 10000


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("=" * 70)
print("MULTIMODAL RAG QUESTION ANSWERING SYSTEM")
print("=" * 70)

print("\nLoading embedding model...")

embedder = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded.")


# ============================================================
# LOAD FAISS INDEX
# ============================================================

if not INDEX_PATH.exists():

    raise FileNotFoundError(
        f"""
FAISS index not found:

{INDEX_PATH}

Run:

python RAG\\build_index.py
"""
    )


if not CHUNKS_PATH.exists():

    raise FileNotFoundError(
        f"""
Chunks file not found:

{CHUNKS_PATH}

Run:

python RAG\\build_index.py
"""
    )


print("\nLoading FAISS index...")

index = faiss.read_index(
    str(INDEX_PATH)
)

print(
    f"FAISS vectors: {index.ntotal}"
)


# ============================================================
# LOAD CHUNKS
# ============================================================

print("\nLoading chunks...")

with open(
    CHUNKS_PATH,
    "rb"
) as f:

    chunks = pickle.load(f)


print(
    f"Chunks loaded: {len(chunks)}"
)

print("=" * 70)


# ============================================================
# CHUNK HELPERS
# ============================================================

def get_chunk_type(chunk):

    if isinstance(chunk, dict):

        return str(
            chunk.get(
                "content_type",
                ""
            )
        ).lower()

    return ""


def get_chunk_source(chunk):

    if isinstance(chunk, dict):

        return str(
            chunk.get(
                "source",
                ""
            )
        )

    return ""


def get_chunk_text(chunk):

    if isinstance(chunk, dict):

        return str(
            chunk.get(
                "text",
                ""
            )
        )

    return str(chunk)


def get_chunk_location(chunk):

    if isinstance(chunk, dict):

        return str(
            chunk.get(
                "location",
                ""
            )
        )

    return ""


# ============================================================
# SLIDE NUMBER
# ============================================================

def get_slide_number(chunk):

    location = get_chunk_location(
        chunk
    )

    match = re.search(
        r"(?:slide|sl)\s*[:#-]?\s*(\d+)",
        location,
        re.IGNORECASE
    )

    if match:

        return int(
            match.group(1)
        )

    return None


# ============================================================
# PAGE NUMBER
# ============================================================

def get_page_number(chunk):

    location = get_chunk_location(
        chunk
    )

    match = re.search(
        r"(?:page|pg)\s*[:#-]?\s*(\d+)",
        location,
        re.IGNORECASE
    )

    if match:

        return int(
            match.group(1)
        )

    return None


# ============================================================
# VISUAL CHUNK
# ============================================================

def is_visual_chunk(chunk):

    chunk_type = get_chunk_type(
        chunk
    )

    return chunk_type in {

        "ppt_image_vlm",
        "ppt_slide_vlm",
        "ppt_image_ocr",

        "pdf_image_vlm",
        "pdf_page_vlm",
        "pdf_image_ocr",
    }


# ============================================================
# EXPLICIT SLIDE NUMBER
# ============================================================

def extract_slide_number_from_question(
    question
):

    patterns = [

        r"\bslide\s*(?:number\s*)?(\d+)\b",

        r"\bsl\s*(\d+)\b",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            question,
            re.IGNORECASE
        )

        if match:

            return int(
                match.group(1)
            )

    return None


# ============================================================
# VISUAL QUESTION
# ============================================================

def question_is_visual(
    question
):

    keywords = [

        "image",
        "diagram",
        "figure",
        "picture",
        "visual",
        "chart",
        "graph",
        "illustration",

        "shown",
        "shows",

        "displayed",

        "looks like",

        "drawn",

        "schematic",

        "block diagram",

        "flowchart",

        "what does",

        "what is shown",
    ]

    question_lower = question.lower()

    return any(
        keyword in question_lower
        for keyword in keywords
    )


# ============================================================
# NORMALIZE WORDS
# ============================================================

def normalize_words(text):

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s-]",
        " ",
        text
    )

    words = text.split()

    return set(words)


# ============================================================
# QUESTION KEYWORDS
# ============================================================

def extract_question_keywords(
    question
):

    stop_words = {

        "what",
        "is",
        "the",
        "a",
        "an",
        "does",
        "do",
        "show",
        "shows",
        "in",
        "on",
        "of",
        "to",
        "and",
        "or",
        "for",
        "this",
        "that",
        "what's",
        "can",
        "you",
        "explain",
        "tell",
        "me",
        "about",
    }

    words = normalize_words(
        question
    )

    keywords = {

        word
        for word in words
        if word not in stop_words
        and len(word) >= 3
    }

    return keywords


# ============================================================
# DOMAIN PHRASES
# ============================================================

def detect_domain_phrases(
    question
):

    q = question.lower()

    phrases = []

    # --------------------------------------------------------
    # ACQUISITION
    # --------------------------------------------------------

    if "acquisition" in q:

        phrases.extend([
            "acquisition",
            "acquisition process",
            "establishing the link",
            "receiver responds",
            "transmitter",
            "receiver",
            "link",
            "fpa",
            "stare mode",
            "beacon",
        ])

    # --------------------------------------------------------
    # TRACKING
    # --------------------------------------------------------

    if "tracking" in q:

        phrases.extend([
            "tracking",
            "track",
            "target",
            "beam",
            "fpa",
        ])

    # --------------------------------------------------------
    # POINTING
    # --------------------------------------------------------

    if "pointing" in q:

        phrases.extend([
            "pointing",
            "gimbal",
            "target",
            "beam",
        ])

    # --------------------------------------------------------
    # UNCERTAINTY
    # --------------------------------------------------------

    if "uncertainty" in q:

        phrases.extend([
            "uncertainty",
            "uncertainty area",
            "uncertainty budget",
            "attitude",
            "ephemeris",
            "gimbal accuracy",
        ])

    return phrases


# ============================================================
# CONTENT TYPE BONUS
# ============================================================

def content_type_bonus(
    chunk_type
):

    bonuses = {

        "ppt_image_vlm": 5.0,

        "ppt_slide_vlm": 4.5,

        "ppt_image_ocr": 2.5,

        "ppt_text": 1.5,

        "pdf_image_vlm": 5.0,

        "pdf_page_vlm": 4.5,

        "pdf_image_ocr": 2.5,

        "pdf_text": 1.5,

        "pdf_table": 1.5,

        "excel_table": 1.5,

        "excel_text": 1.5,
    }

    return bonuses.get(
        chunk_type,
        0.0
    )


# ============================================================
# KEYWORD SCORE
# ============================================================

def calculate_keyword_score(
    question,
    chunk
):

    question_words = extract_question_keywords(
        question
    )

    chunk_text = get_chunk_text(
        chunk
    ).lower()

    chunk_words = normalize_words(
        chunk_text
    )

    if not question_words:

        return 0.0

    matches = (
        question_words
        & chunk_words
    )

    # Each matching question keyword
    # gives a useful boost.

    score = len(matches) * 1.5

    # --------------------------------------------------------
    # DOMAIN PHRASE MATCH
    # --------------------------------------------------------

    domain_phrases = detect_domain_phrases(
        question
    )

    for phrase in domain_phrases:

        if phrase.lower() in chunk_text:

            score += 3.0

    return score


# ============================================================
# ACQUISITION-SPECIFIC RELEVANCE
# ============================================================

def acquisition_relevance_bonus(
    question,
    chunk
):

    q = question.lower()

    text = get_chunk_text(
        chunk
    ).lower()

    score = 0.0

    # Only activate this logic when
    # user asks about acquisition.

    if "acquisition" not in q:

        return score

    # --------------------------------------------------------
    # STRONG ACQUISITION PROCESS TERMS
    # --------------------------------------------------------

    process_terms = [

        "acquisition process",

        "establishing the link",

        "phase 1",

        "phase 2",

        "phase 3",

        "receiver responds",

        "transmitter begins",

        "stare mode",

        "beacon",
    ]

    for term in process_terms:

        if term in text:

            score += 4.0

    # --------------------------------------------------------
    # ACTUAL ACQUISITION DIAGRAM
    # --------------------------------------------------------

    if (
        "phase 1" in text
        or "phase 2" in text
        or "phase 3" in text
    ):

        score += 6.0

    # --------------------------------------------------------
    # UNCERTAINTY BUDGET IS DIFFERENT
    # --------------------------------------------------------

    unrelated_terms = [

        "initial uncertainty area budget",

        "uncertainty area budget",

        "attitude and ephemeris",

        "gimbal jitter",

        "gimbal accuracy",

        "reference calibration",
    ]

    for term in unrelated_terms:

        if term in text:

            score -= 4.0

    return score


# ============================================================
# VISUAL RELEVANCE BONUS
# ============================================================

def visual_relevance_bonus(
    question,
    chunk
):

    if not question_is_visual(
        question
    ):

        return 0.0

    chunk_type = get_chunk_type(
        chunk
    )

    score = 0.0

    if chunk_type == "ppt_image_vlm":

        score += 7.0

    elif chunk_type == "ppt_slide_vlm":

        score += 6.0

    elif chunk_type == "ppt_image_ocr":

        score += 3.0

    elif chunk_type == "pdf_image_vlm":

        score += 7.0

    elif chunk_type == "pdf_page_vlm":

        score += 6.0

    elif chunk_type == "pdf_image_ocr":

        score += 3.0

    return score


# ============================================================
# RETRIEVE CHUNKS
# ============================================================

def retrieve_chunks(
    question
):

    print(
        "\n" + "=" * 70
    )

    print(
        "RETRIEVAL"
    )

    print(
        "=" * 70
    )

    print(
        f"\nQuestion:\n{question}"
    )

    # --------------------------------------------------------
    # QUESTION ANALYSIS
    # --------------------------------------------------------

    explicit_slide = (
        extract_slide_number_from_question(
            question
        )
    )

    visual_question = (
        question_is_visual(
            question
        )
    )

    question_keywords = (
        extract_question_keywords(
            question
        )
    )

    print(
        f"\nVisual question: "
        f"{visual_question}"
    )

    print(
        f"Question keywords: "
        f"{question_keywords}"
    )

    if explicit_slide is not None:

        print(
            f"Explicit slide: "
            f"{explicit_slide}"
        )

    # --------------------------------------------------------
    # EMBEDDING
    # --------------------------------------------------------

    query_embedding = embedder.encode(
        [question],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    # --------------------------------------------------------
    # FAISS SEARCH
    # --------------------------------------------------------

    search_k = min(
        INITIAL_TOP_K,
        index.ntotal
    )

    distances, indices = index.search(
        query_embedding,
        search_k
    )

    # --------------------------------------------------------
    # COLLECT FAISS RESULTS
    # --------------------------------------------------------

    semantic_results = []

    for rank, (
        distance,
        idx
    ) in enumerate(
        zip(
            distances[0],
            indices[0]
        )
    ):

        if (
            idx < 0
            or idx >= len(chunks)
        ):
            continue

        chunk = chunks[
            int(idx)
        ]

        semantic_results.append({

            "idx": int(idx),

            "distance": float(
                distance
            ),

            "rank": rank,

            "chunk": chunk,

            "source": get_chunk_source(
                chunk
            ),

            "slide": get_slide_number(
                chunk
            ),

            "page": get_page_number(
                chunk
            ),

            "location": get_chunk_location(
                chunk
            ),

            "type": get_chunk_type(
                chunk
            ),
        })

    # --------------------------------------------------------
    # GROUP BY SLIDE/PAGE
    # --------------------------------------------------------

    groups = {}

    for item in semantic_results:

        source = item["source"]

        slide = item["slide"]

        page = item["page"]

        if slide is not None:

            key = (
                source,
                "slide",
                slide
            )

        elif page is not None:

            key = (
                source,
                "page",
                page
            )

        else:

            key = (
                source,
                "location",
                item["location"]
            )

        if key not in groups:

            groups[key] = {

                "source": source,

                "slide": slide,

                "page": page,

                "location": item[
                    "location"
                ],

                "items": [],

                "best_semantic": -999999.0,
            }

        groups[key][
            "items"
        ].append(item)

        groups[key][
            "best_semantic"
        ] = max(

            groups[key][
                "best_semantic"
            ],

            item["distance"]
        )

    # --------------------------------------------------------
    # SCORE EACH GROUP
    # --------------------------------------------------------

    scored_groups = []

    for group in groups.values():

        score = (
            group["best_semantic"]
        )

        # ----------------------------------------------------
        # SCORE ALL CONTENT IN GROUP
        # ----------------------------------------------------

        best_keyword_score = 0.0

        best_visual_score = 0.0

        best_acquisition_score = 0.0

        for item in group["items"]:

            chunk = item["chunk"]

            keyword_score = (
                calculate_keyword_score(
                    question,
                    chunk
                )
            )

            visual_score = (
                visual_relevance_bonus(
                    question,
                    chunk
                )
            )

            acquisition_score = (
                acquisition_relevance_bonus(
                    question,
                    chunk
                )
            )

            type_bonus = (
                content_type_bonus(
                    item["type"]
                )
            )

            item_score = (
                keyword_score
                + visual_score
                + acquisition_score
                + type_bonus
            )

            best_keyword_score = max(
                best_keyword_score,
                keyword_score
            )

            best_visual_score = max(
                best_visual_score,
                visual_score
            )

            best_acquisition_score = max(
                best_acquisition_score,
                acquisition_score
            )

            # ------------------------------------------------
            # Keep strongest item contribution
            # ------------------------------------------------

            score += (
                item_score * 0.35
            )

        # ----------------------------------------------------
        # EXPLICIT SLIDE
        # ----------------------------------------------------

        if (
            explicit_slide is not None
            and group["slide"]
            == explicit_slide
        ):

            score += 25.0

        scored_groups.append({

            "group": group,

            "score": score,

            "keyword_score":
                best_keyword_score,

            "visual_score":
                best_visual_score,

            "acquisition_score":
                best_acquisition_score,
        })

    # --------------------------------------------------------
    # SORT GROUPS
    # --------------------------------------------------------

    scored_groups.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # --------------------------------------------------------
    # PRINT GROUP RANKING
    # --------------------------------------------------------

    print(
        "\nCandidate slide ranking:"
    )

    for i, item in enumerate(
        scored_groups[:10],
        start=1
    ):

        group = item["group"]

        print(

            f"{i}. "

            f"{group['location']} | "

            f"score="
            f"{item['score']:.4f} | "

            f"keyword="
            f"{item['keyword_score']:.2f} | "

            f"visual="
            f"{item['visual_score']:.2f} | "

            f"acquisition="
            f"{item['acquisition_score']:.2f}"
        )

    # --------------------------------------------------------
    # SELECT TOP GROUPS
    # --------------------------------------------------------

    selected_groups = (
        scored_groups[:3]
    )

    # --------------------------------------------------------
    # COLLECT ALL CHUNKS FROM
    # SELECTED SLIDES
    # --------------------------------------------------------

    candidate_chunks = []

    selected_keys = set()

    for selected in selected_groups:

        group = selected["group"]

        source = group["source"]

        slide = group["slide"]

        page = group["page"]

        selected_keys.add(
            (
                source,
                slide,
                page
            )
        )

        for item in group["items"]:

            candidate_chunks.append({

                "chunk":
                    item["chunk"],

                "base_score":
                    selected["score"],
            })

    # --------------------------------------------------------
    # COLLECT ALL CHUNKS FROM SAME SLIDE
    # --------------------------------------------------------

    for chunk in chunks:

        source = get_chunk_source(
            chunk
        )

        slide = get_slide_number(
            chunk
        )

        page = get_page_number(
            chunk
        )

        key = (
            source,
            slide,
            page
        )

        if key not in selected_keys:

            continue

        already_exists = any(

            item["chunk"] is chunk

            for item in candidate_chunks
        )

        if already_exists:

            continue

        candidate_chunks.append({

            "chunk": chunk,

            "base_score":
                0.0,
        })

    # --------------------------------------------------------
    # FINAL CHUNK SCORING
    # --------------------------------------------------------

    final_candidates = []

    for item in candidate_chunks:

        chunk = item["chunk"]

        score = (
            item["base_score"]
        )

        # ----------------------------------------------------
        # CONTENT TYPE
        # ----------------------------------------------------

        score += (
            content_type_bonus(
                get_chunk_type(
                    chunk
                )
            )
        )

        # ----------------------------------------------------
        # KEYWORD
        # ----------------------------------------------------

        score += (
            calculate_keyword_score(
                question,
                chunk
            )
        )

        # ----------------------------------------------------
        # VISUAL
        # ----------------------------------------------------

        score += (
            visual_relevance_bonus(
                question,
                chunk
            )
        )

        # ----------------------------------------------------
        # ACQUISITION
        # ----------------------------------------------------

        score += (
            acquisition_relevance_bonus(
                question,
                chunk
            )
        )

        # ----------------------------------------------------
        # EXPLICIT SLIDE
        # ----------------------------------------------------

        if (
            explicit_slide is not None
            and get_slide_number(
                chunk
            ) == explicit_slide
        ):

            score += 25.0

        final_candidates.append({

            "chunk":
                chunk,

            "score":
                score,
        })

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    final_candidates.sort(

        key=lambda x:
            x["score"],

        reverse=True
    )

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    results = []

    seen = set()

    for item in final_candidates:

        chunk = item["chunk"]

        text = get_chunk_text(
            chunk
        ).strip()

        source = get_chunk_source(
            chunk
        )

        location = get_chunk_location(
            chunk
        )

        chunk_type = get_chunk_type(
            chunk
        )

        key = (

            source,

            location,

            chunk_type,

            text[:300],
        )

        if key in seen:

            continue

        seen.add(key)

        results.append({

            "chunk":
                chunk,

            "score":
                item["score"],
        })

        if (
            len(results)
            >= FINAL_CONTEXT_K
        ):

            break

    # --------------------------------------------------------
    # PRINT FINAL RETRIEVAL
    # --------------------------------------------------------

    print(
        "\nFinal retrieved chunks:"
    )

    for i, item in enumerate(
        results,
        start=1
    ):

        chunk = item["chunk"]

        text = get_chunk_text(
            chunk
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        print(

            f"\n{i}. "

            f"{get_chunk_type(chunk)} | "

            f"{get_chunk_location(chunk)} | "

            f"score="
            f"{item['score']:.4f}"
        )

        print(
            text[:300]
        )

    return results


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(
    results
):

    context_parts = []

    total_chars = 0

    for number, item in enumerate(
        results,
        start=1
    ):

        chunk = item["chunk"]

        source = get_chunk_source(
            chunk
        )

        location = get_chunk_location(
            chunk
        )

        chunk_type = get_chunk_type(
            chunk
        )

        text = get_chunk_text(
            chunk
        ).strip()

        if not text:

            continue

        source_name = (
            Path(source).name
            if source
            else "Unknown"
        )

        block = f"""
--- RETRIEVED ITEM {number} ---
Source: {source_name}
Location: {location}
Content type: {chunk_type}
Content:
{text}
"""

        if (
            total_chars
            + len(block)
            > MAX_CONTEXT_CHARS
        ):

            break

        context_parts.append(
            block
        )

        total_chars += len(
            block
        )

    return "\n".join(
        context_parts
    )


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    context
):

    prompt = f"""
/no_think

You are a multimodal RAG assistant for technical documents.

Answer the user's question using ONLY the retrieved context.

IMPORTANT RULES:

1. Use only the retrieved document information.

2. Give the direct answer first.

3. For image or diagram questions, prioritize
   ppt_image_vlm and ppt_slide_vlm descriptions.

4. If multiple visual descriptions refer to the same
   slide, combine them.

5. Do NOT use information from an unrelated slide.

6. Do NOT invent objects, components, formulas,
   labels, or relationships.

7. If the retrieved context is insufficient, say:

"The retrieved document content is not sufficient
to answer this."

8. Do not explain your reasoning.

9. Do not show thinking.

10. Use simple English.

11. Keep the answer concise.

12. If the user asks "what does the diagram show",
    describe the actual diagram and its main stages,
    components, or relationships.

USER QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

ANSWER:
"""

    payload = {

        "model":
            LLM_MODEL,

        "prompt":
            prompt,

        "stream":
            False,

        # Disable Qwen3 thinking
        "think":
            False,

        "keep_alive":
            "10m",

        "options": {

            "temperature":
                0.1,

            "num_predict":
                220,
        },
    }

    try:

        response = requests.post(

            OLLAMA_URL,

            json=payload,

            timeout=120,
        )

        response.raise_for_status()

        result = response.json()

        answer = result.get(
            "response",
            ""
        ).strip()

        if not answer:

            print(
                "\nWARNING: Qwen3 returned "
                "an empty response."
            )

            print(
                "\nRaw Ollama response:"
            )

            print(result)

            return (
                "I could not generate an answer "
                "from the retrieved context."
            )

        return answer

    except requests.exceptions.Timeout:

        return (
            "Qwen3 took too long to respond."
        )

    except requests.exceptions.ConnectionError:

        return (
            "Cannot connect to Ollama. "
            "Make sure Ollama is running."
        )

    except requests.exceptions.RequestException as e:

        return (
            f"Ollama request failed: {e}"
        )

    except Exception as e:

        return (
            f"Unexpected error: {e}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MULTIMODAL RAG ASSISTANT"
    )

    print(
        "=" * 70
    )

    print(
        "\nType your question."
    )

    print(
        "Type 'exit' to stop."
    )

    print(
        "=" * 70
    )

    while True:

        try:

            question = input(
                "\nQuestion: "
            ).strip()

        except KeyboardInterrupt:

            print(
                "\n\nExiting..."
            )

            break

        except EOFError:

            print(
                "\n\nExiting..."
            )

            break

        if not question:

            print(
                "Please enter a question."
            )

            continue

        if question.lower() in {
            "exit",
            "quit",
            "q",
        }:

            print(
                "\nExiting RAG assistant..."
            )

            break

        # ----------------------------------------------------
        # RETRIEVAL
        # ----------------------------------------------------

        results = retrieve_chunks(
            question
        )

        if not results:

            print(
                "\nNo relevant content found."
            )

            continue

        # ----------------------------------------------------
        # CONTEXT
        # ----------------------------------------------------

        context = build_context(
            results
        )

        print(
            "\nContext prepared."
        )

        print(
            f"Context characters: "
            f"{len(context)}"
        )

        # ----------------------------------------------------
        # GENERATION
        # ----------------------------------------------------

        print(
            "\nGenerating answer..."
        )

        answer = generate_answer(
            question,
            context
        )

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        print(
            "\n"
            + "=" * 70
        )

        print(
            "ANSWER"
        )

        print(
            "=" * 70
        )

        print(
            answer
        )

        print(
            "=" * 70
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()