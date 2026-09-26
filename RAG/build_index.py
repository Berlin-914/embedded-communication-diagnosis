from pathlib import Path
import base64
import io
import pickle
import re
import shutil
import subprocess
import tempfile

import fitz
import faiss
import numpy as np
import openpyxl
import pytesseract
import requests

from PIL import Image
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from sentence_transformers import SentenceTransformer


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "DATA"
RAG_DIR = PROJECT_ROOT / "RAG"

FAISS_INDEX_PATH = RAG_DIR / "faiss_index.bin"
CHUNKS_PATH = RAG_DIR / "chunks.pkl"


# ============================================================
# MODELS
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

VLM_MODEL = "qwen2.5vl:3b"

OLLAMA_URL = "http://localhost:11434/api/generate"


# ============================================================
# SETTINGS
# ============================================================

EMBED_BATCH_SIZE = 32

CHUNK_SIZE = 1400
CHUNK_OVERLAP = 250

RENDER_SCALE = 1.5

VLM_TIMEOUT = 600


# ============================================================
# TESSERACT
# ============================================================

TESSERACT_PATH = Path(
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

if TESSERACT_PATH.exists():
    pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_PATH)


# ============================================================
# BASIC TEXT CLEANING
# ============================================================

def clean_text(text):
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# TABLE DETECTION
# ============================================================

def looks_like_table(text):
    if not text:
        return False
    lines = text.splitlines()
    if len(lines) < 2:
        return False
    tab_lines = sum("\t" in line for line in lines)
    if tab_lines >= 2:
        return True
    multi_space_lines = sum(
        len(re.split(r"\s{2,}", line.strip())) >= 3
        for line in lines
    )
    return multi_space_lines >= 2


# ============================================================
# FORMULA / EQUATION DETECTION
# ============================================================

FORMULA_SYMBOL_PATTERN = re.compile(
    r"(=|≈|≤|≥|±|÷|×|√|∑|∫|Σ|Δ|π|θ|α|β|γ|λ|μ|ω|"
    r"\^|_\{|d/dt|dx|dy|sqrt|log|ln\(|sin\(|cos\(|tan\(|exp\(|"
    r"[a-zA-Z]\s*=\s*[^,\.\n]{1,80})"
)

FORMULA_KEYWORD_PATTERN = re.compile(
    r"\b(formula|equation|expression|derived as|given by|"
    r"calculated (?:as|using)|is defined as)\b",
    re.IGNORECASE
)


def looks_like_formula(text):

    if not text:
        return False

    if FORMULA_KEYWORD_PATTERN.search(text):
        return True

    symbol_hits = len(FORMULA_SYMBOL_PATTERN.findall(text))

    if symbol_hits >= 2:
        return True

    for line in text.splitlines():
        line = line.strip()
        if 0 < len(line) <= 120 and re.match(
            r"^[A-Za-z0-9_ΣΔπθαβγλμωΩ]{1,20}\s*=\s*.+$", line
        ):
            return True

    return False


# ============================================================
# ADD CHUNK
# ============================================================

def add_chunk(chunks, text, source, content_type, location=""):
    text = clean_text(text)
    if not text:
        return
    chunks.append({
        "text": text,
        "source": str(source),
        "content_type": content_type,
        "location": location
    })


# ============================================================
# TEXT SPLITTER
# ============================================================

def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = clean_text(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


# ============================================================
# CLASSIFY TEXT CONTENT TYPE (formula > table > plain text)
# ============================================================

def classify_text_content_type(part, formula_type, table_type, plain_type):
    if looks_like_formula(part):
        return formula_type
    if looks_like_table(part):
        return table_type
    return plain_type


# ============================================================
# OCR
# ============================================================

def ocr_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if image.width < 1600:
            scale = 1600 / image.width
            new_size = (
                int(image.width * scale),
                int(image.height * scale)
            )
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        text = pytesseract.image_to_string(image, config="--psm 6")
        return clean_text(text)
    except Exception as e:
        print(f"    OCR warning: {e}")
        return ""


# ============================================================
# VLM IMAGE UNDERSTANDING
# ============================================================

def vlm_explain_image(image_bytes, source, location=""):

    print()
    print(f"    VLM analysing: {source} {location}")

    try:
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        prompt = """
You are an expert embedded systems engineer.

Analyze the provided image carefully.

This image may contain:
- UART communication diagrams
- SPI timing diagrams
- I2C waveforms
- CAN communication diagrams
- circuit diagrams
- microcontroller block diagrams
- pin diagrams
- communication architecture
- flowcharts
- oscilloscope screenshots
- debugging screenshots
- embedded hardware diagrams
- technical tables
- technical figures
- mathematical formulas and equations

IMPORTANT:
Do NOT only perform OCR.
The main task is to understand the VISUAL MEANING.

Describe:
1. What type of diagram, figure, waveform, circuit,
   screenshot, or technical image this is.
2. The important components visible.
3. The connections between components.
4. The direction of signals or data flow.
5. How the system operates according to the image.
6. Important communication signals, pins, buses,
   timing signals, and control lines.
7. Any visible fault, abnormal condition,
   warning, or debugging information.
8. How the image can help diagnose an
   embedded communication problem.
9. Important labels that are actually visible.

IF THE IMAGE CONTAINS ONE OR MORE FORMULAS OR EQUATIONS,
follow this EXACT procedure — this is the most important
part of your task:

- Transcribe each formula on its OWN line, prefixed
  literally with "Formula:" so it can be located later.
- Use ONLY plain text characters: letters, digits,
  Greek letter names or their glyphs exactly as shown
  (e.g. Σ, Δ, π), the operators + - * / ^ =, and
  parentheses. Write a fraction as "(numerator) / (denominator)".
- Do NOT use LaTeX notation. Do NOT write \\frac, \\text,
  $$, \\times, or any backslash commands. Write the raw
  characters only, e.g.:
  "Formula: Azimuth = (A + B - C - D) / Sigma"
- Preserve the EXACT variable names, subscripts,
  superscripts, and the EXACT left-to-right order of terms
  shown in the image. Do not reorder "A + B - C - D" into
  "A - C + B - D" or similar — copy the visual order exactly.
- If there are multiple formulas in the image (e.g. one for
  Azimuth and one for Elevation), transcribe EACH one on its
  own separate "Formula:" line. Never merge two formulas into
  one line and never mix terms from one formula into another.
- Do NOT simplify, round, rearrange, or approximate any formula.
- Do NOT invent a formula if none is visible.
- If part of a formula is unclear or cut off, write
  "Formula: [unclear]" instead of guessing the missing part.

For waveforms:
- identify clock
- identify data
- identify start/stop conditions
- identify ACK/NACK when visible
- explain timing relationships
- identify abnormal waveform conditions

For circuit diagrams:
- identify components
- explain connections
- explain power flow
- explain signal flow

For block diagrams:
- identify important blocks
- explain the relationship between blocks
- explain the data/signal flow

For screenshots:
- describe only what is visible
- identify useful technical information
- do not invent values

IMPORTANT:
Do not hallucinate components,
labels, signals, pin numbers, values,
connections, or formulas.

If something cannot be clearly identified,
say that it is unclear.

The final explanation will be stored in
an embedded-systems RAG knowledge base.

Write a technically accurate,
easy-to-understand explanation.
"""

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": VLM_MODEL,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 700
                }
            },
            timeout=VLM_TIMEOUT
        )

        response.raise_for_status()
        result = response.json()
        explanation = result.get("response", "")
        explanation = clean_text(explanation)

        if not explanation:
            print("    VLM returned empty response.")
            return ""

        print("    VLM explanation created.")
        return explanation

    except requests.exceptions.Timeout:
        print("    VLM timeout.")
        return ""

    except requests.exceptions.ConnectionError:
        print("    ERROR: Ollama is not running.")
        print("    Start Ollama and try again.")
        return ""

    except Exception as e:
        print(f"    VLM warning: {e}")
        return ""


# ============================================================
# RENDER PDF PAGE
# ============================================================

def render_pdf_page(page):
    matrix = fitz.Matrix(RENDER_SCALE, RENDER_SCALE)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    return pix.tobytes("png")


# ============================================================
# PDF PROCESSING
# ============================================================

def process_pdf(pdf_path, chunks):

    print()
    print("=" * 70)
    print(f"PDF: {pdf_path.name}")
    print("=" * 70)

    document = fitz.open(pdf_path)
    total_pages = len(document)

    for page_number, page in enumerate(document, start=1):

        print(f"  Processing PDF page {page_number}/{total_pages}")

        page_text = page.get_text("text")
        page_text = clean_text(page_text)

        if page_text:
            parts = split_text(page_text)
            for part in parts:
                content_type = classify_text_content_type(
                    part, "pdf_formula", "pdf_table", "pdf_text"
                )
                add_chunk(chunks, part, pdf_path, content_type, f"page {page_number}")

        images = page.get_images(full=True)

        if images:
            print(f"    Found {len(images)} embedded image(s)")

        for image_number, image_info in enumerate(images, start=1):
            try:
                xref = image_info[0]
                image_data = document.extract_image(xref)
                image_bytes = image_data["image"]

                explanation = vlm_explain_image(
                    image_bytes,
                    pdf_path.name,
                    f"page {page_number}, image {image_number}"
                )

                if explanation:
                    add_chunk(
                        chunks, explanation, pdf_path, "pdf_image_vlm",
                        f"page {page_number}, image {image_number}"
                    )

                ocr_text = ocr_image(image_bytes)

                if ocr_text:
                    ocr_content_type = (
                        "pdf_formula" if looks_like_formula(ocr_text) else "pdf_image_ocr"
                    )
                    add_chunk(
                        chunks,
                        "Important visible labels/text from the image: " + ocr_text,
                        pdf_path, ocr_content_type,
                        f"page {page_number}, image {image_number}"
                    )

            except Exception as e:
                print(f"    PDF image warning: {e}")

        try:
            rendered_bytes = render_pdf_page(page)
            explanation = vlm_explain_image(
                rendered_bytes, pdf_path.name, f"page {page_number} full-page visual"
            )
            if explanation:
                add_chunk(chunks, explanation, pdf_path, "pdf_page_vlm", f"page {page_number}")

        except Exception as e:
            print(f"    PDF page VLM warning: {e}")

    document.close()


# ============================================================
# PPT IMAGE EXTRACTION
# ============================================================

def extract_ppt_images(shape):
    images = []
    try:
        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            images.append(shape.image.blob)
        elif shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            for child in shape.shapes:
                images.extend(extract_ppt_images(child))
    except Exception:
        pass
    return images


def extract_ppt_placeholder_images(shape, images):
    """
    Some slides put a picture INSIDE a placeholder (Content
    Placeholder, Picture Placeholder). python-pptx reports
    these as shape_type == PLACEHOLDER, not PICTURE, so
    extract_ppt_images() alone misses them. Catch those here
    via shape.image, guarded by try/except.
    """
    try:
        blob = shape.image.blob
        images.append(blob)
    except Exception:
        pass


# ============================================================
# LIBREOFFICE FINDER
# ============================================================

def find_libreoffice():
    possible_paths = [
        r"C:\Program Files\LibreOffice\program\soffice.exe",
        r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"
    ]
    for path in possible_paths:
        if Path(path).exists():
            return path
    found = shutil.which("soffice")
    if found:
        return found
    return None


# ============================================================
# RENDER PPT TO PDF
# ============================================================

def render_ppt_to_pdf(ppt_path):

    soffice = find_libreoffice()

    if not soffice:
        print("  LibreOffice not found.")
        return None

    temp_dir = Path(tempfile.mkdtemp(prefix="ppt_render_"))

    try:
        command = [
            soffice, "--headless", "--convert-to", "pdf",
            "--outdir", str(temp_dir), str(ppt_path)
        ]

        result = subprocess.run(command, capture_output=True, text=True)

        pdf_path = temp_dir / f"{ppt_path.stem}.pdf"

        if pdf_path.exists():
            return pdf_path

        print("  LibreOffice conversion failed.")
        print(result.stderr)
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None

    except Exception as e:
        print(f"  PPT rendering error: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None


# ============================================================
# PROCESS RENDERED PPT
# ============================================================

def process_rendered_ppt(ppt_path, pdf_path, chunks):

    print()
    print("  VLM analysing complete PPT slides...")

    document = fitz.open(pdf_path)
    total_slides = len(document)

    for slide_number, page in enumerate(document, start=1):

        print(f"    Slide {slide_number}/{total_slides}")

        try:
            image_bytes = render_pdf_page(page)
            explanation = vlm_explain_image(
                image_bytes, ppt_path.name, f"slide {slide_number}"
            )
            if explanation:
                add_chunk(chunks, explanation, ppt_path, "ppt_slide_vlm", f"slide {slide_number}")

        except Exception as e:
            print(f"    Slide VLM warning: {e}")

    document.close()


# ============================================================
# PPTX PROCESSING
# ============================================================

def process_pptx(ppt_path, chunks):

    print()
    print("=" * 70)
    print(f"PPTX: {ppt_path.name}")
    print("=" * 70)

    presentation = Presentation(str(ppt_path))
    total_slides = len(presentation.slides)

    for slide_number, slide in enumerate(presentation.slides, start=1):

        print(f"  Processing slide {slide_number}/{total_slides}")

        # TEXT
        slide_text_parts = []
        for shape in slide.shapes:
            try:
                if hasattr(shape, "text"):
                    text_value = clean_text(shape.text)
                    if text_value:
                        slide_text_parts.append(text_value)
            except Exception:
                pass

        slide_text = "\n".join(slide_text_parts)

        if slide_text:
            parts = split_text(slide_text)
            for part in parts:
                content_type = classify_text_content_type(
                    part, "ppt_formula", "ppt_table", "ppt_text"
                )
                add_chunk(chunks, part, ppt_path, content_type, f"slide {slide_number}")

        # EMBEDDED IMAGES — including pictures placed inside placeholders
        image_count = 0

        for shape in slide.shapes:

            images = extract_ppt_images(shape)

            # Catch pictures inside Content/Picture placeholders
            # (shape_type == PLACEHOLDER), which extract_ppt_images
            # does not detect since it only checks PICTURE/GROUP.
            if not images:
                extract_ppt_placeholder_images(shape, images)

            for image_bytes in images:
                image_count += 1
                print(f"    Embedded image {image_count}")

                explanation = vlm_explain_image(
                    image_bytes, ppt_path.name, f"slide {slide_number}, image {image_count}"
                )

                if explanation:
                    add_chunk(
                        chunks, explanation, ppt_path, "ppt_image_vlm",
                        f"slide {slide_number}, image {image_count}"
                    )

                ocr_text = ocr_image(image_bytes)

                if ocr_text:
                    ocr_content_type = (
                        "ppt_formula" if looks_like_formula(ocr_text) else "ppt_image_ocr"
                    )
                    add_chunk(
                        chunks,
                        "Visible labels/text from the image: " + ocr_text,
                        ppt_path, ocr_content_type,
                        f"slide {slide_number}, image {image_count}"
                    )

    # WHOLE SLIDE VISUAL UNDERSTANDING
    rendered_pdf = render_ppt_to_pdf(ppt_path)

    if rendered_pdf:
        try:
            process_rendered_ppt(ppt_path, rendered_pdf, chunks)
        finally:
            shutil.rmtree(rendered_pdf.parent, ignore_errors=True)


# ============================================================
# EXCEL PROCESSING
# ============================================================

def process_excel(excel_path, chunks):

    print()
    print("=" * 70)
    print(f"EXCEL: {excel_path.name}")
    print("=" * 70)

    workbook = openpyxl.load_workbook(excel_path, data_only=True)

    for worksheet in workbook.worksheets:

        print(f"  Worksheet: {worksheet.title}")

        rows = []

        for row in worksheet.iter_rows(values_only=True):
            values = []
            for value in row:
                if value is not None:
                    values.append(str(value))
            if values:
                rows.append(" | ".join(values))

        if rows:
            table_text = "\n".join(rows)
            parts = split_text(table_text)
            for part in parts:
                content_type = "excel_formula" if looks_like_formula(part) else "excel_table"
                add_chunk(chunks, part, excel_path, content_type, f"worksheet {worksheet.title}")


# ============================================================
# FIND INPUT FILES
# ============================================================

def find_files():
    extensions = ["*.pdf", "*.ppt", "*.pptx", "*.xlsx", "*.xlsm"]
    files = []
    for extension in extensions:
        files.extend(DATA_DIR.rglob(extension))
    files = [f for f in files if not f.name.startswith("~$")]
    return sorted(files)


# ============================================================
# CHECK OLLAMA
# ============================================================

def check_ollama():

    print()
    print("Checking Ollama...")

    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        response.raise_for_status()

        models = response.json().get("models", [])
        model_names = [model.get("name", "") for model in models]

        print("Installed Ollama models:")
        for name in model_names:
            print(f"  - {name}")

        if VLM_MODEL in model_names:
            print()
            print(f"OK: {VLM_MODEL} is available.")
            return True

        print()
        print(f"WARNING: {VLM_MODEL} was not found.")
        print(f"Run: ollama pull {VLM_MODEL}")
        return False

    except Exception as e:
        print(f"WARNING: Ollama connection failed: {e}")
        return False


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("MULTIMODAL RAG - QWEN2.5-VL INDEX BUILDER")
    print("=" * 70)

    print()
    print("Project:")
    print(PROJECT_ROOT)

    print()
    print("DATA:")
    print(DATA_DIR)

    check_ollama()

    files = find_files()

    print()
    print(f"Found {len(files)} input file(s).")

    for file in files:
        print(f"  - {file.relative_to(DATA_DIR)}")

    if not files:
        print()
        print("No PDF/PPT/PPTX/Excel files found.")
        return

    chunks = []

    for file_path in files:

        suffix = file_path.suffix.lower()

        try:
            if suffix == ".pdf":
                process_pdf(file_path, chunks)
            elif suffix in [".ppt", ".pptx"]:
                process_pptx(file_path, chunks)
            elif suffix in [".xlsx", ".xlsm"]:
                process_excel(file_path, chunks)

        except Exception as e:
            print()
            print(f"ERROR processing {file_path.name}: {e}")

    print()
    print("=" * 70)
    print("CHUNK SUMMARY")
    print("=" * 70)

    print(f"Total chunks: {len(chunks)}")

    type_counts = {}
    for chunk in chunks:
        content_type = chunk["content_type"]
        type_counts[content_type] = type_counts.get(content_type, 0) + 1

    for content_type, count in sorted(type_counts.items()):
        print(f"  {content_type}: {count}")

    if not chunks:
        print()
        print("No chunks were created.")
        return

    print()
    print("Loading embedding model...")

    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [chunk["text"] for chunk in chunks]

    print()
    print(f"Creating embeddings for {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    print()
    print("Building FAISS index...")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    RAG_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(FAISS_INDEX_PATH))

    with open(CHUNKS_PATH, "wb") as file:
        pickle.dump(chunks, file)

    print()
    print("=" * 70)
    print("MULTIMODAL INDEX BUILD COMPLETE")
    print("=" * 70)

    print()
    print("FAISS index:")
    print(FAISS_INDEX_PATH)

    print()
    print("Chunks:")
    print(CHUNKS_PATH)

    print()
    print("Index contains:")
    print("  ✓ PDF text")
    print("  ✓ PDF tables")
    print("  ✓ PDF formulas")
    print("  ✓ PDF image OCR")
    print("  ✓ PDF image VLM")
    print("  ✓ PDF full-page VLM")
    print("  ✓ PPT text")
    print("  ✓ PPT formulas")
    print("  ✓ PPT image OCR")
    print("  ✓ PPT image VLM (including placeholder images)")
    print("  ✓ PPT full-slide VLM")
    print("  ✓ Excel tables")
    print("  ✓ Excel formulas")

    print()
    print("The multimodal RAG index is ready.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
