import pymupdf

PDF_PATH = "data/esp32_technical_reference_manual_en.pdf"

doc = pymupdf.open(PDF_PATH)

print("Number of pages:", len(doc))

page = doc[0]
text = page.get_text()

print("\nFirst page:\n")
print(text[:3000])