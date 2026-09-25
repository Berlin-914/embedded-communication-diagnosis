import pytesseract
from PIL import Image

image = Image.open("test.png")

text = pytesseract.image_to_string(image)

print("===== OCR RESULT =====")
print(text)