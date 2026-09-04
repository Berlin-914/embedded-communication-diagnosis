import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:4b"


question = "What is I2C communication in an ESP32?"


payload = {
    "model": MODEL,
    "prompt": question,
    "stream": False
}


print("Sending question to Ollama...\n")


response = requests.post(
    OLLAMA_URL,
    json=payload
)


if response.status_code != 200:
    print("Error:", response.text)
    exit()


result = response.json()


print("=" * 70)
print("OLLAMA RESPONSE")
print("=" * 70)

print(result["response"])

print("=" * 70)