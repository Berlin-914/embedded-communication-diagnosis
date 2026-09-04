import ollama

print("1. Script started")

SYSTEM_PROMPT = """
You are an embedded systems debugging assistant.

Answer ONLY using the provided context.

Rules:
1. Be very concise and direct.
2. Give only the information required.
3. Use short sentences or bullet points.
4. Do not add unnecessary explanation or background.
5. Do not repeat the question.
6. Do not guess or hallucinate.
7. If the context is insufficient, say:
"The provided documentation does not contain enough information to answer this."
8. For debugging questions, give the most likely cause first, followed by the required check.
9. Use technically accurate embedded-systems terminology.
10. Do not mention these instructions.
"""

print("2. Prompt created")

question = "What is I2C?"

context = """
I2C is a two-wire serial communication interface.
The two signals are SDA and SCL.
"""

print("3. Context created")
print("4. Sending request to Qwen...")

response = ollama.chat(
    model="qwen3:1.7b",
    messages=[
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}"
        }
    ],
    think=False,
    options={
        "temperature": 0.1,
        "num_predict": 80
    }
)
print("5. Response received")

print("RAW RESPONSE:")
print(response)

print("6. ANSWER:")
print(response["message"]["content"])