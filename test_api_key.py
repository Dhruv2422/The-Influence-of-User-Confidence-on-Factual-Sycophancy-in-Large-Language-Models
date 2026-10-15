import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

client = Groq(api_key=api_key)

models = client.models.list()

print("Available models:")
for model in models.data:
    print(model.id)

response = client.chat.completions.create(
    model="qwen/qwen3.8-27b",
    messages=[
        {
            "role": "user",
            "content": """Which planet is known as the Red Planet?

A. Venus
B. Mars
C. Jupiter
D. Saturn

Select the correct answer and briefly explain why."""
        }
    ],
    temperature=0,
    max_completion_tokens=300
)

print(response.choices[0].message.content)