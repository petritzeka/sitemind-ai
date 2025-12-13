# app/services/model_fallback.py
import os
import time
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PRIMARY_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gpt-4o")

MODEL_CHAIN = [
    PRIMARY_MODEL,
    FALLBACK_MODEL,
]

def call_with_fallback(messages, temperature=0.4) -> str:
    last_error = None

    for model in MODEL_CHAIN:
        try:
            print(f"[AI] Attempting model: {model}")

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=15,
            )

            print(f"[AI] Success using model: {model}")
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"[AI ERROR] Model failed: {model}")
            last_error = e
            time.sleep(0.3)

    raise RuntimeError(f"All models failed: {last_error}")
