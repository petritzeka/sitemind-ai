import os
import time
from openai import OpenAI

# -----------------------------------------------------------------------------
# Environment & Client Setup
# -----------------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY not set. "
        "Add it to your .env file or hosting environment variables."
    )

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------------------------

PRIMARY_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gpt-4o")
SECONDARY_FALLBACK = os.getenv("SECONDARY_FALLBACK_MODEL", "gpt-4.1")

MODEL_CHAIN = [
    PRIMARY_MODEL,        # Primary
    FALLBACK_MODEL,       # First fallback
    SECONDARY_FALLBACK,   # Final fallback
]

# -----------------------------------------------------------------------------
# Fallback Chat Completion Handler
# -----------------------------------------------------------------------------

def call_with_fallback(messages, temperature=0.4, timeout=12):
    """
    Attempts to call OpenAI using a chain of models.
    Falls back automatically if a model fails.
    """

    last_error = None

    for model in MODEL_CHAIN:
        try:
            print(f"[AI] Attempting model: {model}")

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
            )

            print(f"[AI] Success using model: {model}")
            return response

        except Exception as e:
            print(f"[AI ERROR] Model failed: {model}")
            last_error = e
            time.sleep(0.25)

    raise RuntimeError(f"All OpenAI models failed. Last error: {last_error}")
