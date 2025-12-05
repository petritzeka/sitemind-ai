from openai import OpenAI

client = OpenAI()

# Ordered list of models (top = preferred, bottom = last resort)
MODEL_CHAIN = [
    "gpt-5-mini",     # main model (you can change later)
    "gpt-4.1",        # fallback if 5-mini goes down
    "gpt-4o-mini",    # lighter fallback
]

def call_with_fallback(messages, temperature=0.4):
    """
    Tries each model in MODEL_CHAIN until one succeeds.
    Prevents downtime when OpenAI updates or breaks models.
    """
    last_error = None

    for model in MODEL_CHAIN:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            print(f"[AI] Using model: {model}")
            return response

        except Exception as e:
            print(f"[AI ERROR] {model} failed → trying next model...")
            last_error = e

    raise Exception(f"❌ All fallback models failed: {last_error}")
