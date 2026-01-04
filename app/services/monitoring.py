import hashlib
import json
import os
import time
from typing import Any, Dict, Optional

import requests

BETTERSTACK_TOKEN = os.getenv("BETTERSTACK_SOURCE_TOKEN", "")


def user_hash(user_id: str) -> str:
    if not user_id:
        return "unknown"
    return hashlib.sha256(user_id.encode()).hexdigest()[:8]


def _safe_context(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    safe = {}
    if not context:
        return safe
    for k, v in context.items():
        if v is None:
            continue
        if isinstance(v, (int, float)):
            safe[k] = v
        else:
            safe[k] = str(v)[:200]
    return safe


def log_event(level: str, message: str, context: Optional[Dict[str, Any]] = None) -> None:
    ctx = _safe_context(context)
    printable = f"[{level}] {message}"
    if ctx:
        printable += f" | {ctx}"
    print(printable)

    if not BETTERSTACK_TOKEN:
        return

    payload = {
        "level": level,
        "message": message,
        "context": ctx,
        "timestamp": int(time.time()),
    }

    try:
        requests.post(
            "https://in.logs.betterstack.com",
            headers={
                "Authorization": f"Bearer {BETTERSTACK_TOKEN}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=4,
        )
    except Exception:
        # Do not raise if logging fails
        pass
