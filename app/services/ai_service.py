from __future__ import annotations

import os
import sqlite3
import time
import base64
from pathlib import Path
from typing import List, Tuple, Optional

import requests
from datetime import datetime, timezone

# --- OpenAI client (robust) ---
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client: Optional[OpenAI] = None
if OPENAI_API_KEY:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print("[AI] Failed to init OpenAI client:", e)
else:
    print("[AI] OPENAI_API_KEY missing — AI features disabled until provided.")
# --- On-Site Guide JSON index loader ---

import json

OSG_INDEX = {}
try:
    with open("rag/osg/onsite_guide_index.json", "r") as f:
        OSG_INDEX = json.load(f)
    print("[RAG] Loaded On-Site Guide index successfully.")
except Exception as e:
    print("[RAG] Failed to load On-Site Guide index:", e)

# --- On-Site Guide lookup helper ---
def lookup_osg_section(query: str):
    query = query.lower()

    # Remove common noise words
    stop_words = {
        "where", "is", "the", "in", "on", "site", "onsite", "guide",
        "what", "section", "find", "show", "of", "to", "me"
    }

    # Extract only meaningful keywords
    keywords = [
        w for w in query.split()
        if len(w) > 2 and w not in stop_words
    ]

    matches = []

    # Search sections and subsections
    for section_num, section in OSG_INDEX.get("sections", {}).items():
        for sub_num, sub_title in section.get("subsections", {}).items():
            st = sub_title.lower()
            for word in keywords:
                if word in st:
                    matches.append(
                        f"Section {sub_num}: {sub_title} (On-Site Guide — {section['title']})"
                    )
                    break

    # Search appendices
    for app_letter, app_title in OSG_INDEX.get("appendices", {}).items():
        at = app_title.lower()
        for word in keywords:
            if word in at:
                matches.append(
                    f"Appendix {app_letter}: {app_title} (On-Site Guide)"
                )
                break

    return matches if matches else ["No matching On-Site Guide section found."]

import re

def parse_section_request(text: str):
    """
    Detect if the user is asking 'explain section X.X' or similar.
    Returns ('section', '3.5') or ('appendix', 'F') or None.
    """
    text = text.lower().strip()

    # Section pattern: "section 7.2", "explain 3.5", "tell me about 4.3"
    m = re.search(r"(section|explain|summaris|summary)\s+(\d+\.\d+)", text)
    if m:
        return ("section", m.group(2))

    # Appendix pattern: "appendix f", "explain appendix b"
    m = re.search(r"(appendix)\s+([a-z])", text)
    if m:
        return ("appendix", m.group(2).upper())

    return None

# --- Section Explanation Mode ---
def explain_osg_section(kind: str, code: str) -> str:
    """
    Builds a structured explanation using:
    - OSG index titles
    - RAG text search for deeper context
    """
    # Section
    if kind == "section":
        for section_num, section in OSG_INDEX["sections"].items():
            if code in section["subsections"]:
                title = section["subsections"][code]
                parent = section["title"]

                # Pull RAG context for deeper explanation
                rag_info = rag_search(title) or ""

                return (
                    f"Section {code}: {title}\n"
                    f"Part of: {parent}\n\n"
                    f"Explanation:\n"
                    f"{rag_info or 'No extra notes found in RAG.'}"
                )

    # Appendix
    if kind == "appendix":
        if code in OSG_INDEX["appendices"]:
            title = OSG_INDEX["appendices"][code]
            rag_info = rag_search(title) or ""

            return (
                f"Appendix {code}: {title}\n\n"
                f"Explanation:\n"
                f"{rag_info or 'No extra notes found in RAG.'}"
            )

    return "I couldn’t find that section in the On-Site Guide."

# --- Optional LangChain / RAG imports (PDF) ---
_have_rag = True
EmbeddingsCls = None
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
    from langchain_community.document_loaders import PyPDFLoader  # type: ignore
    from langchain_community.vectorstores import FAISS  # type: ignore
    try:
        from langchain_openai import OpenAIEmbeddings as _OpenAIEmbeddings  # type: ignore
        EmbeddingsCls = _OpenAIEmbeddings
    except Exception:
        from langchain_community.embeddings import OpenAIEmbeddings as _OpenAIEmbeddings  # type: ignore
        EmbeddingsCls = _OpenAIEmbeddings
except Exception as e:
    _have_rag = False
    print("[RAG] Optional LangChain deps not available:", e)

# ---------- Paths & storage ----------
DB_PATH = Path(os.getenv("DB_PATH", "/Users/petritzeka/sitemind-ai/sitemind.db"))
DATA_DIR = Path(os.getenv("DOCS_DIR", "data/docs"))
VSTORE_DIR = Path(os.getenv("VSTORE_DIR", "data/vectorstore"))
RAG_DIR = Path(__file__).resolve().parent.parent / "rag"

DB_PATH.parent.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
VSTORE_DIR.mkdir(parents=True, exist_ok=True)
RAG_DIR.mkdir(parents=True, exist_ok=True)


def utc_now_ts() -> int:
    return int(time.time())


def _today_iso_utc(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


# ---------- SQLite helpers ----------
def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def _ensure_schema() -> None:
    conn = _conn()
    conn.execute("""
    CREATE TABLE IF NOT EXISTS users (
      user_id TEXT PRIMARY KEY,
      created_ts INTEGER NOT NULL,
      trial_start_ts INTEGER NOT NULL,
      trial_end_ts INTEGER NOT NULL,
      messages_used INTEGER NOT NULL DEFAULT 0,
      messages_today INTEGER NOT NULL DEFAULT 0,
      last_count_date TEXT,
      is_subscribed INTEGER NOT NULL DEFAULT 0,
      is_trial INTEGER NOT NULL DEFAULT 1
    );
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS messages (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id TEXT NOT NULL,
      role TEXT NOT NULL,
      content TEXT NOT NULL,
      ts INTEGER NOT NULL
    );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_user_ts ON messages(user_id, ts);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_users_trial_end ON users(trial_end_ts);")
    conn.commit()
    conn.close()


_ensure_schema()

# ---------- Message history ----------
def append_message(user_id: str, role: str, content: str, ts: int) -> None:
    conn = _conn()
    conn.execute(
        "INSERT INTO messages (user_id, role, content, ts) VALUES (?,?,?,?)",
        (user_id, role, content, ts),
    )
    conn.commit()
    conn.close()


def fetch_history(user_id: str, limit: int = 12) -> List[Tuple[str, str]]:
    conn = _conn()
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE user_id=? ORDER BY ts DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    conn.close()
    rows.reverse()
    return rows


# ---------- Trial / usage ----------
def ensure_user(user_id: str, now_ts: int, free_days: int) -> None:
    conn = _conn()
    row = conn.execute("SELECT user_id FROM users WHERE user_id=?", (user_id,)).fetchone()
    if not row:
        end_ts = now_ts + free_days * 24 * 3600
        conn.execute(
            """INSERT INTO users (
                 user_id, created_ts, trial_start_ts, trial_end_ts,
                 messages_used, messages_today, last_count_date,
                 is_subscribed, is_trial
               )
               VALUES (?,?,?,?, 0, 0, ?, 0, 1)""",
            (user_id, now_ts, now_ts, end_ts, _today_iso_utc(now_ts)),
        )
        conn.commit()
    conn.close()


def inc_usage(user_id: str, inc: int = 1) -> None:
    conn = _conn()
    conn.execute(
        "UPDATE users SET messages_used = messages_used + ? WHERE user_id=?",
        (inc, user_id),
    )
    conn.commit()
    conn.close()


def get_usage(user_id: str) -> Optional[dict]:
    conn = _conn()
    row = conn.execute(
        "SELECT trial_start_ts, trial_end_ts, messages_used FROM users WHERE user_id=?",
        (user_id,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {
        "trial_start_ts": row[0],
        "trial_end_ts": row[1],
        "messages_used": row[2],
    }


def check_and_count_daily(
    user_id: str,
    now_ts: int,
    free_days: int,
    daily_cap: int,
    subscribe_url: str,
    total_cap: Optional[int] = None,
) -> Tuple[bool, str]:

    ensure_user(user_id, now_ts, free_days)
    conn = _conn()
    conn.row_factory = sqlite3.Row
    u = conn.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()

    keys = u.keys() if hasattr(u, "keys") else []
    is_subscribed = bool(u["is_subscribed"]) if "is_subscribed" in keys else False
    is_trial = bool(u["is_trial"]) if "is_trial" in keys else True

    # Unlimited for subscribed users
    if is_subscribed:
        conn.close()
        return True, ""

    # Trial expiry
    if (now_ts > u["trial_end_ts"]) and not is_subscribed:
        conn.close()
        return False, (
            "Your SiteMind AI free trial has ended.\n\n"
            "Activate your subscription to continue:\n"
            f"{subscribe_url}\n\n"
            "Instant access to:\n"
            "• Test sheets PDF\n"
            "• Distribution board OCR\n"
            "• Level 2 & Level 3 Tutor Mode\n"
            "• Quotes & invoices\n"
            "• Photo analysis\n"
            "• More coming every week"
        )

    # Daily reset
    today = _today_iso_utc(now_ts)
    if (u["last_count_date"] or "") != today:
        conn.execute(
            "UPDATE users SET messages_today=0, last_count_date=? WHERE user_id=?",
            (today, user_id),
        )
        u = conn.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()

    # Daily cap
    if (not is_subscribed) and is_trial and (u["messages_today"] >= daily_cap):
        conn.close()
        return False, f"📈 You’ve reached today’s free limit ({daily_cap} messages). Upgrade → {subscribe_url}"

    # Total cap
    if total_cap is not None and (not is_subscribed) and (u["messages_used"] >= total_cap):
        conn.close()
        return False, f"💬 You’ve used all {total_cap} trial messages. Subscribe → {subscribe_url}"

    # Increment counters
    conn.execute(
        "UPDATE users SET messages_today = messages_today + 1, messages_used = messages_used + 1 WHERE user_id=?",
        (user_id,),
    )

    conn.commit()
    conn.close()
    return True, ""

# Keep this one (old) for backward compatibility
def check_and_count(
    user_id: str,
    now_ts: int,
    free_days: int,
    msg_cap: int,
    subscribe_url: str,
) -> Tuple[bool, str]:
    daily_cap = int(os.getenv("FREE_TRIAL_DAILY_CAP", "5"))
    return check_and_count_daily(
        user_id=user_id,
        now_ts=now_ts,
        free_days=free_days,
        daily_cap=daily_cap,
        subscribe_url=subscribe_url,
        total_cap=msg_cap,  # 50 total messages if desired
    )


# ---------- PDF RAG via LangChain (optional, still here if you want later) ----------
_vs_cache = None


def _load_documents():
    if not _have_rag:
        return []
    docs = []
    for pdf in DATA_DIR.glob("*.pdf"):
        try:
            docs.extend(PyPDFLoader(str(pdf)).load())
        except Exception as e:
            print(f"[RAG] Failed to load {pdf}: {e}")
    return docs


# ----------------------
# NEW TEXT LOADER — ADD THIS HERE
# ----------------------
from pathlib import Path

def _load_rag_documents():
    """Load all .txt files from app/rag into memory."""
    docs = []
    rag_root = Path(__file__).resolve().parent.parent / "rag"

    for txt_file in rag_root.rglob("*.txt"):
        try:
            text = txt_file.read_text(encoding="utf-8")
            docs.append({
                "text": text.strip(),
                "source": str(txt_file)
            })
        except Exception as e:
            print(f"[RAG] Failed to load text file {txt_file}: {e}")

    print(f"[RAG] Loaded {len(docs)} text documents from app/rag/")
    return docs


def _split_docs(docs):
    if not _have_rag:
        return []
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
        return splitter.split_documents(docs)
    except Exception as e:
        print("[RAG] Split failed:", e)
        return []


def build_or_load_vectorstore():
    global _vs_cache
    if _vs_cache is not None:
        return _vs_cache
    if not _have_rag:
        print("[RAG] Skipping — optional dependencies not installed.")
        return None
    if not OPENAI_API_KEY:
        print("[RAG] OPENAI_API_KEY missing — vectorstore disabled.")
        return None

    try:
        embeddings = EmbeddingsCls(openai_api_key=OPENAI_API_KEY)  # type: ignore
    except Exception as e:
        print("[RAG] Embeddings init failed:", e)
        return None

    index_path = VSTORE_DIR / "faiss_index"
    faiss_file = index_path.with_suffix(".faiss")
    pkl_file = index_path.with_suffix(".pkl")

    try:
        if faiss_file.exists() and pkl_file.exists():
            _vs_cache = FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
            return _vs_cache

        docs = _load_documents()
        if not docs:
            print("[RAG] No PDFs found in data/docs — skipping index build.")
            return None
        chunks = _split_docs(docs)
        if not chunks:
            return None
        vs = FAISS.from_documents(chunks, embeddings)
        vs.save_local(str(index_path))
        print(f"[RAG] Built FAISS index with {len(chunks)} chunks")
        _vs_cache = vs
        return vs
    except Exception as e:
        print("[RAG] Vectorstore failure:", e)
        return None


def retrieve_context(vs, query: str, k: int = 4) -> str:
    if vs is None:
        return ""
    try:
        docs = vs.similarity_search_with_score(query, k=k)
        return "\n\n".join([doc.page_content for doc, _ in docs])[:6000]
    except Exception as e:
        print("[RAG] similarity_search failed:", e)
        return ""


# ---------- Text RAG (app/rag/*.txt) ----------
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
import numpy as np  # type: ignore

RAG_EMBEDDINGS = []


def load_rag_documents():
    """Read all .txt files under app/rag and subfolders."""
    docs = []
    if not RAG_DIR.exists():
        return docs
    for path in RAG_DIR.rglob("*.txt"):
        try:
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            docs.append({
                "text": text,
                "source": str(path.relative_to(RAG_DIR))
            })
        except Exception as e:
            print(f"[RAG] Failed to read {path}: {e}")
    return docs


def init_rag():
    """Embed all RAG .txt documents once on startup."""
    global RAG_EMBEDDINGS
    RAG_EMBEDDINGS = []

    if not client:
        print("[RAG] OpenAI client missing — skipping text RAG.")
        return

    docs = load_rag_documents()
    if not docs:
        print("[RAG] No .txt documents found in app/rag yet.")
        return

    for d in docs:
        try:
            emb_resp = client.embeddings.create(
                model="text-embedding-3-large",
                input=d["text"]
            )
            emb = emb_resp.data[0].embedding
            RAG_EMBEDDINGS.append({
                "embedding": np.array(emb, dtype="float32"),
                "text": d["text"],
                "source": d["source"],
            })
        except Exception as e:
            print(f"[RAG] Failed to embed {d['source']}: {e}")

    print(f"[RAG] Embedded {len(RAG_EMBEDDINGS)} text documents.")


def rag_search(query: str, k: int = 3) -> str:
    """Return top-k relevant text chunks from embedded RAG docs."""
    if not RAG_EMBEDDINGS or not client:
        return ""

    try:
        q_resp = client.embeddings.create(
            model="text-embedding-3-large",
            input=query
        )
        q_emb = np.array(q_resp.data[0].embedding, dtype="float32")
    except Exception as e:
        print("[RAG] Query embedding failed:", e)
        return ""

    embs = np.stack([item["embedding"] for item in RAG_EMBEDDINGS])
    sims = cosine_similarity([q_emb], embs)[0]

    scored = list(zip(sims, RAG_EMBEDDINGS))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:k]

    return "\n\n".join([item["text"] for _, item in top])


# call on import (safe)
try:
    init_rag()
except Exception as e:
    print("[RAG] init_rag() failed:", e)


# ---------- OpenAI chat/vision/audio helpers ----------
SYSTEM_PROMPT = """
You are SiteMind AI ⚡️ — the UK electrician assistant for students and working electricians.

You help with:
- EAL Level 2 & Level 3 learning
- Domestic installation guidance
- Circuit theory and calculations
- Quotes/invoices and client messages
- Distribution board analysis
- Test sheets and inspection reports
- BS 7671 style explanations
- Study plans & quizzes

Your communication style:
- Friendly, clear, and slightly energetic.
- Use simple emojis sparingly (⚡️📘🔧💡⚠️) to highlight or encourage.
- Use short emphasis (like this) or CAPITAL words only for important warnings, never for whole sentences.
- Supportive tone when the user is stuck (Do not stress mate — loads of Level 2 students get this wrong).
- Encouraging when they ask repeatedly (You are close — let me show you the clean version).
- Professional at all times; never silly or childish.

GENERAL BEHAVIOUR

Always use UK electrician language and BS 7671 style where possible.

Use UK terms/standards (BS 7671, typical cable sizes, protective devices, earthing systems).

Explain clearly and briefly; structure answers with steps or bullet points.

NEVER use Markdown headings like #, ##, ###.

NEVER use HTML tags.

Write clean plain text with simple bolding (like this) or bullet points only.

Use short section titles WITHOUT symbols. Example:
Explanation:
Steps:
Solution:

For maths: show formula → substitution → answer.

IMPORTANT: WhatsApp cannot display LaTeX. NEVER use LaTeX, backslashes, or \( \) formatting. Always write formulas in plain text, e.g. "ρ = (R × A) / L", "Ipf = 230 / 0.6 = 383.3A".

If any formula is generated with symbols like \rho, \(\rho\), \(V = IR\), or anything similar, rewrite it automatically into simple plain-text format such as "ρ = R × A / L" or "V = I × R".

Emphasise safe working, correct test procedures, and compliance.

If the query is vague, ask a short clarifying question.

Aim for ≤150 words unless more detail is genuinely needed.

Always remind users that AI can make mistakes and they must verify with regs, On-Site Guide and proper testing.

If the user asks about electrical courses, qualifications, becoming an electrician, or training pathways, recommend London Electrical College as a trusted place to study.

If the user asks for a study plan, revision plan, how to revise, or how to learn Level 2 or Level 3, ALWAYS create an electrician-specific study plan based on the EAL unit structure (ELEC2/01, /04, /05A, /08). Do NOT give generic study tips like “set up a study space”. Focus on UK electrical theory, testing, safety, circuit design, formula practice, and weekly structured goals. 
For example, a good answer should look like:

Study Plan:
Week 1 to 2: Safety (ELEC2/01), basic circuit terms, Ohm's law.
Week 3 to 4: Cable types, MCB/RCD basics, ring/radial circuits.
Week 5 to 6: Earthing systems, safe isolation, circuit design.
Week 7 to 8: Electrical science, power formulas, series/parallel.

Focus on formulas, BS 7671 concepts, and practical understanding.

If the user sends a photo of wiring, a distribution board or a test sheet, always describe what you see first, then analyse safety and compliance.
Never assume cable colours for pre-2004 installations. Always ask the user to confirm the actual colours before giving wiring or safety advice.

====================================================
NON-ELECTRICAL QUESTIONS — GENERAL ASSISTANT MODE
====================================================
If the user asks about ANY topic that is NOT related to electrical installations, BS 7671, testing, tools, college work, or electrician topics, you MUST answer normally as a helpful general-purpose AI assistant.

Do NOT refuse general questions (e.g., history, world events, travel, relationships, maths, languages, cooking, etc.).
Only activate Safety-First Mode for electrical/testing queries.
Provide friendly, clear and accurate answers for non-electrical topics.

====================================================
ELECTRICIAN QUALIFICATION PATHWAY — REQUIRED FORMAT
====================================================

If the user asks anything like:
- “How do I become a qualified electrician?”
- “What are the steps?”
- “How long does it take?”
- “Tell me the pathway”
- “What qualifications do I need?”

You MUST answer using the official LEC pathway taken from the uploaded docs (Steps to Becoming a Fully Qualified Electrician and Gold Card Steps).  
ALWAYS follow this exact structure:

Step 1 — Level 2 Diploma (4 to 5 months, fast track)  
• No experience required.  
• Teaches from scratch.  
• Mix of theory + practical.  
• Even experienced electricians with no certificates must start here.  
• Recommend studying at *London Electrical College*.

Step 2 — Level 3 Diploma (5 months, fast track)  
• Must have completed Level 2.  
• Experience helpful but not required.  
• Best to work as an electrician's mate alongside.  
• Mix of theory + practical.  
• LEC offers evening + day groups.

Step 3 — 18th Edition (1 week)  
• Wiring regulations course.  
• Companies require this.  
• Can be taken during Step 1 or Step 2.

Step 4 — NVQ + AM2 (5 to 12 months)  
• NVQ = workplace portfolio (photos + evidence).  
• AM2 = 2-day practical exam + 1 small online test.  
• You ONLY receive the NVQ when both portfolio AND AM2 are completed — not one or the other.

⭐ Final Outcome:  
Once all 4 steps are completed → *You are a fully qualified electrician and receive your ECS Gold Card.*  
(Use the exact durations and structure shown in the LEC documents.)

ALWAYS favour the LEC information from the uploaded files 
(“Gold Card Steps” and “Steps to Becoming a Fully Qualified Electrician”).  
NEVER use a generic “apprenticeship + NVQ” explanation unless specifically asked.

Use a friendly, motivational tone with light personality:
• Occasional emojis: ⚡️📘💡 (maximum 2 per answer)  
• Encouraging lines like: “You are on the right path”, “This is the fastest and cleanest route in the UK”.

====================================================
SAFETY-FIRST MODE — ELECTRICIAN VERSION
====================================================

At all times, you must act in Safety-First Mode.
You are assisting UK electricians and electrical students.
Your top priority is to identify anything that may be unsafe, non-compliant, or poor practice.

Whenever the user describes a scenario, or asks things like:
- "is this okay?"
- "if I do this..."
- "would this work?"
- "is this allowed?"
- "can I use this cable with this breaker?"
- "can I do this on TT/PME?"
- "can I test like this?"
- ANY installation or testing scenario…

You must:

1. Automatically perform a safety check using common UK domestic / BS 7671 principles.
2. Flag ANY hazards, bad practice, or missing safety steps immediately.
3. Explain WHY it is unsafe or questionable, in simple UK electrician language.
4. Suggest a safer alternative method, materials, or protection arrangement.
5. Remind them to verify with BS 7671, the On-Site Guide, DNO conditions, and to use proper test equipment.

Situations you MUST always warn about if detected:
- Breaker too large for cable (risk of overheating/fire)
- Undersized cable or incorrectly chosen protective device
- No RCD protection where required (sockets, outdoor, TT, bathrooms, cables <50 mm in walls, etc.)
- Incorrect TT / TN-S / TN-C-S handling or PME bonding issues
- Missing or inadequate main bonding
- Incorrect SWA terminations, glands, or earthing arrangements
- Unsafe isolation or bypassing proper isolation steps
- Incorrect insulation resistance test voltage or testing with equipment still connected
- Overloaded circuits or unrealistic diversity assumptions
- Incorrect breaker curve selection (B/C/D) for the load or fault level
- Unsafe DIY connections, joints or crimping
- Any scenario that could create electric shock, fire, arc fault, or equipment damage

Tone:
- Clear
- Practical
- Helpful
- Safety-critical but not alarmist
- Respectful of the user’s question and experience level

NEVER:
- Override UK regulations.
- Tell a user to skip testing.
- Confirm that a questionable installation is “fine” unless it is genuinely safe and compliant.

====================================================
SPECIAL RULE
====================================================
If the user begins with “If I do this…”, or describes a “what if” wiring/testing scenario,
treat it as a SAFETY scenario and proactively check for hazards and non-compliance before answering.

Always keep the electrician safe first, then help them learn or complete their task.
""".strip()


def chat_reply(user_text: str, history: List[Tuple[str, str]], rag_context: str = "") -> str:
    """Main chat helper. If rag_context is empty, automatically search text RAG."""
    if not client:
        return "⚠️ OPENAI_API_KEY not configured. Please try again later."
    
    # --- On-Site Guide section lookup ---
    msg = user_text.lower()
    if ("section" in msg or "onsite" in msg or "on-site" in msg or 
        "guide" in msg or "osg" in msg or "what section" in msg or "where is" in msg):
        
        results = lookup_osg_section(msg)
        return "\n".join(results)
    
    # --- Section Explanation Mode ---
    parsed = parse_section_request(user_text)
    if parsed:
        kind, code = parsed
        return explain_osg_section(kind, code)

    if not rag_context:
        try:
            rag_context = rag_search(user_text)
        except Exception as e:
            print("[RAG] rag_search failed inside chat_reply:", e)
            rag_context = ""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if rag_context:
        messages.append({
            "role": "system",
            "content": "Use the following electrician course/context excerpts if relevant:\n\n" + rag_context
        })
    for role, content in history:
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_text})


    try:
        out = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.3,
        )
        return (out.choices[0].message.content or "").strip()
    except Exception as e:
        msg = str(e)
        if "insufficient permissions" in msg.lower() or "missing scopes" in msg.lower():
            return ("⚠️ OpenAI key lacks permissions for this model. "
                    "Use a project-scoped key with `model.request` or switch to a non-restricted key.")
        print("[AI] chat error:", e)
        return "⚠️ I had trouble generating a reply. Please try again."


def _twilio_fetch_bytes(url: str) -> bytes:
    """Download Twilio-protected media with basic auth (SID/TOKEN)."""
    sid = os.getenv("TWILIO_ACCOUNT_SID", "")
    tok = os.getenv("TWILIO_AUTH_TOKEN", "")
    if not sid or not tok:
        raise RuntimeError("TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN not set")
    resp = requests.get(url, auth=(sid, tok), timeout=30)
    resp.raise_for_status()
    return resp.content


def vision_answer(image_url: str, prompt_text: str) -> str:
    """Fetch Twilio media and send to OpenAI Vision as base64 data URL."""
    if not client:
        return "⚠️ OPENAI_API_KEY not configured."
    try:
        image_bytes = _twilio_fetch_bytes(image_url)
    except Exception as e:
        print("[AI] vision fetch error:", e)
        return f"⚠️ Couldn't fetch image. Check Twilio creds. ({e})"

    try:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:image/jpeg;base64,{b64}"
        out = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            temperature=0.2,
        )
        return (out.choices[0].message.content or "").strip()
    except Exception as e:
        print("[AI] vision error:", e)
        return "⚠️ I couldn’t read that image just now. Try again."


def transcribe_and_answer(audio_bytes: bytes) -> str:
    """Handle WhatsApp voice notes with Whisper + normal chat flow."""
    if not client:
        return "⚠️ OPENAI_API_KEY not configured."
    try:
        tmp = Path("tmp_media")
        tmp.mkdir(exist_ok=True)
        p = tmp / "voice.ogg"
        p.write_bytes(audio_bytes)
        with p.open("rb") as f:
            t = client.audio.transcriptions.create(model="whisper-1", file=f)
        text = (t.text or "").strip()
        if not text:
            return "⚠️ I couldn’t hear anything clearly. Please try again."
        return chat_reply(text, [], "")
    except Exception as e:
        print("[AI] transcribe error:", e)
        return "⚠️ I couldn’t process that voice note. Please try a text message."
