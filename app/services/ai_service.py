import os, sqlite3, time
from pathlib import Path
from typing import List, Tuple
import requests
from openai import OpenAI

# LangChain RAG
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

DB_PATH = Path("db/sitemind.db")
DATA_DIR = Path("data/docs")
VSTORE_DIR = Path("data/vectorstore")
VSTORE_DIR.mkdir(parents=True, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=OPENAI_API_KEY)

def utc_now_ts() -> int:
    return int(time.time())

# ---------- MEMORY ----------
def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def append_message(user_id: str, role: str, content: str, ts: int):
    conn = _conn()
    conn.execute("INSERT INTO messages (user_id, role, content, ts) VALUES (?,?,?,?)",
                 (user_id, role, content, ts))
    conn.commit()
    conn.close()

def fetch_history(user_id: str, limit: int) -> List[Tuple[str,str]]:
    conn = _conn()
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE user_id=? ORDER BY ts DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    conn.close()
    rows.reverse()
    return rows

def ensure_user(user_id: str, now_ts: int, free_days: int):
    conn = _conn()
    row = conn.execute("SELECT user_id FROM users WHERE user_id=?", (user_id,)).fetchone()
    if not row:
        end_ts = now_ts + free_days*24*3600
        conn.execute(
            "INSERT INTO users (user_id, created_ts, trial_start_ts, trial_end_ts, messages_used) VALUES (?,?,?,?,0)",
            (user_id, now_ts, now_ts, end_ts),
        )
        conn.commit()
    conn.close()

def inc_usage(user_id: str, inc: int = 1):
    conn = _conn()
    conn.execute("UPDATE users SET messages_used = messages_used + ? WHERE user_id=?",
                 (inc, user_id))
    conn.commit()
    conn.close()

def get_usage(user_id: str):
    conn = _conn()
    row = conn.execute(
        "SELECT trial_start_ts, trial_end_ts, messages_used FROM users WHERE user_id=?",
        (user_id,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {"trial_start_ts": row[0], "trial_end_ts": row[1], "messages_used": row[2]}

def check_and_count(user_id: str, now_ts: int, free_days: int, msg_cap: int, subscribe_url: str):
    ensure_user(user_id, now_ts, free_days)
    usage = get_usage(user_id)
    if not usage:
        inc_usage(user_id, 1); return True, ""
    if now_ts > usage["trial_end_ts"]:
        return False, f"Your free trial has ended. 💳 Subscribe: {subscribe_url}"
    if usage["messages_used"] >= msg_cap:
        return False, f"You’ve used all free-trial messages. 💳 Subscribe: {subscribe_url}"
    inc_usage(user_id, 1)
    return True, ""

# ---------- RAG ----------
def _load_documents():
    docs = []
    for pdf in DATA_DIR.glob("*.pdf"):
        try:
            docs.extend(PyPDFLoader(str(pdf)).load())
        except Exception as e:
            print(f"[RAG] Failed to load {pdf}: {e}")
    return docs

def _split_docs(docs):
    return RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150).split_documents(docs)

def build_or_load_vectorstore():
    if not OPENAI_API_KEY:
        print("[RAG] OPENAI_API_KEY missing — vectorstore disabled.")
        return None
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    index_path = VSTORE_DIR / "faiss_index"
    if (index_path.with_suffix(".faiss").exists() and index_path.with_suffix(".pkl").exists()):
        return FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
    docs = _load_documents()
    if not docs:
        print("[RAG] No PDFs in data/docs — skipping index build.")
        return None
    chunks = _split_docs(docs)
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(str(index_path))
    print("[RAG] Built FAISS index with", len(chunks), "chunks")
    return vs

def retrieve_context(vs, query: str, k: int = 4) -> str:
    if vs is None: return ""
    docs = vs.similarity_search_with_score(query, k=k)
    return "\n\n".join([doc.page_content for doc, _ in docs])[:6000]

# ---------- OPENAI HELPERS ----------
SYSTEM_PROMPT = (  """
You are SiteMind AI ⚡️ — an intelligent tutor for UK electrical installation learners and working electricians.

Do:
- Use UK terms/standards (BS 7671, cable sizes, protective devices).
- Explain clearly and briefly; structure with steps or bullets.
- For maths: show formula → substitution → answer.
- Emphasise safe working and compliance.
- Ask a clarifying question if the query is vague.
- Keep replies ≤150 words unless needed.
- Remind users that AI can make mistakes and should be verified.

Tone: confident, instructive, practical.
""")

def chat_reply(user_text: str, history: List[tuple], rag_context: str) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if rag_context:
        messages.append({
            "role": "system",
            "content": "Use the following course/context excerpts if relevant:\n\n" + rag_context
        })
    for role, content in history:
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_text})
    out = client.chat.completions.create(model=OPENAI_MODEL, messages=messages, temperature=0.3)
    return out.choices[0].message.content.strip()

def vision_answer(image_url: str, prompt_text: str) -> str:
    out = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":[
                {"type":"text","text":prompt_text},
                {"type":"image_url","image_url":image_url},
            ]}
        ]
    )
    return out.choices[0].message.content.strip()

def transcribe_and_answer(audio_bytes: bytes) -> str:
    Path("tmp_media").mkdir(exist_ok=True)
    p = Path("tmp_media/voice.ogg")
    p.write_bytes(audio_bytes)
    with p.open("rb") as f:
        t = client.audio.transcriptions.create(model="whisper-1", file=f)
    text = t.text.strip()
    return chat_reply(text, [], "")
