import os
import time
import sqlite3
from pathlib import Path
from typing import Any, Sequence

try:
    import psycopg  # type: ignore
    from psycopg.rows import dict_row  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psycopg = None  # type: ignore
    dict_row = None  # type: ignore

DB_URL = os.getenv("DATABASE_URL", "").strip()
DB_PATH = Path(os.getenv("DB_PATH", "sitemind.db")).expanduser().resolve()
DB_IS_POSTGRES = DB_URL.startswith("postgres")

QMARK = "%s" if DB_IS_POSTGRES else "?"


def _fmt(sql: str) -> str:
    """
    Convert SQLite-style '?' placeholders to Postgres '%s' if needed.
    """
    return sql.replace("?", "%s") if DB_IS_POSTGRES else sql


def get_connection():
    """
    Return a database connection using either SQLite (default) or Postgres (DATABASE_URL).
    """
    if DB_IS_POSTGRES:
        if psycopg is None:
            raise RuntimeError("DATABASE_URL set but psycopg is not installed")
        return psycopg.connect(DB_URL, autocommit=True)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def execute(sql: str, params: Sequence[Any] = ()) -> None:
    conn = get_connection()
    try:
        cur = conn.cursor(row_factory=dict_row) if DB_IS_POSTGRES else conn.cursor()
        cur.execute(_fmt(sql), params)
        if not DB_IS_POSTGRES:
            conn.commit()
    finally:
        conn.close()


def fetchone(sql: str, params: Sequence[Any] = ()) -> Optional[Any]:
    conn = get_connection()
    try:
        cur = conn.cursor(row_factory=dict_row) if DB_IS_POSTGRES else conn.cursor()
        cur.execute(_fmt(sql), params)
        return cur.fetchone()
    finally:
        conn.close()


def fetchall(sql: str, params: Sequence[Any] = ()) -> list[Any]:
    conn = get_connection()
    try:
        cur = conn.cursor(row_factory=dict_row) if DB_IS_POSTGRES else conn.cursor()
        cur.execute(_fmt(sql), params)
        rows = cur.fetchall()
        return list(rows)
    finally:
        conn.close()


def _add_column(cur, table: str, col_def: str):
    try:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")
    except Exception:
        # Column likely exists already
        pass


def ensure_schema() -> None:
    """
    Create core tables and add new columns idempotently.
    """
    conn = get_connection()
    try:
        cur = conn.cursor(row_factory=dict_row) if DB_IS_POSTGRES else conn.cursor()

        message_pk = "BIGSERIAL PRIMARY KEY" if DB_IS_POSTGRES else "INTEGER PRIMARY KEY AUTOINCREMENT"

        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS users (
              user_id TEXT PRIMARY KEY,
              created_ts BIGINT NOT NULL,
              trial_start_ts BIGINT NOT NULL,
              trial_end_ts BIGINT NOT NULL,
              messages_used INTEGER NOT NULL DEFAULT 0,
              messages_today INTEGER NOT NULL DEFAULT 0,
              last_count_date TEXT,
              is_subscribed BOOLEAN NOT NULL DEFAULT FALSE,
              is_trial BOOLEAN NOT NULL DEFAULT TRUE,
              plan_tier TEXT DEFAULT 'core',
              stripe_customer_id TEXT,
              stripe_subscription_id TEXT,
              study_images_today INTEGER NOT NULL DEFAULT 0,
              heavy_images_today INTEGER NOT NULL DEFAULT 0,
              image_count_date TEXT
            );
            """
        )

        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS messages (
              id {message_pk},
              user_id TEXT NOT NULL,
              role TEXT NOT NULL,
              content TEXT NOT NULL,
              ts BIGINT NOT NULL
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS stripe_events (
              event_id TEXT PRIMARY KEY,
              created_ts BIGINT NOT NULL
            );
            """
        )

        # Backfill any missing columns for existing SQLite databases
        for col_def in (
            "plan_tier TEXT DEFAULT 'core'",
            "stripe_customer_id TEXT",
            "stripe_subscription_id TEXT",
            "study_images_today INTEGER NOT NULL DEFAULT 0",
            "heavy_images_today INTEGER NOT NULL DEFAULT 0",
            "image_count_date TEXT",
        ):
            _add_column(cur, "users", col_def)

        cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_user_ts ON messages(user_id, ts);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_users_trial_end ON users(trial_end_ts);")

        if not DB_IS_POSTGRES:
            conn.commit()
    finally:
        conn.close()


def now_ts() -> int:
    return int(time.time())
