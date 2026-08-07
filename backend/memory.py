"""
Memory module.

- Short-term memory: recent conversation turns per student, used to give the
  orchestrator's planner context-aware conversation ability.
- Long-term memory: durable key/value facts about a student (e.g. stated
  interests, recurring preferences) that persist across sessions, stored in
  SQLite so they survive app restarts -- this is the "Long-term memory"
  stretch goal from the brief.
"""

import sqlite3
from contextlib import closing

from backend.config import MEMORY_DB_PATH


def init_db():
    with closing(sqlite3.connect(MEMORY_DB_PATH)) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT,
                role TEXT,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS long_term_facts (
                student_id TEXT,
                key TEXT,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (student_id, key)
            )"""
        )
        conn.commit()


def save_message(student_id: str, role: str, content: str):
    with closing(sqlite3.connect(MEMORY_DB_PATH)) as conn:
        conn.execute(
            "INSERT INTO conversations (student_id, role, content) VALUES (?, ?, ?)",
            (student_id, role, content),
        )
        conn.commit()


def get_recent_history(student_id: str, limit: int = 8) -> list[dict]:
    with closing(sqlite3.connect(MEMORY_DB_PATH)) as conn:
        rows = conn.execute(
            "SELECT role, content FROM conversations WHERE student_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (student_id, limit),
        ).fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]


def save_fact(student_id: str, key: str, value: str):
    with closing(sqlite3.connect(MEMORY_DB_PATH)) as conn:
        conn.execute(
            "INSERT INTO long_term_facts (student_id, key, value, updated_at) "
            "VALUES (?, ?, ?, CURRENT_TIMESTAMP) "
            "ON CONFLICT(student_id, key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP",
            (student_id, key, value),
        )
        conn.commit()


def get_facts(student_id: str) -> dict:
    with closing(sqlite3.connect(MEMORY_DB_PATH)) as conn:
        rows = conn.execute(
            "SELECT key, value FROM long_term_facts WHERE student_id = ?",
            (student_id,),
        ).fetchall()
    return {k: v for k, v in rows}
