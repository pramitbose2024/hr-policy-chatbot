"""
logger.py — SQLite logging for the HR chatbot.
Logs every query, answer, sources, and latency to a local database.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_PATH = "logs/chat_logs.db"


def init_db():
    """
    Creates the logs/ folder and database tables if they don't exist.
    Safe to call on every app startup — IF NOT EXISTS means it never
    overwrites existing data.
    """
    Path("logs").mkdir(exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Main query log — one row per user message
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            question    TEXT    NOT NULL,
            answer      TEXT    NOT NULL,
            sources     TEXT    NOT NULL,
            latency_ms  INTEGER NOT NULL,
            flagged     INTEGER NOT NULL DEFAULT 0
        )
    """)

    # Evaluation results — written by evaluate.py in Phase 4
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS eval_results (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_timestamp   TEXT    NOT NULL,
            question        TEXT    NOT NULL,
            expected_answer TEXT    NOT NULL,
            actual_answer   TEXT    NOT NULL,
            retrieval_pass  INTEGER NOT NULL,
            judge_score     INTEGER,
            judge_reasoning TEXT,
            failure_mode    TEXT
        )
    """)

    conn.commit()
    conn.close()


def log_query(question: str, answer: str, sources: list, latency_ms: int):
    """
    Writes one query+answer pair to the database.
    Called by app.py after every chatbot response.

    flagged=1 when the answer is a "couldn't find" response —
    these are your knowledge gaps, visible in the eval dashboard.
    """
    flagged = 1 if "couldn't find" in answer.lower() else 0

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO queries
            (timestamp, question, answer, sources, latency_ms, flagged)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        question,
        answer,
        json.dumps(sources),   # list → JSON string (SQLite has no array type)
        latency_ms,
        flagged,
    ))

    conn.commit()
    conn.close()


def log_eval_result(
    run_timestamp: str,
    question: str,
    expected_answer: str,
    actual_answer: str,
    retrieval_pass: bool,
    judge_score: int | None,
    judge_reasoning: str | None,
    failure_mode: str | None,
):
    """Written by evaluate.py during Phase 4 eval runs."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO eval_results (
            run_timestamp, question, expected_answer, actual_answer,
            retrieval_pass, judge_score, judge_reasoning, failure_mode
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        run_timestamp, question, expected_answer, actual_answer,
        int(retrieval_pass), judge_score, judge_reasoning, failure_mode,
    ))

    conn.commit()
    conn.close()


def get_all_queries() -> list[dict]:
    """Returns all logged queries, newest first."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM queries ORDER BY id DESC")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    for row in rows:
        row["sources"] = json.loads(row["sources"])
    return rows


def get_eval_runs() -> list[str]:
    """Returns distinct eval run timestamps, newest first."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT run_timestamp FROM eval_results
        ORDER BY run_timestamp DESC
    """)
    runs = [row[0] for row in cursor.fetchall()]
    conn.close()
    return runs


def get_eval_results_for_run(run_timestamp: str) -> list[dict]:
    """Returns all eval results for one run."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM eval_results WHERE run_timestamp = ?
        ORDER BY id
    """, (run_timestamp,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_summary_stats() -> dict:
    """Aggregate stats for the eval dashboard."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM queries")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM queries WHERE flagged = 1")
    flagged = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(latency_ms) FROM queries")
    avg_latency = cursor.fetchone()[0] or 0

    cursor.execute("""
        SELECT AVG(judge_score), AVG(retrieval_pass)
        FROM eval_results
        WHERE run_timestamp = (SELECT MAX(run_timestamp) FROM eval_results)
    """)
    row = cursor.fetchone()
    avg_judge     = row[0] or 0
    avg_retrieval = row[1] or 0

    conn.close()

    return {
        "total_queries":   total,
        "flagged_count":   flagged,
        "flagged_rate":    round(flagged / total * 100, 1) if total > 0 else 0,
        "avg_latency_ms":  round(avg_latency),
        "avg_judge_score": round(avg_judge, 2),
        "avg_retrieval":   round(avg_retrieval * 100, 1),
    }