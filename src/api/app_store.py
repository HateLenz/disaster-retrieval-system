from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_APP_DB = REPO_ROOT / "db" / "retrieval_app.db"
ROLES = ("retriever", "admin", "developer")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    if not salt:
        salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        120_000,
    )
    return salt, digest.hex()


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


class LocalAppStore:
    def __init__(self, db_path: str | Path = DEFAULT_APP_DB) -> None:
        self.db_path = Path(db_path)

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    role TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_login_at TEXT,
                    is_active INTEGER NOT NULL DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS search_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    query_time TEXT NOT NULL,
                    query_kind TEXT NOT NULL,
                    query_label TEXT NOT NULL,
                    disaster_type TEXT,
                    disaster TEXT,
                    filter_mode TEXT NOT NULL,
                    aggregation TEXT NOT NULL,
                    top_k INTEGER NOT NULL,
                    query_json TEXT NOT NULL,
                    gallery_json TEXT NOT NULL,
                    elapsed_ms REAL NOT NULL,
                    result_count INTEGER NOT NULL,
                    top_score REAL,
                    scope TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                );

                CREATE TABLE IF NOT EXISTS search_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    search_id INTEGER NOT NULL,
                    rank INTEGER NOT NULL,
                    positive_id TEXT,
                    tile_id TEXT,
                    score REAL,
                    best_patch_score REAL,
                    disaster TEXT,
                    disaster_type TEXT,
                    damage_label TEXT,
                    pre_patch_path TEXT,
                    pre_patch_url TEXT,
                    paired_post_patch_path TEXT,
                    paired_post_patch_url TEXT,
                    result_json TEXT NOT NULL,
                    FOREIGN KEY(search_id) REFERENCES search_records(id)
                );

                CREATE TABLE IF NOT EXISTS feedback_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    search_result_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    feedback TEXT NOT NULL,
                    note TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(search_result_id) REFERENCES search_results(id),
                    FOREIGN KEY(user_id) REFERENCES users(id)
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
                CREATE INDEX IF NOT EXISTS idx_search_records_user ON search_records(user_id);
                CREATE INDEX IF NOT EXISTS idx_search_results_search ON search_results(search_id);
                CREATE INDEX IF NOT EXISTS idx_feedback_result ON feedback_records(search_result_id);
                """
            )
            connection.commit()
        self.ensure_user_columns()
        self.seed_demo_users()

    def ensure_user_columns(self) -> None:
        with self.connect() as connection:
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(users)").fetchall()
            }
            if "is_active" not in columns:
                connection.execute("ALTER TABLE users ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1")
            connection.commit()

    def seed_demo_users(self) -> None:
        self.migrate_removed_roles()
        demo_users = [
            ("admin", "admin123", "admin", "系统管理员"),
            ("developer", "developer123", "developer", "算法开发人员"),
            ("retriever", "retriever123", "retriever", "检索用户"),
        ]
        for username, password, role, display_name in demo_users:
            if not self.user_exists(username):
                self.create_user(username, password, role, display_name)

    def migrate_removed_roles(self) -> None:
        placeholders = ", ".join("?" for _ in ROLES)
        with self.connect() as connection:
            connection.execute("UPDATE users SET role = 'retriever' WHERE role = 'analyst'")
            connection.execute(
                f"UPDATE users SET role = 'retriever' WHERE role NOT IN ({placeholders})",
                ROLES,
            )
            connection.commit()

    def user_exists(self, username: str) -> bool:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM users WHERE username = ?",
                (username.strip(),),
            ).fetchone()
            return row is not None

    def create_user(self, username: str, password: str, role: str, display_name: str | None = None) -> dict[str, Any]:
        username = username.strip()
        if not username:
            raise ValueError("Username is required.")
        if len(password) < 6:
            raise ValueError("Password must contain at least 6 characters.")
        if role not in ROLES:
            raise ValueError(f"Role must be one of: {', '.join(ROLES)}")
        display_name = (display_name or username).strip() or username
        salt, digest = hash_password(password)
        with self.connect() as connection:
            try:
                cursor = connection.execute(
                    """
                    INSERT INTO users(username, password_hash, salt, role, display_name, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (username, digest, salt, role, display_name, utc_now()),
                )
                connection.commit()
            except sqlite3.IntegrityError as error:
                raise ValueError("Username already exists.") from error
            return self.get_user_by_id(int(cursor.lastrowid))

    def authenticate(self, username: str, password: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE username = ?",
                (username.strip(),),
            ).fetchone()
            if row is None:
                raise ValueError("Invalid username or password.")
            if int(row["is_active"]) != 1:
                raise ValueError("Invalid username or password.")
            _, expected = hash_password(password, row["salt"])
            if not hmac.compare_digest(expected, row["password_hash"]):
                raise ValueError("Invalid username or password.")
            connection.execute(
                "UPDATE users SET last_login_at = ? WHERE id = ?",
                (utc_now(), row["id"]),
            )
            connection.commit()
            return self._public_user(row_to_dict(row))

    def create_session(self, user_id: int, hours: int = 12) -> str:
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now(timezone.utc) + timedelta(hours=hours)
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO sessions(token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (token, user_id, utc_now(), expires_at.replace(microsecond=0).isoformat()),
            )
            connection.commit()
        return token

    def revoke_session(self, token: str) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM sessions WHERE token = ?", (token,))
            connection.commit()

    def get_user_by_token(self, token: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT users.*
                FROM sessions
                JOIN users ON users.id = sessions.user_id
                WHERE sessions.token = ? AND sessions.expires_at > ?
                """,
                (token, utc_now()),
            ).fetchone()
            return self._public_user(row_to_dict(row))

    def get_user_by_id(self, user_id: int) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            if row is None:
                raise ValueError("User not found.")
            return self._public_user(row_to_dict(row))

    def list_users(self, viewer: dict[str, Any] | None = None, query: str | None = None) -> list[dict[str, Any]]:
        where = ""
        params: list[Any] = []
        filters = ["is_active = 1"]
        if viewer and viewer.get("role") == "admin":
            filters.append("role != ?")
            params.append("developer")
        if query and query.strip():
            filters.append("LOWER(username) LIKE LOWER(?)")
            params.append(f"%{query.strip()}%")
        if filters:
            where = "WHERE " + " AND ".join(filters)
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT id, username, role, created_at, last_login_at, is_active
                FROM users
                {where}
                ORDER BY id
                """,
                params,
            ).fetchall()
            return [dict(row) for row in rows]

    def update_user(
        self,
        user_id: int,
        username: str | None = None,
        role: str | None = None,
        display_name: str | None = None,
        password: str | None = None,
        is_active: bool | None = None,
    ) -> dict[str, Any]:
        if role is not None and role not in ROLES:
            raise ValueError(f"Role must be one of: {', '.join(ROLES)}")
        if password is not None and len(password) < 6:
            raise ValueError("Password must contain at least 6 characters.")
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            if row is None:
                raise ValueError("User not found.")

            updates: list[str] = []
            params: list[Any] = []
            if username is not None:
                clean_username = username.strip()
                if not clean_username:
                    raise ValueError("Username is required.")
                updates.append("username = ?")
                params.append(clean_username)
            if role is not None and role != row["role"]:
                if row["role"] == "admin" and role != "admin":
                    remaining_admins = connection.execute(
                        "SELECT COUNT(*) AS count FROM users WHERE role = 'admin' AND id != ?",
                        (user_id,),
                    ).fetchone()["count"]
                    if remaining_admins == 0:
                        raise ValueError("At least one administrator account is required.")
                updates.append("role = ?")
                params.append(role)
            if display_name is not None:
                clean_name = display_name.strip()
                if not clean_name:
                    raise ValueError("Display name is required.")
                updates.append("display_name = ?")
                params.append(clean_name)
            if password is not None:
                salt, digest = hash_password(password)
                updates.extend(["salt = ?", "password_hash = ?"])
                params.extend([salt, digest])
            if is_active is not None and int(is_active) != int(row["is_active"]):
                if row["role"] == "admin" and not is_active:
                    remaining_admins = connection.execute(
                        "SELECT COUNT(*) AS count FROM users WHERE role = 'admin' AND id != ? AND is_active = 1",
                        (user_id,),
                    ).fetchone()["count"]
                    if remaining_admins == 0:
                        raise ValueError("At least one active administrator account is required.")
                updates.append("is_active = ?")
                params.append(1 if is_active else 0)

            if updates:
                params.append(user_id)
                try:
                    connection.execute(
                        f"UPDATE users SET {', '.join(updates)} WHERE id = ?",
                        params,
                    )
                except sqlite3.IntegrityError as error:
                    raise ValueError("Username already exists.") from error
                connection.commit()
        return self.get_user_by_id(user_id)

    def deactivate_user(self, user_id: int) -> dict[str, Any]:
        return self.update_user(user_id, is_active=False)

    def create_search_record(
        self,
        user_id: int,
        query_kind: str,
        query_label: str,
        disaster_type: str | None,
        disaster: str | None,
        filter_mode: str,
        aggregation: str,
        top_k: int,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        results = payload.get("results", [])
        top_score = float(results[0]["score"]) if results else None
        query = payload.get("query", {})
        gallery = payload.get("gallery", {})
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO search_records(
                    user_id, query_time, query_kind, query_label, disaster_type, disaster,
                    filter_mode, aggregation, top_k, query_json, gallery_json,
                    elapsed_ms, result_count, top_score, scope
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    utc_now(),
                    query_kind,
                    query_label,
                    disaster_type,
                    disaster,
                    filter_mode,
                    aggregation,
                    top_k,
                    json_dumps(query),
                    json_dumps(gallery),
                    float(payload.get("elapsed_ms", 0.0)),
                    len(results),
                    top_score,
                    gallery.get("scope"),
                ),
            )
            search_id = int(cursor.lastrowid)
            enriched_results = []
            for item in results:
                result_cursor = connection.execute(
                    """
                    INSERT INTO search_results(
                        search_id, rank, positive_id, tile_id, score, best_patch_score,
                        disaster, disaster_type, damage_label, pre_patch_path, pre_patch_url,
                        paired_post_patch_path, paired_post_patch_url, result_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        search_id,
                        int(item.get("rank", 0)),
                        item.get("positive_id"),
                        item.get("tile_id"),
                        float(item.get("score", 0.0)),
                        float(item.get("best_patch_score", 0.0)),
                        item.get("disaster"),
                        item.get("disaster_type"),
                        item.get("damage_label"),
                        item.get("pre_patch_path"),
                        item.get("pre_patch_url"),
                        item.get("paired_post_patch_path"),
                        item.get("paired_post_patch_url"),
                        json_dumps(item),
                    ),
                )
                enriched = dict(item)
                enriched["search_result_id"] = int(result_cursor.lastrowid)
                enriched_results.append(enriched)
            connection.commit()

        payload = dict(payload)
        payload["search_id"] = search_id
        payload["results"] = enriched_results
        return payload

    def list_history(self, user: dict[str, Any], limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 200))
        where = ""
        params: list[Any] = []
        if user["role"] == "retriever":
            where = "WHERE search_records.user_id = ?"
            params.append(user["id"])
        params.append(limit)
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT search_records.*, users.username, users.display_name
                FROM search_records
                JOIN users ON users.id = search_records.user_id
                {where}
                ORDER BY search_records.id DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
            return [self._history_summary(row) for row in rows]

    def get_history_detail(self, search_id: int, user: dict[str, Any]) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT search_records.*, users.username, users.display_name
                FROM search_records
                JOIN users ON users.id = search_records.user_id
                WHERE search_records.id = ?
                """,
                (search_id,),
            ).fetchone()
            if row is None:
                raise ValueError("Search record not found.")
            record = dict(row)
            self._assert_record_access(record, user)
            result_rows = connection.execute(
                """
                SELECT search_results.*,
                       feedback_records.feedback AS feedback,
                       feedback_records.note AS feedback_note
                FROM search_results
                LEFT JOIN feedback_records ON feedback_records.search_result_id = search_results.id
                WHERE search_results.search_id = ?
                ORDER BY search_results.rank
                """,
                (search_id,),
            ).fetchall()
            results = []
            for result_row in result_rows:
                result = json_loads(result_row["result_json"], {})
                result["search_result_id"] = result_row["id"]
                result["feedback"] = result_row["feedback"]
                result["feedback_note"] = result_row["feedback_note"]
                results.append(result)
            return {
                **self._history_summary(row),
                "query": json_loads(record.get("query_json"), {}),
                "gallery": json_loads(record.get("gallery_json"), {}),
                "results": results,
            }

    def get_result_detail(self, result_id: int, user: dict[str, Any]) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT search_results.*, search_records.user_id, search_records.query_json,
                       search_records.query_label, search_records.query_time,
                       feedback_records.feedback, feedback_records.note AS feedback_note
                FROM search_results
                JOIN search_records ON search_records.id = search_results.search_id
                LEFT JOIN feedback_records ON feedback_records.search_result_id = search_results.id
                WHERE search_results.id = ?
                """,
                (result_id,),
            ).fetchone()
            if row is None:
                raise ValueError("Search result not found.")
            record = dict(row)
            self._assert_record_access(record, user)
            result = json_loads(record["result_json"], {})
            result["search_result_id"] = record["id"]
            result["search_id"] = record["search_id"]
            result["query"] = json_loads(record["query_json"], {})
            result["query_label"] = record["query_label"]
            result["query_time"] = record["query_time"]
            result["feedback"] = record["feedback"]
            result["feedback_note"] = record["feedback_note"]
            return result

    def add_feedback(
        self,
        search_result_id: int,
        user_id: int,
        feedback: str,
        note: str | None,
        replace_result_feedback: bool = False,
    ) -> dict[str, Any]:
        if feedback not in {"correct", "wrong", "uncertain"}:
            raise ValueError("Feedback must be correct, wrong, or uncertain.")
        with self.connect() as connection:
            if replace_result_feedback:
                connection.execute("DELETE FROM feedback_records WHERE search_result_id = ?", (search_result_id,))
            existing = connection.execute(
                "SELECT id FROM feedback_records WHERE search_result_id = ? AND user_id = ?",
                (search_result_id, user_id),
            ).fetchone()
            if existing:
                connection.execute(
                    """
                    UPDATE feedback_records
                    SET feedback = ?, note = ?, created_at = ?
                    WHERE id = ?
                    """,
                    (feedback, note, utc_now(), existing["id"]),
                )
                feedback_id = int(existing["id"])
            else:
                cursor = connection.execute(
                    """
                    INSERT INTO feedback_records(search_result_id, user_id, feedback, note, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (search_result_id, user_id, feedback, note, utc_now()),
                )
                feedback_id = int(cursor.lastrowid)
            connection.commit()
        return {"id": feedback_id, "search_result_id": search_result_id, "feedback": feedback, "note": note}

    def list_failures(self, user: dict[str, Any], score_threshold: float = 0.25, limit: int = 100) -> list[dict[str, Any]]:
        where = ["(search_records.result_count = 0 OR search_records.top_score < ?)"]
        params: list[Any] = [score_threshold]
        if user["role"] == "retriever":
            where.append("search_records.user_id = ?")
            params.append(user["id"])
        params.append(max(1, min(limit, 200)))
        with self.connect() as connection:
            low_score_rows = connection.execute(
                f"""
                SELECT search_records.*, users.username, users.display_name
                FROM search_records
                JOIN users ON users.id = search_records.user_id
                WHERE {" AND ".join(where)}
                ORDER BY search_records.id DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
            wrong_rows = connection.execute(
                """
                SELECT search_records.*, users.username, users.display_name,
                       search_results.id AS failure_result_id,
                       feedback_records.feedback AS failure_feedback,
                       feedback_records.note AS failure_note
                FROM feedback_records
                JOIN search_results ON search_results.id = feedback_records.search_result_id
                JOIN search_records ON search_records.id = search_results.search_id
                JOIN users ON users.id = search_records.user_id
                WHERE feedback_records.feedback = 'wrong'
                ORDER BY feedback_records.id DESC
                LIMIT ?
                """,
                (max(1, min(limit, 200)),),
            ).fetchall()
        merged: dict[int, dict[str, Any]] = {}
        for row in low_score_rows:
            record = self._history_summary(row)
            if user["role"] == "retriever" and record["user_id"] != user["id"]:
                continue
            record["failure_reason"] = "low_score" if record["result_count"] else "no_result"
            record["search_result_id"] = self._first_result_id(record["id"])
            record["feedback"] = None
            record["feedback_note"] = None
            merged[record["id"]] = record
        for row in wrong_rows:
            record = self._history_summary(row)
            if user["role"] == "retriever" and record["user_id"] != user["id"]:
                continue
            record["failure_reason"] = "wrong_feedback"
            record["search_result_id"] = row["failure_result_id"]
            record["feedback"] = row["failure_feedback"]
            record["feedback_note"] = row["failure_note"]
            merged[record["id"]] = record
        return list(merged.values())[:limit]

    def _first_result_id(self, search_id: int) -> int | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT id FROM search_results WHERE search_id = ? ORDER BY rank LIMIT 1",
                (search_id,),
            ).fetchone()
            return int(row["id"]) if row else None

    def history_export_rows(self, user: dict[str, Any], ids: list[int] | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        if ids:
            rows = [self.get_history_detail(search_id, user) for search_id in ids[:500]]
        else:
            rows = self.list_history(user, limit=limit)
        return [self._export_history_row(row) for row in rows]

    def failure_export_rows(
        self,
        user: dict[str, Any],
        ids: list[int] | None = None,
        score_threshold: float = 0.35,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        failures = self.list_failures(user, score_threshold=score_threshold, limit=limit)
        if ids:
            selected = set(ids)
            failures = [row for row in failures if int(row["id"]) in selected]
        return [self._export_failure_row(row) for row in failures]

    def _export_history_row(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "search_id": row.get("id"),
            "query_time": row.get("query_time"),
            "username": row.get("username"),
            "query_kind": row.get("query_kind"),
            "query_label": row.get("query_label"),
            "disaster_type": row.get("disaster_type"),
            "disaster": row.get("disaster"),
            "filter_mode": row.get("filter_mode"),
            "aggregation": row.get("aggregation"),
            "top_k": row.get("top_k"),
            "elapsed_ms": row.get("elapsed_ms"),
            "result_count": row.get("result_count"),
            "top_score": row.get("top_score"),
            "scope": row.get("scope"),
        }

    def _export_failure_row(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            **self._export_history_row(row),
            "failure_reason": row.get("failure_reason"),
            "search_result_id": row.get("search_result_id"),
            "feedback": row.get("feedback"),
            "feedback_note": row.get("feedback_note"),
        }

    def algorithm_analysis(self, score_threshold: float = 0.35, limit: int = 10) -> dict[str, Any]:
        score_threshold = max(0.0, min(float(score_threshold), 1.0))
        limit = max(1, min(int(limit), 50))
        with self.connect() as connection:
            summary = dict(
                connection.execute(
                    """
                    SELECT
                        COUNT(*) AS total_searches,
                        AVG(elapsed_ms) AS avg_elapsed_ms,
                        AVG(top_score) AS avg_top_score,
                        SUM(CASE WHEN result_count = 0 THEN 1 ELSE 0 END) AS zero_result_count,
                        SUM(CASE WHEN top_score IS NOT NULL AND top_score < ? THEN 1 ELSE 0 END) AS low_score_count
                    FROM search_records
                    """,
                    (score_threshold,),
                ).fetchone()
            )
            disaster_rows = connection.execute(
                """
                SELECT
                    COALESCE(disaster_type, '-') AS disaster_type,
                    COUNT(*) AS searches,
                    AVG(top_score) AS avg_top_score,
                    SUM(CASE WHEN result_count = 0 THEN 1 ELSE 0 END) AS zero_result_count,
                    SUM(CASE WHEN top_score IS NOT NULL AND top_score < ? THEN 1 ELSE 0 END) AS low_score_count
                FROM search_records
                GROUP BY COALESCE(disaster_type, '-')
                ORDER BY searches DESC
                """,
                (score_threshold,),
            ).fetchall()
            strategy_rows = connection.execute(
                """
                SELECT
                    filter_mode,
                    aggregation,
                    COUNT(*) AS searches,
                    AVG(top_score) AS avg_top_score,
                    AVG(elapsed_ms) AS avg_elapsed_ms
                FROM search_records
                GROUP BY filter_mode, aggregation
                ORDER BY searches DESC
                """
            ).fetchall()
            feedback_rows = connection.execute(
                """
                SELECT feedback, COUNT(*) AS count
                FROM feedback_records
                GROUP BY feedback
                ORDER BY feedback
                """
            ).fetchall()

        feedback_counts = {"correct": 0, "wrong": 0, "uncertain": 0}
        feedback_counts.update({row["feedback"]: row["count"] for row in feedback_rows})
        return {
            "score_threshold": score_threshold,
            "summary": {
                "total_searches": int(summary.get("total_searches") or 0),
                "avg_elapsed_ms": summary.get("avg_elapsed_ms"),
                "avg_top_score": summary.get("avg_top_score"),
                "zero_result_count": int(summary.get("zero_result_count") or 0),
                "low_score_count": int(summary.get("low_score_count") or 0),
            },
            "feedback_counts": feedback_counts,
            "by_disaster_type": [dict(row) for row in disaster_rows],
            "by_strategy": [dict(row) for row in strategy_rows],
            "recent_failures": self.list_failures(
                {"id": 0, "role": "developer"},
                score_threshold=score_threshold,
                limit=limit,
            ),
        }

    def markdown_report(self, search_id: int, user: dict[str, Any]) -> str:
        detail = self.get_history_detail(search_id, user)
        lines = [
            "# 灾后检索报告",
            "",
            f"- 检索编号：{detail['id']}",
            f"- 检索时间：{detail['query_time']}",
            f"- 用户：{detail['username']}",
            f"- 查询图像：{detail['query_label']}",
            f"- 灾害类型：{detail.get('disaster_type') or '-'}",
            f"- 灾害地区：{detail.get('disaster') or '-'}",
            f"- 过滤模式：{detail['filter_mode']}",
            f"- 聚合策略：{detail['aggregation']}",
            f"- 检索耗时：{detail['elapsed_ms']} ms",
            f"- 索引范围：{detail.get('scope') or '-'}",
            "",
            "## Top 结果",
            "",
            "| Rank | Score | Tile ID | Disaster | Damage | Pre Patch |",
            "| --- | ---: | --- | --- | --- | --- |",
        ]
        for item in detail["results"]:
            lines.append(
                "| {rank} | {score} | {tile_id} | {disaster} | {damage_label} | {pre_patch_path} |".format(
                    rank=item.get("rank"),
                    score=item.get("score"),
                    tile_id=item.get("tile_id"),
                    disaster=item.get("disaster"),
                    damage_label=item.get("damage_label"),
                    pre_patch_path=item.get("pre_patch_path"),
                )
            )
        lines.append("")
        return "\n".join(lines)

    def _history_summary(self, row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        record = dict(row)
        return {
            "id": record["id"],
            "user_id": record["user_id"],
            "username": record.get("username"),
            "display_name": record.get("display_name"),
            "query_time": record["query_time"],
            "query_kind": record["query_kind"],
            "query_label": record["query_label"],
            "disaster_type": record["disaster_type"],
            "disaster": record["disaster"],
            "filter_mode": record["filter_mode"],
            "aggregation": record["aggregation"],
            "top_k": record["top_k"],
            "elapsed_ms": record["elapsed_ms"],
            "result_count": record["result_count"],
            "top_score": record["top_score"],
            "scope": record["scope"],
        }

    def _assert_record_access(self, record: dict[str, Any], user: dict[str, Any]) -> None:
        if user["role"] == "retriever" and int(record["user_id"]) != int(user["id"]):
            raise PermissionError("This record belongs to another user.")

    def _public_user(self, user: dict[str, Any] | None) -> dict[str, Any] | None:
        if user is None:
            return None
        return {
            "id": user["id"],
            "username": user["username"],
            "role": user["role"],
            "created_at": user["created_at"],
            "last_login_at": user.get("last_login_at"),
            "is_active": bool(user.get("is_active", 1)),
        }


__all__ = ["DEFAULT_APP_DB", "LocalAppStore", "ROLES"]
