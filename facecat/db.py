from __future__ import annotations

import argparse
import json
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from . import config


def ensure_vector(conn) -> None:
    """Create the pgvector extension on this database if it is missing.

    Makes every connection self-healing: a freshly created (or reset)
    database has no 'vector' type yet, which would otherwise make
    register_vector() raise ProgrammingError before any schema exists.
    """
    with conn.cursor() as cur:
        cur.execute("select 1 from pg_type where typname = 'vector'")
        if cur.fetchone() is None:
            cur.execute("create extension if not exists vector")
    conn.commit()


def connect():
    conn = psycopg.connect(config.DATABASE_URL, row_factory=dict_row)
    ensure_vector(conn)
    register_vector(conn)
    return conn


def _split_sql_statements(sql_text: str) -> list[str]:
    # Strip line comments so semicolons inside comments do not split statements.
    lines = []
    for line in sql_text.splitlines():
        idx = line.find("--")
        if idx >= 0:
            line = line[:idx]
        lines.append(line)
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


def run_schema_file() -> None:
    schema_path = Path(__file__).with_name("schema.sql")
    sql_text = schema_path.read_text(encoding="utf-8")

    statements = _split_sql_statements(sql_text)

    with connect() as conn:
        with conn.cursor() as cur:
            for stmt in statements:
                cur.execute(stmt)
        conn.commit()


# ---------------------------------------------------------------------------
# Job tracking (crash-resumable index-all / group rebuild)
# ---------------------------------------------------------------------------

def start_job(job_type: str, stats: dict | None = None) -> int:
    """Start a job; reclaims an interrupted ('running') job of the same type."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select id from jobs
                where job_type = %s and status = 'running'
                order by id desc limit 1
                """,
                (job_type,),
            )
            row = cur.fetchone()
            if row is not None:
                job_id = int(row["id"])
                cur.execute(
                    "update jobs set started_at = now(), finished_at = null, error = null, stats = %s::jsonb where id = %s",
                    (json.dumps(stats or {}), job_id),
                )
            else:
                cur.execute(
                    "insert into jobs(job_type, status, stats) values (%s, 'running', %s::jsonb) returning id",
                    (job_type, json.dumps(stats or {})),
                )
                job_id = int(cur.fetchone()["id"])
        conn.commit()
    return job_id


def update_job_stats(job_id: int, stats: dict) -> None:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "update jobs set stats = %s::jsonb where id = %s",
                (json.dumps(stats), job_id),
            )
        conn.commit()


def finish_job(job_id: int, status: str = "done", error: str | None = None) -> None:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "update jobs set status = %s, finished_at = now(), error = %s where id = %s",
                (status, error, job_id),
            )
        conn.commit()


def get_latest_jobs(limit: int = 10) -> list[dict]:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("select * from jobs order by id desc limit %s", (limit,))
            rows = cur.fetchall()
    return rows


# ---------------------------------------------------------------------------
# Group rebuild edges (persisted per block for crash-resume)
# ---------------------------------------------------------------------------

def clear_group_edges(job_id: int) -> None:
    """Drop this job's old edges and any orphaned edges from dead jobs."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("delete from face_group_edges where job_id = %s", (job_id,))
            cur.execute(
                "delete from face_group_edges where job_id not in (select id from jobs where status = 'running')"
            )
        conn.commit()


def insert_group_edges(conn, job_id: int, triples: list[tuple[int, int, float]]) -> None:
    if not triples:
        return
    with conn.cursor() as cur:
        cur.executemany(
            "insert into face_group_edges(job_id, face_id, nbr_face_id, similarity) values (%s, %s, %s, %s)",
            [(job_id, a, b, s) for (a, b, s) in triples],
        )
    conn.commit()


def count_group_edges(job_id: int) -> int:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("select count(*) as n from face_group_edges where job_id = %s", (job_id,))
            return int(cur.fetchone()["n"])


# ---------------------------------------------------------------------------
# Admin helpers
# ---------------------------------------------------------------------------

def database_size_bytes() -> int:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("select pg_database_size(current_database()) as n")
            return int(cur.fetchone()["n"])


def add_root(path: str) -> int:
    p = str(Path(path).resolve())
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into roots(path, enabled)
                values (%s, true)
                on conflict (path) do update set enabled = true
                returning id
                """,
                (p,),
            )
            row = cur.fetchone()
        conn.commit()
    return int(row["id"])


def list_roots() -> list[dict]:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select id, path, enabled, created_at, last_indexed_at
                from roots
                where enabled = true
                order by id
                """
            )
            rows = cur.fetchall()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init")
    p_add = sub.add_parser("add-root")
    p_add.add_argument("path")

    args = parser.parse_args()

    if args.cmd == "init":
        run_schema_file()
        print("schema initialized")
    elif args.cmd == "add-root":
        root_id = add_root(args.path)
        print(f"root registered: id={root_id}")


if __name__ == "__main__":
    main()
