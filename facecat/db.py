from __future__ import annotations

import argparse
import json
import shutil
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


def _schema_sql() -> str:
    return Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")


def ensure_schema(conn) -> None:
    """Create the schema on this database if it is missing.

    Makes every connection self-healing, like ensure_vector(): a freshly
    created (or reset with down -v) database needs no manual init step
    before add-root / index-all. The check is one to_regclass() lookup;
    the full file only runs when the marker table is absent. schema.sql is
    fully idempotent (if not exists everywhere), so a first-run race between
    processes is harmless.
    """
    with conn.cursor() as cur:
        cur.execute("select to_regclass('public.roots') is not null as has_schema")
        if cur.fetchone()["has_schema"]:
            return
        # A single parameterless execute() uses the simple query protocol, so
        # Postgres itself parses comments, quoted strings and all statements -
        # no fragile client-side splitting (and no psycopg version dependency).
        cur.execute(_schema_sql())
    conn.commit()


def connect():
    conn = psycopg.connect(config.DATABASE_URL, row_factory=dict_row)
    ensure_vector(conn)
    ensure_schema(conn)
    register_vector(conn)
    # TypeInfo.fetch() inside register_vector() runs SELECTs and leaves an
    # implicit transaction open; commit so callers start idle - otherwise a
    # later conn.transaction() would only create a savepoint in that open
    # transaction instead of a real commit boundary.
    conn.commit()
    return conn


def run_schema_file() -> None:
    # connect() already self-heals; re-run the file explicitly so `init` also
    # applies schema changes to an existing database.
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(_schema_sql())
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


def catalog_stats() -> dict:
    """Aggregate counts and last-activity timestamps for the `stats` command.

    "Last update" is derived from columns already maintained on every index -
    files.indexed_at (set to now() each time a file is indexed) and
    roots.last_indexed_at - so no separate bookkeeping column can drift out of
    sync with what actually happened. Timestamps are aware datetimes or None.
    """
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select
                    count(*) filter (where is_deleted = false) as photos_present,
                    count(*) filter (where is_deleted = true)  as photos_deleted,
                    max(indexed_at)                            as last_file_indexed,
                    max(last_seen_at)                          as last_scan
                from files
                """
            )
            f = cur.fetchone()
            cur.execute("select count(*) as n from faces where invalidated = false")
            faces_n = int(cur.fetchone()["n"])
            cur.execute("select count(*) as n from face_groups")
            groups_n = int(cur.fetchone()["n"])
            cur.execute("select max(last_indexed_at) as t from roots")
            last_root = cur.fetchone()["t"]
    return {
        "photos_present": int(f["photos_present"]),
        "photos_deleted": int(f["photos_deleted"]),
        "faces": faces_n,
        "groups": groups_n,
        "last_file_indexed": f["last_file_indexed"],
        "last_scan": f["last_scan"],
        "last_root_indexed": last_root,
    }


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


def reset_database(keep_thumbs: bool = False, clear_roots: bool = False) -> dict:
    """Clear catalog data for a fresh start.

    By default roots are kept so you can immediately re-run index-all; pass
    clear_roots to remove them too. Thumbnails on disk are removed unless
    keep_thumbs is set (they would otherwise be orphaned). Returns a summary
    of row counts cleared and thumbnails deleted.
    """
    tables_to_truncate = [
        "face_group_members",
        "face_groups",
        "face_group_edges",
        "faces",
        "files",
        "jobs",
    ]
    if clear_roots:
        tables_to_truncate.append("roots")

    counts: dict[str, int] = {}
    with connect() as conn:
        with conn.cursor() as cur:
            for table in ("files", "faces", "face_groups", "jobs", "roots"):
                cur.execute(f"select count(*) as n from {table}")
                counts[table] = int(cur.fetchone()["n"])
            # Every FK target is either in this list or `roots` (kept unless
            # clear_roots), so no CASCADE is needed. restart identity gives
            # fresh 1-based ids after the reset.
            cur.execute("truncate " + ", ".join(tables_to_truncate) + " restart identity")
        conn.commit()

    thumbs_removed = 0
    if not keep_thumbs:
        for sub in ("files", "faces"):
            d = config.THUMBS_DIR / sub
            if d.is_dir():
                thumbs_removed += sum(1 for p in d.rglob("*") if p.is_file())
                shutil.rmtree(d)

    return {"cleared": counts, "thumbs_removed": thumbs_removed}


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
