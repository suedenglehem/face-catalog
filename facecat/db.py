from __future__ import annotations

import argparse
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from . import config


def connect():
    conn = psycopg.connect(config.DATABASE_URL, row_factory=dict_row)
    register_vector(conn)
    return conn


def run_schema_file() -> None:
    schema_path = Path(__file__).with_name("schema.sql")
    sql_text = schema_path.read_text(encoding="utf-8")

    statements = [s.strip() for s in sql_text.split(";") if s.strip()]

    with connect() as conn:
        with conn.cursor() as cur:
            for stmt in statements:
                cur.execute(stmt)
        conn.commit()


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
