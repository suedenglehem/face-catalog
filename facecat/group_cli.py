from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from . import config, db


class UnionFind:
    def __init__(self, ids: list[int]) -> None:
        self.parent = {x: x for x in ids}
        self.rank = {x: 0 for x in ids}

    def find(self, x: int) -> int:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return

        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def load_faces(conn) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            select
                f.id,
                f.embedding,
                coalesce(f.quality_score, 0) as quality_score
            from faces f
            join files fi on fi.id = f.file_id
            where f.invalidated = false
              and fi.is_deleted = false
            order by f.id
            """
        )
        rows = cur.fetchall()

    out = []
    for row in rows:
        emb = row["embedding"]
        if not isinstance(emb, np.ndarray):
            emb = emb.to_numpy()  # pgvector Vector -> ndarray
        out.append(
            {
                "id": int(row["id"]),
                "embedding": np.asarray(emb, dtype=np.float32),
                "quality_score": float(row["quality_score"]),
            }
        )
    return out


def nearest_neighbors(conn, face_id: int, embedding, k: int) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            select
                f.id,
                1 - (f.embedding <=> %s::vector) as similarity
            from faces f
            join files fi on fi.id = f.file_id
            where f.id <> %s
              and f.invalidated = false
              and fi.is_deleted = false
            order by f.embedding <=> %s::vector
            limit %s
            """,
            (embedding.tolist(), face_id, embedding.tolist(), k),
        )
        rows = cur.fetchall()

    return [{"id": int(r["id"]), "similarity": float(r["similarity"])} for r in rows]


def clear_groups(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("delete from face_group_members")
        cur.execute("delete from face_groups")


def store_groups(conn, groups: list[list[int]], face_map: dict[int, dict]) -> dict:
    with conn.cursor() as cur:
        group_count = 0
        member_count = 0

        for members in groups:
            if len(members) < 2:
                continue

            rep_face_id = max(members, key=lambda fid: face_map[fid]["quality_score"])
            rep_emb = face_map[rep_face_id]["embedding"]

            cur.execute(
                """
                insert into face_groups(label, rep_face_id, member_count)
                values (null, %s, %s)
                returning id
                """,
                (rep_face_id, len(members)),
            )
            group_id = int(cur.fetchone()["id"])

            for face_id in members:
                sim = float(np.dot(face_map[face_id]["embedding"], rep_emb))
                cur.execute(
                    """
                    insert into face_group_members(group_id, face_id, score_to_rep)
                    values (%s, %s, %s)
                    """,
                    (group_id, face_id, sim),
                )
                member_count += 1

            group_count += 1

    return {"groups_created": group_count, "members_assigned": member_count}


def rebuild_groups() -> dict:
    with db.connect() as conn:
        faces = load_faces(conn)
        if not faces:
            clear_groups(conn)
            conn.commit()
            return {
                "faces_considered": 0,
                "edges_kept": 0,
                "groups_created": 0,
                "members_assigned": 0,
            }

        face_map = {row["id"]: row for row in faces}
        uf = UnionFind([row["id"] for row in faces])
        edges_kept = 0

        for row in faces:
            neighbors = nearest_neighbors(
                conn=conn,
                face_id=row["id"],
                embedding=row["embedding"],
                k=config.GROUP_K,
            )
            for nbr in neighbors:
                if nbr["similarity"] >= config.GROUP_THRESHOLD:
                    uf.union(row["id"], nbr["id"])
                    edges_kept += 1

        buckets: dict[int, list[int]] = defaultdict(list)
        for row in faces:
            buckets[uf.find(row["id"])].append(row["id"])

        groups = [sorted(ids) for ids in buckets.values()]
        groups.sort(key=len, reverse=True)

        clear_groups(conn)
        store_stats = store_groups(conn, groups, face_map)
        conn.commit()

    return {
        "faces_considered": len(faces),
        "edges_kept": edges_kept,
        **store_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("rebuild")

    args = parser.parse_args()

    if args.cmd == "rebuild":
        stats = rebuild_groups()
        print(stats)


if __name__ == "__main__":
    main()
