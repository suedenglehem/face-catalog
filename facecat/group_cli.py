from __future__ import annotations

import argparse
import queue
import threading
from collections import defaultdict

import numpy as np

from . import config, db
from .gpu_knn import GpuMatMulKNN, adaptive_chunk


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


class CpuMatMulKNN:
    """CPU fallback with the same interface as GpuMatMulKNN."""

    def __init__(self, b_matrix: np.ndarray) -> None:
        self.b = np.ascontiguousarray(b_matrix, dtype=np.float32)
        self.n_total = int(self.b.shape[0])

    def topk(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(queries, dtype=np.float32)
        sims = q @ self.b.T
        kk = min(k, self.n_total)
        part = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
        vals = np.take_along_axis(sims, part, axis=1)
        return part, vals


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


def _parse_gpus(spec: str | None) -> list[int]:
    if not spec:
        return list(config.GPUS)
    idxs = [int(x) for x in spec.split(",") if x.strip()]
    return idxs or list(config.GPUS)


def rebuild_groups(gpus: list[int] | None = None, threads_per_gpu: int | None = None) -> dict:
    gpus = _parse_gpus(None if gpus is None else ",".join(str(g) for g in gpus))
    if threads_per_gpu is None:
        threads_per_gpu = config.THREADS_PER_GPU
    threads_per_gpu = max(1, int(threads_per_gpu))

    job_id = db.start_job("group_rebuild")

    with db.connect() as conn:
        faces = load_faces(conn)

    n = len(faces)
    base_stats = {
        "faces_considered": n,
        "gpus": gpus,
        "threads_per_gpu": threads_per_gpu,
    }

    if n == 0:
        db.clear_group_edges(job_id)
        with db.connect() as conn:
            clear_groups(conn)
            conn.commit()
        db.update_job_stats(job_id, base_stats)
        db.finish_job(job_id, "done")
        return {**base_stats, "edges_kept": 0, "groups_created": 0, "members_assigned": 0}

    ids = [f["id"] for f in faces]
    id_to_idx = {fid: i for i, fid in enumerate(ids)}
    M = np.stack([f["embedding"] for f in faces]).astype(np.float32)  # (n, 512)

    # Resume detection: a reclaimed job may already have persisted edges.
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute("select stats from jobs where id = %s", (job_id,))
            prev_stats = dict(cur.fetchone()["stats"] or {})
    resumed = (
        int(prev_stats.get("faces_considered", -1)) == n
        and len(prev_stats.get("done_blocks", [])) > 0
    )
    if not resumed:
        db.clear_group_edges(job_id)

    done_blocks: set[int] = set(int(b) for b in prev_stats.get("done_blocks", []))
    progress_lock = threading.Lock()

    block_size = adaptive_chunk(n, n)
    blocks = [(s, min(s + block_size, n)) for s in range(0, n, block_size)]
    pending_idx = [i for i in range(len(blocks)) if i not in done_blocks]

    print(
        f"[rebuild] faces={n} gpus={gpus or ['cpu']} threads_per_gpu={threads_per_gpu} "
        f"blocks={len(blocks)} block_size={block_size} resumed={'yes' if resumed else 'no'}"
    )

    # One session per GPU; its T worker threads share it (thread-safe run()).
    sessions: dict[int, object] = {}
    if gpus:
        for g in gpus:
            print(f"[rebuild] creating kNN session on GPU {g} ...")
            sessions[g] = GpuMatMulKNN(g, M)
    else:
        sessions[0] = CpuMatMulKNN(M)

    db_conn = db.connect()
    db_lock = threading.Lock()
    retried: set[int] = set()

    def worker(gpu_key: int, sess: object) -> None:
        while True:
            try:
                block_idx = work_q.get_nowait()
            except queue.Empty:
                return
            try:
                persist_block_with(sess, block_idx)
            except Exception as exc:
                print(f"[rebuild][gpu{gpu_key}] block {block_idx} failed: {exc!r}")
                with db_lock:
                    try:
                        db_conn.rollback()
                    except Exception:
                        pass
                # Re-queue once so the block is not lost; if it fails again it
                # stays pending and a re-run of the command will continue.
                with progress_lock:
                    if block_idx in retried:
                        print(f"[rebuild] giving up on block {block_idx}; re-run to continue")
                    else:
                        retried.add(block_idx)
                        work_q.put(block_idx)

    def persist_block_with(sess: object, block_idx: int) -> None:
        s, e = blocks[block_idx]
        idxs, vals = sess.topk(M[s:e], config.GROUP_K)
        triples: list[tuple[int, int, float]] = []
        for r in range(e - s):
            q_idx = s + r
            for j in range(idxs.shape[1]):
                nb = int(idxs[r, j])
                if nb == q_idx:
                    continue
                sim = float(vals[r, j])
                if sim >= config.GROUP_THRESHOLD:
                    triples.append((ids[q_idx], ids[nb], sim))

        with db_lock:
            db.insert_group_edges(db_conn, job_id, triples)

        with progress_lock:
            done_blocks.add(block_idx)
            total_done = len(done_blocks)
            if total_done % 10 == 0 or total_done == len(blocks):
                print(f"[rebuild] blocks done: {total_done}/{len(blocks)}")
                try:
                    db.update_job_stats(
                        job_id,
                        {**base_stats, "done_blocks": sorted(done_blocks)},
                    )
                except Exception:
                    pass

    work_q: queue.Queue = queue.Queue()
    for i in pending_idx:
        work_q.put(i)

    threads = []
    for g, sess in sessions.items():
        for _ in range(threads_per_gpu):
            t = threading.Thread(target=worker, args=(g, sess), daemon=True)
            threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=7200)
    db_conn.close()

    if len(done_blocks) < len(blocks):
        raise RuntimeError(
            f"rebuild incomplete ({len(done_blocks)}/{len(blocks)} blocks); re-run to continue"
        )

    # Union-find over all persisted edges (covers resumed + newly computed).
    uf = UnionFind(list(range(n)))
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "select face_id, nbr_face_id from face_group_edges where job_id = %s",
                (job_id,),
            )
            edge_rows = cur.fetchall()

    edges_kept = 0
    for row in edge_rows:
        a = id_to_idx[int(row["face_id"])]
        b = id_to_idx[int(row["nbr_face_id"])]
        if a != b:
            uf.union(a, b)
            edges_kept += 1

    buckets: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        buckets[uf.find(i)].append(ids[i])

    groups = [sorted(members) for members in buckets.values()]
    groups.sort(key=len, reverse=True)

    face_map = {f["id"]: f for f in faces}
    with db.connect() as conn:
        clear_groups(conn)
        store_stats = store_groups(conn, groups, face_map)
        conn.commit()

    stats = {**base_stats, "edges_kept": edges_kept, **store_stats}
    db.update_job_stats(job_id, stats)
    db.finish_job(job_id, "done")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(prog="facecat.group_cli")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rebuild = sub.add_parser(
        "rebuild",
        help="Recompute face groups. Re-running after a crash continues where it stopped.",
    )
    p_rebuild.add_argument(
        "--threads-per-gpu", type=int, default=config.THREADS_PER_GPU,
        help=f"worker threads per GPU sharing one kNN session (default {config.THREADS_PER_GPU})",
    )
    p_rebuild.add_argument("--gpus", default=None, help="comma-separated GPU ids, e.g. 0,1 (default: auto-detect)")

    args = parser.parse_args()

    if args.cmd == "rebuild":
        gpus = _parse_gpus(args.gpus)
        stats = rebuild_groups(gpus=gpus, threads_per_gpu=args.threads_per_gpu)
        print(stats)


if __name__ == "__main__":
    main()