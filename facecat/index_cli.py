from __future__ import annotations

import argparse
import json
import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from . import config, db, imaging
from .vision import FaceEngine


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

@dataclass
class FileTask:
    root_id: int
    rel_path: str
    abs_path: Path
    size: int
    mtime: float


@dataclass
class Stats:
    files_seen: int = 0
    files_skipped: int = 0
    files_reindexed: int = 0
    files_deferred: int = 0
    faces_found: int = 0
    errors: int = 0
    roots_done: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def inc(self, name: str, n: int = 1) -> None:
        with self.lock:
            setattr(self, name, getattr(self, name) + n)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "files_seen": self.files_seen,
                "files_skipped_unchanged": self.files_skipped,
                "files_reindexed": self.files_reindexed,
                "files_deferred_batch_limit": self.files_deferred,
                "faces_found": self.faces_found,
                "errors": self.errors,
                "roots_done": self.roots_done,
            }


class _BatchBudget:
    """Caps how many files this run enqueues for indexing.

    The scan still walks every file (so last_seen_at touches and missing-file
    deletion stay correct); only the enqueue is limited. Deferred files are
    new/changed ones that simply wait for the next run, which skips already
    indexed files by size+mtime - so re-running continues where this stopped.
    """

    def __init__(self, limit: int | None) -> None:
        self.limit = limit
        self.remaining = limit

    @property
    def exhausted(self) -> bool:
        return self.limit is not None and self.remaining <= 0

    def take(self) -> bool:
        if self.exhausted:
            return False
        if self.limit is not None:
            self.remaining -= 1
        return True


def _connect() -> psycopg.Connection:
    conn = psycopg.connect(config.DATABASE_URL, row_factory=dict_row)
    db.ensure_vector(conn)
    db.ensure_schema(conn)
    register_vector(conn)
    return conn


def _mtime_to_tz(epoch: float) -> datetime:
    """files.mtime is timestamptz; convert the filesystem epoch to an aware UTC dt."""
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


# ---------------------------------------------------------------------------
# Stage 1: scanner (fast DB pre-check before any decode)
# ---------------------------------------------------------------------------

def _scan_root(
    root_id: int,
    root_path: Path,
    fresh: bool,
    stats: Stats,
    decode_q: queue.Queue,
    budget: _BatchBudget,
) -> None:
    """Walk one root; enqueue changed/new files for decoding.

    Unchanged files (same size + mtime in DB) are only touched so that an
    interrupted re-run skips them almost instantly - this is what makes job
    continuation cheap. With a batch budget, the walk still covers every file
    but enqueues only while the shared (cross-root) budget has room; the rest
    count as deferred and are picked up by the next run.
    """
    seen_paths: list[str] = []
    flushed = 0

    def flush_seen() -> None:
        # Touch last_seen_at in batches (progress / audit) but keep the full
        # list so mark_missing_deleted can delete by "not seen on disk".
        nonlocal flushed
        if len(seen_paths) <= flushed:
            return
        batch = seen_paths[flushed:]
        with db.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "update files set last_seen_at = now() where abs_path = any(%s)",
                    (batch,),
                )
            conn.commit()
        flushed = len(seen_paths)

    # One bulk lookup instead of one query per file.
    existing: dict[str, tuple[int, float]] = {}
    if not fresh:
        with db.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "select abs_path, file_size, mtime from files where is_deleted = false"
                )
                for r in cur.fetchall():
                    mt = r["mtime"]
                    epoch = mt.timestamp() if isinstance(mt, datetime) else float(mt)
                    existing[r["abs_path"]] = (int(r["file_size"]), epoch)

    for path in sorted(root_path.rglob("*")):
        if not path.is_file() or not imaging.is_supported_image(path):
            continue

        st = path.stat()
        stats.inc("files_seen")
        rel = str(path.relative_to(root_path))

        seen_paths.append(str(path))
        if len(seen_paths) >= 1000:
            flush_seen()

        if fresh:
            if budget.take():
                decode_q.put(FileTask(root_id, rel, path, st.st_size, st.st_mtime))
            else:
                stats.inc("files_deferred")
            continue

        row = existing.get(str(path))
        unchanged = (
            row is not None
            and row[0] == st.st_size
            and abs(row[1] - st.st_mtime) < 1e-3
        )
        if unchanged:
            stats.inc("files_skipped")
        elif budget.take():
            decode_q.put(FileTask(root_id, rel, path, st.st_size, st.st_mtime))
        else:
            stats.inc("files_deferred")

    # Touch last_seen_at for everything present on disk. The returned list is
    # the authoritative "seen this scan" set used to delete only files truly
    # gone from the filesystem - independent of wall-clock timing, which keeps
    # crash-resume safe: a file committed by an earlier interrupted run but
    # still on disk (and skipped here as unchanged) must not be marked deleted.
    flush_seen()
    return seen_paths


# ---------------------------------------------------------------------------
# Stage 2: CPU decode pool (RAW conversion, EXIF, file thumbnail)
# ---------------------------------------------------------------------------

def _save_file_thumb(task: FileTask) -> str:
    safe_rel = task.rel_path.replace("/", "_").replace("\\", "_")
    thumb_rel = f"files/{task.root_id}/{safe_rel}.jpg"
    return thumb_rel


def _decode_worker(decode_q: queue.Queue, gpu_q: queue.Queue, stats: Stats) -> None:
    while True:
        task = decode_q.get()
        try:
            if task is None:
                break

            rgb = imaging.load_rgb(task.abs_path)
            exif = imaging.extract_exif(task.abs_path)
            thumb_rel = _save_file_thumb(task)
            imaging.save_jpeg(rgb, config.THUMBS_DIR / thumb_rel, (320, 320))

            gpu_q.put((task, rgb, exif, thumb_rel))
        except Exception as exc:
            print(f"[decode] {task.abs_path}: {exc}")
            stats.inc("errors")
        finally:
            decode_q.task_done()


# ---------------------------------------------------------------------------
# Stage 3: GPU workers (T threads per GPU, each with its own engine + conn)
# ---------------------------------------------------------------------------

def _process_file(
    conn,
    engine: FaceEngine,
    task: FileTask,
    rgb: np.ndarray,
    exif: dict,
    thumb_rel: str,
    stats: Stats,
) -> None:
    h, w = rgb.shape[:2]

    with conn.cursor() as cur:
        cur.execute(
            """
            insert into files (
                root_id, rel_path, abs_path, file_size, mtime,
                width, height, exif, file_thumb_rel, indexed_at, last_seen_at, is_deleted
            ) values (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, now(), now(), false)
            on conflict (abs_path) do update set
                root_id = excluded.root_id,
                rel_path = excluded.rel_path,
                file_size = excluded.file_size,
                mtime = excluded.mtime,
                width = excluded.width,
                height = excluded.height,
                exif = excluded.exif,
                file_thumb_rel = excluded.file_thumb_rel,
                indexed_at = now(),
                last_seen_at = now(),
                is_deleted = false
            returning id
            """,
            (
                task.root_id,
                task.rel_path,
                str(task.abs_path),
                task.size,
                _mtime_to_tz(task.mtime),
                w,
                h,
                json.dumps(exif),
                thumb_rel,
            ),
        )
        file_id = int(cur.fetchone()["id"])

    with conn.cursor() as cur:
        cur.execute("delete from faces where file_id = %s", (file_id,))

    resized, scale = imaging.resize_for_detector(rgb, config.MAX_DETECT_SIDE)
    faces = engine.detect(resized)

    with conn.cursor() as cur:
        for i, face in enumerate(faces):
            box = [int(x * scale) for x in face["bbox"]]
            crop = imaging.crop_face(rgb, box)
            face_thumb_rel = f"faces/{file_id}_{i}.jpg"
            imaging.save_jpeg(crop, config.THUMBS_DIR / face_thumb_rel, (256, 256))

            cur.execute(
                """
                insert into faces (
                    file_id, face_index, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    det_score, quality_score, embedding, face_thumb_rel
                ) values (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s)
                """,
                (
                    file_id,
                    i,
                    box[0],
                    box[1],
                    box[2],
                    box[3],
                    face["det_score"],
                    face["quality_score"],
                    face["embedding"].tolist(),
                    face_thumb_rel,
                ),
            )

    conn.commit()
    stats.inc("files_reindexed")
    stats.inc("faces_found", len(faces))


def _gpu_worker(gpu_q: queue.Queue, gpu_id: int, worker_idx: int, stats: Stats) -> None:
    # Engine is created inside the worker thread so its CUDA context and
    # ONNX session are bound to this thread / device.
    engine = FaceEngine(ctx_id=gpu_id)
    conn = _connect()

    while True:
        item = gpu_q.get()
        try:
            if item is None:
                break
            task, rgb, exif, thumb_rel = item
            _process_file(conn, engine, task, rgb, exif, thumb_rel, stats)
        except Exception as exc:
            print(f"[gpu{gpu_id}/w{worker_idx}] {getattr(item[0], 'abs_path', '?')}: {exc}")
            try:
                conn.rollback()
            except Exception:
                pass
            stats.inc("errors")
        finally:
            gpu_q.task_done()

    conn.close()


# ---------------------------------------------------------------------------
# Root finalization
# ---------------------------------------------------------------------------

def mark_missing_deleted(root_id: int, seen_paths: list[str]) -> None:
    # Delete only rows for this root whose abs_path was NOT observed on disk
    # during this scan. `abs_path <> all(empty)` is vacuously true, so an
    # image-less scan correctly marks the whole root deleted. This avoids the
    # wall-clock race of a stale last_seen_at after a long decode/GPU drain.
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                update files set is_deleted = true
                where root_id = %s and is_deleted = false and abs_path <> all(%s)
                """,
                (root_id, seen_paths),
            )
        conn.commit()


def touch_root_indexed(root_id: int) -> None:
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute("update roots set last_indexed_at = now() where id = %s", (root_id,))
        conn.commit()


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------

class _StatsFlusher(threading.Thread):
    def __init__(self, job_id: int, stats: Stats, stop_evt: threading.Event) -> None:
        super().__init__(daemon=True)
        self.job_id = job_id
        self.stats = stats
        self.stop_evt = stop_evt

    def run(self) -> None:
        while not self.stop_evt.wait(10):
            try:
                db.update_job_stats(self.job_id, self.stats.snapshot())
            except Exception:
                pass


def _parse_gpus(spec: str | None) -> list[int]:
    if not spec:
        return list(config.GPUS)
    idxs = [int(x) for x in spec.split(",") if x.strip()]
    if not idxs:
        return list(config.GPUS)
    translated = config.translate_gpus(idxs)
    if not translated:
        raise ValueError(f"--gpus {spec!r}: every requested GPU is hidden by CUDA_VISIBLE_DEVICES")
    return translated


def run_pipeline(
    roots: list[dict],
    gpus: list[int],
    threads_per_gpu: int,
    cpu_workers: int,
    fresh: bool,
    batch_size: int | None = None,
) -> Stats:
    stats = Stats()
    job_id = db.start_job("index_all")

    decode_q: queue.Queue = queue.Queue(maxsize=config.INDEX_DECODE_QUEUE)
    gpu_q: queue.Queue = queue.Queue(maxsize=config.INDEX_GPU_QUEUE)

    n_gpu_workers = len(gpus) * threads_per_gpu
    budget = _BatchBudget(batch_size)

    # Warm the model cache once in the main thread so worker threads do not
    # race on first-time model download, then release its GPU memory.
    print(f"[init] warming face engine on GPU {gpus[0]} ...")
    _warm = FaceEngine(ctx_id=gpus[0])
    del _warm
    batch_note = f" batch_size={batch_size}" if batch_size is not None else ""
    print(
        f"[init] gpus={gpus} threads_per_gpu={threads_per_gpu} "
        f"gpu_workers={n_gpu_workers} cpu_decode_workers={cpu_workers}{batch_note}"
    )

    stop_evt = threading.Event()
    flusher = _StatsFlusher(job_id, stats, stop_evt)
    flusher.start()

    decode_threads = [
        threading.Thread(target=_decode_worker, args=(decode_q, gpu_q, stats), daemon=True)
        for _ in range(cpu_workers)
    ]
    gpu_threads = []
    for g in gpus:
        for w in range(threads_per_gpu):
            t = threading.Thread(target=_gpu_worker, args=(gpu_q, g, w, stats), daemon=True)
            gpu_threads.append(t)

    for t in decode_threads + gpu_threads:
        t.start()

    try:
        for root in roots:
            root_id = int(root["id"])
            root_path = Path(str(root["path"]))
            print(f"[scan] root {root_id}: {root_path}")

            # Scan inline (workers already running) so we capture the seen set;
            # bounded decode_q provides back-pressure if workers fall behind.
            seen_paths = _scan_root(root_id, root_path, fresh, stats, decode_q, budget)

            # All tasks for this root are in the pipeline; wait until both
            # stages have fully drained before finalizing the root.
            decode_q.join()
            gpu_q.join()

            mark_missing_deleted(root_id, seen_paths)
            touch_root_indexed(root_id)
            stats.inc("roots_done")
            print(
                f"[done] root {root_id}: seen={stats.files_seen} "
                f"skipped_unchanged={stats.files_skipped} reindexed={stats.files_reindexed} "
                f"faces={stats.faces_found} errors={stats.errors}"
            )

        # Shut down worker pools.
        for _ in decode_threads:
            decode_q.put(None)
        for _ in gpu_threads:
            gpu_q.put(None)
        for t in decode_threads + gpu_threads:
            t.join(timeout=600)

        db.finish_job(job_id, "done")
    except Exception as exc:
        # Leave the job row 'running' so a re-run reclaims it; per-file
        # commits already persisted are skipped on continuation.
        print(f"[error] {exc!r} - re-run the same command to continue.")
        try:
            db.update_job_stats(job_id, stats.snapshot())
        except Exception:
            pass
        raise
    finally:
        stop_evt.set()

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_pipeline_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--threads-per-gpu", type=int, default=config.THREADS_PER_GPU,
        help=f"GPU worker threads per GPU (default {config.THREADS_PER_GPU})",
    )
    p.add_argument("--gpus", default=None, help="comma-separated GPU ids, e.g. 0,1 (default: auto-detect)")
    p.add_argument(
        "--cpu-workers", type=int, default=config.INDEX_CPU_WORKERS,
        help=f"CPU decode threads for RAW/EXIF/thumbnails (default {config.INDEX_CPU_WORKERS})",
    )
    p.add_argument("--fresh", action="store_true", help="re-detect all files even if unchanged")
    p.add_argument(
        "--batch-size", type=int, default=None, metavar="N",
        help="index at most N new/changed files this run; re-run to continue with the rest "
             "(default: no limit)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="facecat.index_cli")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init")
    p_add = sub.add_parser("add-root")
    p_add.add_argument("path")
    sub.add_parser("list-roots")

    p_all = sub.add_parser(
        "index-all",
        help="Index every enabled root. Re-running after a crash continues where it stopped.",
    )
    _add_pipeline_args(p_all)

    p_root = sub.add_parser("index-root", help="Register (if needed) and index one root.")
    p_root.add_argument("path")
    _add_pipeline_args(p_root)

    p_reset = sub.add_parser(
        "reset",
        help="Clear all catalog data (files, faces, groups, jobs). Roots are kept by default.",
    )
    p_reset.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    p_reset.add_argument("--keep-thumbs", action="store_true", help="do not delete thumbnail files on disk")
    p_reset.add_argument("--clear-roots", action="store_true", help="also remove registered roots")

    args = parser.parse_args()

    if args.cmd == "init":
        db.run_schema_file()
        print("schema initialized")
    elif args.cmd == "add-root":
        root_id = db.add_root(args.path)
        print(f"root registered: id={root_id}")
    elif args.cmd == "list-roots":
        for row in db.list_roots():
            print(
                f"id={row['id']} path={row['path']} last_indexed_at={row.get('last_indexed_at')}"
            )
    elif args.cmd in ("index-all", "index-root"):
        gpus = _parse_gpus(args.gpus)
        threads_per_gpu = max(1, int(args.threads_per_gpu))
        cpu_workers = max(1, int(args.cpu_workers))

        if args.cmd == "index-all":
            roots = db.list_roots()
            if not roots:
                print("no enabled roots; run add-root first")
                return
        else:
            root_id = db.add_root(args.path)
            with db.connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("select id, path from roots where id = %s", (root_id,))
                    row = cur.fetchone()
            roots = [dict(row)]

        batch_size = args.batch_size
        if batch_size is not None and batch_size < 1:
            parser.error("--batch-size must be >= 1")
        stats = run_pipeline(roots, gpus, threads_per_gpu, cpu_workers, bool(args.fresh), batch_size)
        snap = stats.snapshot()
        print(f"finished: {snap}")
        if snap["errors"]:
            print(f"warning: {snap['errors']} file(s) failed (see log above)")
        if snap["files_deferred_batch_limit"]:
            print(
                f"batch limit reached: {snap['files_deferred_batch_limit']} file(s) "
                "left for the next run - re-run the same command to continue"
            )
        # Exit non-zero when nothing succeeded so callers/CI notice a fully
        # broken run (e.g. missing exiftool crashing every file).
        if snap["errors"] and not snap["files_reindexed"]:
            raise SystemExit(1)
    elif args.cmd == "reset":
        if not args.yes:
            answer = input("Clear all catalog data? [y/N] ").strip().lower()
            if answer not in {"y", "yes"}:
                print("aborted")
                return
        summary = db.reset_database(keep_thumbs=args.keep_thumbs, clear_roots=args.clear_roots)
        print(f"reset done: cleared={summary['cleared']} thumbs_removed={summary['thumbs_removed']}")
        if args.clear_roots:
            print("roots removed - re-add them with add-root before index-all")
        else:
            print("roots kept - run 'index-all' to rebuild the catalog")


if __name__ == "__main__":
    main()
