from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import signal
import subprocess
import sys
import threading
import time
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

log = logging.getLogger("facecat.index")


def _setup_logging() -> Path:
    """Point the module logger at $LOGS_DIR/log-<firing time>.log plus stdout.

    Every index run gets its own file named after when it was fired, so a
    re-run after an interruption never mixes with the previous run's log.
    The directory comes from LOGS_DIR (setenv.sh), defaulting to ./logs.
    """
    logs_dir = config.LOGS_DIR
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / f"log-{datetime.now():%Y%m%d-%H%M%S}.log"
    log.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    return path


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
    # Same as db.connect(): leave the connection idle so per-file
    # conn.transaction() blocks are real commit boundaries, not savepoints.
    conn.commit()
    return conn


def _mtime_to_tz(epoch: float) -> datetime:
    """files.mtime is timestamptz; convert the filesystem epoch to an aware UTC dt."""
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


# ---------------------------------------------------------------------------
# Stage 1: scanner (fast DB pre-check before any decode)
# ---------------------------------------------------------------------------

def _walk_files(root_path: Path):
    """Yield every file under root_path, following directory symlinks.

    Photo archives are commonly assembled from symlinked mounts (e.g. a year
    folder pointing at an external drive). os.walk(followlinks=True) descends
    into those; a realpath set guards against symlink cycles so a loop can't
    spin forever - os.walk has no cycle protection of its own, and it also
    prevents double-processing when two paths reach the same physical dir.
    """
    visited: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root_path, followlinks=True):
        real_dir = os.path.realpath(dirpath)
        if real_dir in visited:
            # Reached this physical directory before (cycle or shared mount);
            # prune so we neither loop nor index its contents twice.
            dirnames[:] = []
            continue
        visited.add(real_dir)
        dirnames.sort()
        for name in sorted(filenames):
            yield Path(dirpath) / name


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

    for path in _walk_files(root_path):
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


def _decode_worker(
    decode_q: queue.Queue, gpu_q: queue.Queue, stats: Stats, stop_evt: threading.Event
) -> None:
    while True:
        if stop_evt.is_set():
            break  # Ctrl-C: leave the rest of the queue for the next run
        task = decode_q.get()
        try:
            if task is None:
                break

            rgb = imaging.load_rgb(task.abs_path)
            exif = imaging.extract_exif(task.abs_path)
            thumb_rel = _save_file_thumb(task)
            imaging.save_jpeg(rgb, config.THUMBS_DIR / thumb_rel, (320, 320))

            # Bounded put: on Ctrl-C the GPU workers may already have exited,
            # so an unbounded put() would block forever once gpu_q is full.
            while True:
                try:
                    gpu_q.put((task, rgb, exif, thumb_rel), timeout=1.0)
                    break
                except queue.Full:
                    if stop_evt.is_set():
                        return  # GPU side is gone; this file waits for the next run
        except Exception as exc:
            log.error("[decode] %s: %s", task.abs_path, exc)
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

    # All CPU/GPU work happens before the first DB write. The whole file then
    # lands in a single transaction (one commit), so an interrupted run can
    # never leave a half-written files/faces pair behind: either everything
    # below is committed or Postgres rolls it all back on disconnect.
    resized, scale = imaging.resize_for_detector(rgb, config.MAX_DETECT_SIDE)
    faces = engine.detect(resized)

    with conn.transaction():
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

    log.info("indexed %s faces=%d", task.abs_path, len(faces))
    stats.inc("files_reindexed")
    stats.inc("faces_found", len(faces))


def _gpu_worker(
    gpu_q: queue.Queue,
    gpu_id: int,
    worker_idx: int,
    stats: Stats,
    ready_evt: threading.Event | None = None,
    stop_evt: threading.Event | None = None,
) -> None:
    # Engine is created inside the worker thread so its CUDA context and
    # ONNX session are bound to this thread / device. In --threads-max mode
    # the launcher waits on ready_evt before starting the next worker, so
    # every VRAM check sees all previously started engines' memory already
    # allocated; signal it even when init fails so one dead engine cannot
    # stall the whole launch sequence.
    try:
        engine = FaceEngine(ctx_id=gpu_id)
        conn = _connect()
    except Exception as exc:
        log.error("[gpu%d/w%d] init failed: %s", gpu_id, worker_idx, exc)
        if ready_evt is not None:
            ready_evt.set()
        return
    if ready_evt is not None:
        ready_evt.set()

    while True:
        if stop_evt is not None and stop_evt.is_set():
            break  # Ctrl-C: leave the rest of the queue for the next run
        item = gpu_q.get()
        try:
            if item is None:
                break
            task, rgb, exif, thumb_rel = item
            _process_file(conn, engine, task, rgb, exif, thumb_rel, stats)
        except Exception as exc:
            log.error("[gpu%d/w%d] %s: %s", gpu_id, worker_idx, getattr(item[0], "abs_path", "?"), exc)
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


def _visible_to_physical(visible_id: int) -> int | None:
    """Map an ORT visible device id to the physical index nvidia-smi reports.

    Identity when CUDA_VISIBLE_DEVICES is unset; otherwise the entry at that
    position in CVD. UUID-form entries cannot be mapped by index and yield
    None, which callers treat as "VRAM unknown" (launch allowed).
    """
    raw = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return visible_id
    entries = [e.strip() for e in raw.split(",") if e.strip()]
    if visible_id >= len(entries):
        return None
    try:
        return int(entries[visible_id])
    except ValueError:
        return None


class _VramGuard:
    """Gates GPU worker launches on the free VRAM of each card.

    nvidia-smi speaks physical indices while ORT device ids are visible
    (position within CUDA_VISIBLE_DEVICES), so each requested id is resolved
    to a physical index once up front. Free memory is re-queried before every
    launch decision and never cached: the launcher waits for each worker's
    engine to be ready first, then asks "is there still room on this card?".
    """

    def __init__(self, gpus: list[int], reserve_mib: int) -> None:
        self.reserve_mib = reserve_mib
        self._physical = {g: _visible_to_physical(g) for g in gpus}

    def free_mib(self, gpu_id: int) -> int | None:
        """Current free VRAM (MiB) on a card; None when it cannot be read."""
        phys = self._physical.get(gpu_id)
        if phys is None:
            return None
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout
        except Exception as exc:
            print(f"[vram] nvidia-smi unavailable ({exc}); assuming GPU {gpu_id} has room")
            return None
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == phys:
                try:
                    return int(parts[1])
                except ValueError:
                    return None
        return None


def _launch_gpu_workers_vram(
    gpu_q: queue.Queue,
    gpus: list[int],
    threads_max: int,
    stats: Stats,
    stop_evt: threading.Event,
    threads: list[threading.Thread],
) -> None:
    """Start up to `threads_max` GPU workers one at a time, VRAM-gated.

    Cards are visited round-robin; before each launch the card's free VRAM is
    re-read and if it has dropped below RESERVE_VRAM no more workers go there.
    Each worker's engine init is awaited (ready_evt) before the next decision,
    so every check sees all previously started engines' memory already
    allocated. Note the last worker on a card may still push free VRAM below
    the reserve -- the guard only gates *further* launches.

    Launched threads are appended to `threads` (owned by run_pipeline) as they
    start, so an interruption mid-launch can still join every started worker.
    """
    guard = _VramGuard(gpus, int(config.RESERVE_VRAM_GB * 1024))
    full_cards: set[int] = set()
    launched = 0

    while launched < threads_max and len(full_cards) < len(gpus):
        for g in gpus:
            if launched >= threads_max or g in full_cards:
                continue
            free = guard.free_mib(g)
            if free is not None and free < guard.reserve_mib:
                print(
                    f"[vram] GPU {g}: {free} MiB free < reserve "
                    f"{guard.reserve_mib} MiB -- no more workers on this card"
                )
                full_cards.add(g)
            else:
                ready = threading.Event()
                t = threading.Thread(
                    target=_gpu_worker, args=(gpu_q, g, launched, stats, ready, stop_evt), daemon=True
                )
                t.start()
                threads.append(t)
                if not ready.wait(timeout=600):
                    print(f"[vram] GPU {g}: worker init took >600s; continuing without waiting")
                launched += 1

    if not threads:
        raise RuntimeError(
            "no GPU workers could be started -- every card is below the VRAM reserve"
        )
    print(f"[init] vram-aware launch: {launched} GPU worker(s) started (max {threads_max})")


def run_pipeline(
    roots: list[dict],
    gpus: list[int],
    threads_per_gpu: int,
    cpu_workers: int,
    fresh: bool,
    batch_size: int | None = None,
    threads_max: int | None = None,
) -> Stats:
    log_path = _setup_logging()
    print(f"[init] logging to {log_path}")

    stats = Stats()
    job_id = db.start_job("index_all")

    decode_q: queue.Queue = queue.Queue(maxsize=config.INDEX_DECODE_QUEUE)
    gpu_q: queue.Queue = queue.Queue(maxsize=config.INDEX_GPU_QUEUE)

    budget = _BatchBudget(batch_size)

    # Warm the model cache once in the main thread so worker threads do not
    # race on first-time model download, then release its GPU memory.
    print(f"[init] warming face engine on GPU {gpus[0]} ...")
    _warm = FaceEngine(ctx_id=gpus[0])
    del _warm
    batch_note = f" batch_size={batch_size}" if batch_size is not None else ""
    if threads_max is not None:
        mode_note = (
            f"threads_max={threads_max} (vram-aware, reserve "
            f"{config.RESERVE_VRAM_GB:g} GB/card)"
        )
    else:
        mode_note = f"threads_per_gpu={threads_per_gpu} gpu_workers={len(gpus) * threads_per_gpu}"
    print(f"[init] gpus={gpus} {mode_note} cpu_decode_workers={cpu_workers}{batch_note}")
    log.info(
        "index started: job=%d gpus=%s %s cpu_decode_workers=%d%s",
        job_id, gpus, mode_note, cpu_workers, batch_note,
    )

    stop_evt = threading.Event()
    flusher = _StatsFlusher(job_id, stats, stop_evt)
    flusher.start()

    decode_threads = [
        threading.Thread(target=_decode_worker, args=(decode_q, gpu_q, stats, stop_evt), daemon=True)
        for _ in range(cpu_workers)
    ]
    for t in decode_threads:
        t.start()

    # Ctrl-C handling: the first press raises KeyboardInterrupt (caught below
    # for an orderly stop); a second press SIGKILLs the whole process group so
    # nothing can survive - including workers wedged inside C-level CUDA calls.
    sigint_presses = 0
    prev_sigint = signal.getsignal(signal.SIGINT)

    def _on_sigint(signum, frame):
        nonlocal sigint_presses
        sigint_presses += 1
        if sigint_presses == 1:
            print("\n[ctrl-c] stopping after in-flight files finish (Ctrl-C again to force-kill)...", flush=True)
            raise KeyboardInterrupt
        os.killpg(os.getpgrp(), signal.SIGKILL)

    signal.signal(signal.SIGINT, _on_sigint)

    gpu_threads: list[threading.Thread] = []
    try:
        if threads_max is not None:
            # VRAM-aware mode: workers are started one at a time (already running
            # when this returns); the launcher re-checks free VRAM before each.
            _launch_gpu_workers_vram(gpu_q, gpus, threads_max, stats, stop_evt, gpu_threads)
        else:
            for g in gpus:
                for w in range(threads_per_gpu):
                    t = threading.Thread(
                        target=_gpu_worker, args=(gpu_q, g, w, stats, None, stop_evt), daemon=True
                    )
                    t.start()
                    gpu_threads.append(t)

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
            done_msg = (
                f"root {root_id}: seen={stats.files_seen} "
                f"skipped_unchanged={stats.files_skipped} reindexed={stats.files_reindexed} "
                f"faces={stats.faces_found} errors={stats.errors}"
            )
            print(f"[done] {done_msg}")
            log.info("scan done: %s", done_msg)

        # Shut down worker pools.
        for _ in decode_threads:
            decode_q.put(None)
        for _ in gpu_threads:
            gpu_q.put(None)
        for t in decode_threads + gpu_threads:
            t.join(timeout=600)

        db.finish_job(job_id, "done")
        log.info("index finished: %s", stats.snapshot())
    except KeyboardInterrupt:
        # Orderly stop: workers finish the file they are on (per-file commits
        # keep the DB consistent either way), then exit at their next loop
        # check. Sentinels wake any worker blocked in get(); put_nowait so a
        # full queue can't deadlock the shutdown itself.
        stop_evt.set()
        for q, n in ((decode_q, len(decode_threads)), (gpu_q, len(gpu_threads))):
            for _ in range(n):
                try:
                    q.put_nowait(None)
                except queue.Full:
                    pass
        deadline = time.monotonic() + 60
        for t in decode_threads + gpu_threads:
            t.join(timeout=max(0.0, deadline - time.monotonic()))
        stuck = [t for t in decode_threads + gpu_threads if t.is_alive()]
        try:
            db.finish_job(job_id, "interrupted", error="Ctrl-C")
        except Exception:
            pass
        log.info("index interrupted by Ctrl-C: %s", stats.snapshot())
        print(f"[ctrl-c] stopped - re-run the same command to continue (log: {log_path})", flush=True)
        if stuck:
            # A worker wedged in a C-level call won't see stop_evt; nuke the
            # whole process group (this python plus any children) so nothing lingers.
            print(f"[ctrl-c] {len(stuck)} worker(s) still running after 60s - force-killing process group", flush=True)
            os.killpg(os.getpgrp(), signal.SIGKILL)
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
        signal.signal(signal.SIGINT, prev_sigint)

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _fmt_ts(dt: datetime | None) -> str:
    """Render an aware DB timestamp in local time, or 'never' when absent."""
    if dt is None:
        return "never"
    return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _print_stats() -> None:
    s = db.catalog_stats()
    print("catalog statistics")
    print(f"  photos present : {s['photos_present']}")
    print(f"  photos deleted : {s['photos_deleted']}")
    print(f"  faces          : {s['faces']}")
    print(f"  face groups    : {s['groups']}")
    print("last activity")
    print(f"  file indexed   : {_fmt_ts(s['last_file_indexed'])}")
    print(f"  root indexed   : {_fmt_ts(s['last_root_indexed'])}")
    print(f"  last scan      : {_fmt_ts(s['last_scan'])}")

    jobs = db.get_latest_jobs(5)
    if not jobs:
        return
    print("recent jobs")
    for j in jobs:
        st = j["stats"] or {}
        detail = (
            f"reindexed={st.get('files_reindexed', 0)} "
            f"faces={st.get('faces_found', 0)} errors={st.get('errors', 0)}"
        )
        print(
            f"  #{j['id']} {j['job_type']:<12} {j['status']:<8} "
            f"{_fmt_ts(j['started_at'])} -> {_fmt_ts(j['finished_at'])}  {detail}"
        )


def _add_pipeline_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--threads-per-gpu", type=int, default=None, metavar="N",
        help=f"GPU worker threads per GPU (default {config.THREADS_PER_GPU}; "
             f"ignored when --threads-max is given)",
    )
    p.add_argument(
        "--threads-max", type=int, default=None, metavar="N",
        help="start at most N GPU workers total, one at a time across cards "
             "(round-robin), re-checking free VRAM before each launch and "
             f"stopping on any card below RESERVE_VRAM ({config.RESERVE_VRAM_GB:g} GB); "
             "mutually exclusive with --threads-per-gpu",
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
    sub.add_parser(
        "stats",
        help="Dump catalog counts and last-update times (photos, faces, groups, recent jobs).",
    )

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
    elif args.cmd == "stats":
        _print_stats()
    elif args.cmd in ("index-all", "index-root"):
        gpus = _parse_gpus(args.gpus)
        if args.threads_per_gpu is not None and args.threads_max is not None:
            parser.error("use either --threads-per-gpu or --threads-max, not both")
        threads_max = args.threads_max
        if threads_max is not None and threads_max < 1:
            parser.error("--threads-max must be >= 1")
        if threads_max is not None:
            threads_per_gpu = config.THREADS_PER_GPU  # unused in vram-aware mode
        else:
            threads_per_gpu = (
                max(1, int(args.threads_per_gpu))
                if args.threads_per_gpu is not None
                else config.THREADS_PER_GPU
            )
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
        stats = run_pipeline(
            roots, gpus, threads_per_gpu, cpu_workers, bool(args.fresh), batch_size, threads_max
        )
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
