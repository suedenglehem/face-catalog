from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from . import config, db, imaging
from .vision import FaceEngine


def utc_from_ts(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def slug(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:24]


def iter_images(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and imaging.is_supported_image(path):
            yield path


def ensure_root(path: Path) -> int:
    return db.add_root(str(path.resolve()))


def upsert_file(
    conn,
    root_id: int,
    root_path: Path,
    abs_path: Path,
    exif: dict,
    width: int,
    height: int,
    file_thumb_rel: str | None,
    seen_at: datetime,
) -> tuple[int, bool]:
    stat = abs_path.stat()
    mtime = utc_from_ts(stat.st_mtime)
    rel_path = str(abs_path.resolve().relative_to(root_path.resolve()))
    abs_path_str = str(abs_path.resolve())

    with conn.cursor() as cur:
        cur.execute(
            "select id, file_size, mtime from files where abs_path = %s",
            (abs_path_str,),
        )
        old = cur.fetchone()

        changed = (
            old is None
            or int(old["file_size"]) != int(stat.st_size)
            or old["mtime"] != mtime
        )

        cur.execute(
            """
            insert into files(
                root_id, rel_path, abs_path, file_size, mtime,
                width, height, exif, file_thumb_rel, indexed_at,
                last_seen_at, is_deleted
            )
            values (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, now(), %s, false)
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
                last_seen_at = excluded.last_seen_at,
                is_deleted = false
            returning id
            """,
            (
                root_id,
                rel_path,
                abs_path_str,
                stat.st_size,
                mtime,
                width,
                height,
                json.dumps(exif),
                file_thumb_rel,
                seen_at,
            ),
        )
        row = cur.fetchone()

    return int(row["id"]), changed


def replace_faces_for_file(
    conn,
    file_id: int,
    root_slug: str,
    file_slug: str,
    rgb,
    detections: list[dict],
    inv_scale: float,
) -> int:
    h, w = rgb.shape[:2]

    with conn.cursor() as cur:
        cur.execute("delete from faces where file_id = %s", (file_id,))

        inserted = 0

        for i, det in enumerate(detections):
            box_small = det["bbox"]
            box = [int(round(v * inv_scale)) for v in box_small]
            box = imaging.clamp_box(box, w, h)

            crop = imaging.crop_face(rgb, box)
            if crop.size == 0:
                continue

            face_thumb_rel = f"faces/{root_slug}/{file_slug}_{i}.jpg"
            imaging.save_jpeg(
                crop,
                config.THUMBS_DIR / face_thumb_rel,
                size=(256, 256),
            )

            cur.execute(
                """
                insert into faces(
                    file_id, face_index,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    det_score, quality_score,
                    embedding, face_thumb_rel
                )
                values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    file_id,
                    i,
                    box[0],
                    box[1],
                    box[2],
                    box[3],
                    float(det["det_score"]),
                    float(det["quality_score"]),
                    det["embedding"].tolist(),
                    face_thumb_rel,
                ),
            )
            inserted += 1

    return inserted


def process_file(conn, engine: FaceEngine, root_path: Path, abs_path: Path, seen_at: datetime) -> dict:
    rgb = imaging.load_rgb(abs_path)
    exif = imaging.extract_exif(abs_path)
    h, w = rgb.shape[:2]

    root_slug = slug(str(root_path.resolve()))
    file_slug = slug(str(abs_path.resolve()))
    file_thumb_rel = f"files/{root_slug}/{file_slug}.jpg"

    imaging.save_jpeg(rgb, config.THUMBS_DIR / file_thumb_rel, size=(768, 768))

    root_id = ensure_root(root_path)
    file_id, changed = upsert_file(
        conn=conn,
        root_id=root_id,
        root_path=root_path,
        abs_path=abs_path,
        exif=exif,
        width=w,
        height=h,
        file_thumb_rel=file_thumb_rel,
        seen_at=seen_at,
    )

    if not changed:
        return {
            "files_seen": 1,
            "files_reindexed": 0,
            "faces_found": 0,
        }

    resized, inv_scale = imaging.resize_for_detector(rgb, config.MAX_DETECT_SIDE)
    detections = engine.detect(resized)
    faces_found = replace_faces_for_file(
        conn=conn,
        file_id=file_id,
        root_slug=root_slug,
        file_slug=file_slug,
        rgb=rgb,
        detections=detections,
        inv_scale=inv_scale,
    )

    return {
        "files_seen": 1,
        "files_reindexed": 1,
        "faces_found": faces_found,
    }


def mark_missing_deleted(conn, root_id: int, seen_at: datetime) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            update files
            set is_deleted = true
            where root_id = %s
              and last_seen_at < %s
              and is_deleted = false
            returning id
            """,
            (root_id, seen_at),
        )
        rows = cur.fetchall()
    return len(rows)


def touch_root_indexed(conn, root_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "update roots set last_indexed_at = now() where id = %s",
            (root_id,),
        )


def index_one_root(root_path: Path) -> dict:
    root_path = root_path.resolve()
    if not root_path.exists() or not root_path.is_dir():
        raise ValueError(f"Not a directory: {root_path}")

    root_id = ensure_root(root_path)
    engine = FaceEngine()
    seen_at = datetime.now(timezone.utc)

    totals = {
        "root_id": root_id,
        "root_path": str(root_path),
        "files_seen": 0,
        "files_reindexed": 0,
        "faces_found": 0,
        "deleted_files": 0,
        "errors": 0,
    }

    with db.connect() as conn:
        for path in iter_images(root_path):
            try:
                stats = process_file(conn, engine, root_path, path, seen_at)
                totals["files_seen"] += stats["files_seen"]
                totals["files_reindexed"] += stats["files_reindexed"]
                totals["faces_found"] += stats["faces_found"]
                conn.commit()
            except Exception as exc:
                conn.rollback()
                totals["errors"] += 1
                print(f"ERROR {path}: {exc}")

        totals["deleted_files"] = mark_missing_deleted(conn, root_id, seen_at)
        touch_root_indexed(conn, root_id)
        conn.commit()

    return totals


def index_all_roots() -> list[dict]:
    results = []
    for row in db.list_roots():
        root_path = Path(row["path"])
        results.append(index_one_root(root_path))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add-root")
    p_add.add_argument("path")

    p_index_root = sub.add_parser("index-root")
    p_index_root.add_argument("path")

    sub.add_parser("index-all")

    args = parser.parse_args()

    if args.cmd == "add-root":
        root_id = ensure_root(Path(args.path))
        print(f"root registered: id={root_id}")

    elif args.cmd == "index-root":
        stats = index_one_root(Path(args.path))
        print(json.dumps(stats, indent=2, default=str))

    elif args.cmd == "index-all":
        rows = index_all_roots()
        print(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
