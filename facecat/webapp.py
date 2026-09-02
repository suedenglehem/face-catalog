from __future__ import annotations

from html import escape
from io import BytesIO

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

from . import config, db, imaging
from .vision import FaceEngine


app = FastAPI(title="Face Catalog")
app.mount("/thumbs", StaticFiles(directory=str(config.THUMBS_DIR)), name="thumbs")

# Web search engine is pinned to the first GPU; indexing uses its own pool.
ENGINE = FaceEngine(ctx_id=config.GPUS[0])


def page(title: str, body: str) -> HTMLResponse:
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
    body {
    margin: 0;
    font-family: system-ui, sans-serif;
    background: #0f172a;
    color: #e5e7eb;
    }
    .wrap {
    max-width: 1100px;
    margin: 0 auto;
    padding: 24px;
    }
    .nav {
    margin-bottom: 20px;
    }
    .nav a {
    color: #93c5fd;
    text-decoration: none;
    margin-right: 16px;
    }
    .card {
    background: #111827;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 18px;
    }
    h1, h2, h3 {
    margin-top: 0;
    }
    .muted {
    color: #94a3b8;
    }
    .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 16px;
    }
    .result {
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 14px;
    background: #1f2937;
    }
    .thumb-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 12px;
    }
    img.thumb {
    max-width: 220px;
    max-height: 220px;
    border-radius: 10px;
    border: 1px solid #334155;
    background: #000;
    }
    code {
    background: #0b1220;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 2px 6px;
    }
    table {
    width: 100%;
    border-collapse: collapse;
    }
    th, td {
    padding: 10px 8px;
    border-bottom: 1px solid #334155;
    text-align: left;
    vertical-align: top;
    }
    input[type="file"] {
    margin-right: 12px;
    }
    button {
    padding: 10px 14px;
    border: 0;
    border-radius: 8px;
    background: #60a5fa;
    color: #08111f;
    font-weight: 700;
    cursor: pointer;
    }
    .btn-sm {
    padding: 6px 10px;
    font-size: 13px;
    background: #f87171;
    margin-top: 8px;
    }
    .btn-sm.restore {
    background: #34d399;
    }
    .badge-invalid {
    display: inline-block;
    background: #7f1d1d;
    color: #fecaca;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 12px;
    margin-left: 8px;
    }
    .kvs {
    margin-top: 10px;
    font-size: 14px;
    line-height: 1.5;
    }
    .kvs div {
    margin-bottom: 4px;
    }
</style>
</head>
<body>
<div class="wrap">
    <div class="nav">
    <a href="/">Search</a>
    <a href="/groups">Groups</a>
    <a href="/admin">Admin</a>
    </div>
    <div class="card">
    <h1>__TITLE__</h1>
    __BODY__
    </div>
</div>
</body>
</html>
"""
    html = html.replace("__TITLE__", escape(title))
    html = html.replace("__BODY__", body)
    return HTMLResponse(html)


def query_image_to_rgb(data: bytes) -> np.ndarray:
    with Image.open(BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return np.asarray(im)


def exif_block(exif: dict) -> str:
    if not exif:
        return '<div class="muted">No EXIF</div>'

    keys = [
        "DateTimeOriginal",
        "CreateDate",
        "Make",
        "Model",
        "LensModel",
        "FocalLength",
        "FNumber",
        "ISO",
        "ExposureTime",
        "GPSLatitude",
        "GPSLongitude",
    ]

    parts = []
    for k in keys:
        if k in exif and exif[k] not in (None, ""):
            parts.append(
                f"<div><strong>{escape(str(k))}:</strong> {escape(str(exif[k]))}</div>"
            )

    if not parts:
        return '<div class="muted">No compact EXIF fields available</div>'

    return '<div class="kvs">' + "".join(parts) + "</div>"


def home_body() -> str:
    return """
<form action="/search" method="post" enctype="multipart/form-data">
<input type="file" name="file" accept="image/*" required>
<button type="submit">Search by face</button>
</form>

<p class="muted">
Upload a query image containing a face. Detection + embedding run through InsightFace on CUDA.
EXIF for indexed source images is stored in PostgreSQL and shown on result cards.
</p>
"""


def search_db(embedding, limit: int) -> list[dict]:
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select
                    f.id as face_id,
                    1 - (f.embedding <=> %s::vector) as similarity,
                    f.face_thumb_rel,
                    fi.file_thumb_rel,
                    fi.abs_path,
                    fi.rel_path,
                    fi.exif,
                    g.id as group_id
                from faces f
                join files fi on fi.id = f.file_id
                left join face_group_members gm on gm.face_id = f.id
                left join face_groups g on g.id = gm.group_id
                where f.invalidated = false
                and fi.is_deleted = false
                order by f.embedding <=> %s::vector
                limit %s
                """,
                (embedding.tolist(), embedding.tolist(), limit),
            )
            rows = cur.fetchall()
    return rows


def render_search_results(rows: list[dict]) -> str:
    if not rows:
        return '<p class="muted">No matches found.</p>'

    parts = [f'<p class="muted">Found {len(rows)} matches.</p><div class="grid">']

    for row in rows:
        face_thumb = (
            f'<img class="thumb" src="/thumbs/{escape(row["face_thumb_rel"])}" alt="face crop">'
            if row.get("face_thumb_rel")
            else ""
        )
        file_thumb = (
            f'<img class="thumb" src="/thumbs/{escape(row["file_thumb_rel"])}" alt="source image">'
            if row.get("file_thumb_rel")
            else ""
        )

        group_html = ""
        if row.get("group_id") is not None:
            group_html = (
                f'<div><strong>Group:</strong> '
                f'<a href="/groups/{int(row["group_id"])}">#{int(row["group_id"])}</a></div>'
            )
        else:
            group_html = '<div class="muted">Group: none</div>'

        exif = row.get("exif") or {}

        parts.append(
            '<div class="result">'
            f'<div><strong>Similarity:</strong> {float(row["similarity"]):.4f}</div>'
            f'<div><strong>File:</strong> <code>{escape(str(row["abs_path"]))}</code></div>'
            f'{group_html}'
            f'<div class="thumb-row">{face_thumb}{file_thumb}</div>'
            f'<h3>EXIF</h3>{exif_block(exif)}'
            '<form method="post" action="/faces/' + str(int(row["face_id"])) + '/toggle">'
            '<input type="hidden" name="next" value="/">'
            '<button class="btn-sm" type="submit">Invalidate match</button>'
            '</form>'
            '</div>'
        )

    parts.append("</div>")
    return "".join(parts)


def load_groups() -> list[dict]:
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select
                    g.id,
                    g.label,
                    g.member_count,
                    rf.face_thumb_rel as rep_face_thumb_rel
                from face_groups g
                left join faces rf on rf.id = g.rep_face_id
                order by g.member_count desc, g.id asc
                """
            )
            rows = cur.fetchall()
    return rows


def render_groups(rows: list[dict]) -> str:
    if not rows:
        return '<p class="muted">No groups built yet. Run <code>python -m facecat.group_cli rebuild</code>.</p>'

    parts = [f'<p class="muted">Groups: {len(rows)}</p><div class="grid">']
    for row in rows:
        rep = ""
        if row.get("rep_face_thumb_rel"):
            rep = f'<img class="thumb" src="/thumbs/{escape(row["rep_face_thumb_rel"])}" alt="group representative">'

        label = escape(str(row["label"])) if row.get("label") else "(unlabeled)"
        parts.append(
            '<div class="result">'
            f'<div><strong>Group #{int(row["id"])}</strong></div>'
            f'<div><strong>Label:</strong> {label}</div>'
            f'<div><strong>Members:</strong> {int(row["member_count"])}</div>'
            f'<div style="margin-top:10px;"><a href="/groups/{int(row["id"])}">Open group</a></div>'
            f'<div class="thumb-row" style="margin-top:12px;">{rep}</div>'
            '</div>'
        )

    parts.append("</div>")
    return "".join(parts)


def load_group_detail(group_id: int) -> tuple[dict | None, list[dict]]:
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select
                    g.id,
                    g.label,
                    g.member_count,
                    rf.face_thumb_rel as rep_face_thumb_rel
                from face_groups g
                left join faces rf on rf.id = g.rep_face_id
                where g.id = %s
                """,
                (group_id,),
            )
            group_row = cur.fetchone()

            if group_row is None:
                return None, []

            cur.execute(
                """
                select
                    f.id as face_id,
                    f.face_thumb_rel,
                    f.invalidated,
                    fi.file_thumb_rel,
                    fi.abs_path,
                    fi.exif,
                    gm.score_to_rep
                from face_group_members gm
                join faces f on f.id = gm.face_id
                join files fi on fi.id = f.file_id
                where gm.group_id = %s
                and fi.is_deleted = false
                order by gm.score_to_rep desc nulls last, f.id
                """,
                (group_id,),
            )
            members = cur.fetchall()

    return group_row, members


def render_group_detail(group_row: dict, members: list[dict]) -> str:
    rep = ""
    if group_row.get("rep_face_thumb_rel"):
        rep = f'<img class="thumb" src="/thumbs/{escape(group_row["rep_face_thumb_rel"])}" alt="group representative">'

    label = escape(str(group_row["label"])) if group_row.get("label") else "(unlabeled)"

    parts = [
        '<div class="result">',
        f'<div><strong>Group #{int(group_row["id"])}</strong></div>',
        f'<div><strong>Label:</strong> {label}</div>',
        f'<div><strong>Members:</strong> {int(group_row["member_count"])}</div>',
        f'<div class="thumb-row" style="margin-top:12px;">{rep}</div>',
        '</div>',
        f'<h2>Members ({len(members)})</h2>',
        '<div class="grid">',
    ]

    for row in members:
        face_thumb = (
            f'<img class="thumb" src="/thumbs/{escape(row["face_thumb_rel"])}" alt="face crop">'
            if row.get("face_thumb_rel")
            else ""
        )
        file_thumb = (
            f'<img class="thumb" src="/thumbs/{escape(row["file_thumb_rel"])}" alt="source image">'
            if row.get("file_thumb_rel")
            else ""
        )

        exif = row.get("exif") or {}

        invalid_badge = '<span class="badge-invalid">invalidated</span>' if row.get("invalidated") else ""
        btn_class = "btn-sm restore" if row.get("invalidated") else "btn-sm"
        btn_label = "Restore match" if row.get("invalidated") else "Invalidate match"

        parts.append(
            '<div class="result">'
            f'<div><strong>Face ID:</strong> {int(row["face_id"])}{invalid_badge}</div>'
            f'<div><strong>Score to representative:</strong> {float(row["score_to_rep"]):.4f}</div>'
            f'<div><strong>File:</strong> <code>{escape(str(row["abs_path"]))}</code></div>'
            f'<div class="thumb-row">{face_thumb}{file_thumb}</div>'
            f'<h3>EXIF</h3>{exif_block(exif)}'
            '<form method="post" action="/faces/' + str(int(row["face_id"])) + '/toggle">'
            '<input type="hidden" name="next" value="/groups/' + str(group_row["id"]) + '">'
            f'<button class="{btn_class}" type="submit">{btn_label}</button>'
            '</form>'
            '</div>'
        )

    parts.append("</div>")
    return "".join(parts)


def load_admin() -> dict:
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute("select count(*) as n from roots where enabled = true")
            roots_count = int(cur.fetchone()["n"])

            cur.execute("select count(*) as n from files where is_deleted = false")
            files_count = int(cur.fetchone()["n"])

            cur.execute("select count(*) as n from faces where invalidated = false")
            faces_count = int(cur.fetchone()["n"])

            cur.execute("select count(*) as n from face_groups")
            groups_count = int(cur.fetchone()["n"])

            cur.execute(
                """
                select id, path, last_indexed_at
                from roots
                where enabled = true
                order by id
                """
            )
            roots = cur.fetchall()

    db_size_bytes = db.database_size_bytes()
    jobs = db.get_latest_jobs(10)

    return {
        "roots_count": roots_count,
        "files_count": files_count,
        "faces_count": faces_count,
        "groups_count": groups_count,
        "db_size_bytes": db_size_bytes,
        "jobs": jobs,
        "roots": roots,
    }


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def render_admin(data: dict) -> str:
    parts = [
        "<table>",
        "<tbody>",
        f"<tr><th>Roots</th><td>{data['roots_count']}</td></tr>",
        f"<tr><th>Files</th><td>{data['files_count']}</td></tr>",
        f"<tr><th>Faces</th><td>{data['faces_count']}</td></tr>",
        f"<tr><th>Groups</th><td>{data['groups_count']}</td></tr>",
        f"<tr><th>Database size</th><td>{_human_size(data['db_size_bytes'])}</td></tr>",
        "</tbody>",
        "</table>",

        "<h2>Indexation / reindexation history (jobs)</h2>",
    ]

    if data["jobs"]:
        parts.append(
            "<table>"
            "<thead><tr><th>ID</th><th>Type</th><th>Status</th><th>Started</th><th>Finished</th><th>Stats</th></tr></thead>"
            "<tbody>"
        )
        for job in data["jobs"]:
            stats = job.get("stats") or {}
            if isinstance(stats, dict):
                stats_text = ", ".join(f"{k}={v}" for k, v in stats.items() if k != "done_blocks")
                done_blocks = stats.get("done_blocks")
                if done_blocks:
                    stats_text += f", blocks_done={len(done_blocks)}"
            else:
                stats_text = str(stats)
            parts.append(
                "<tr>"
                f"<td>{int(job['id'])}</td>"
                f"<td>{escape(str(job['job_type']))}</td>"
                f"<td>{escape(str(job['status']))}"
                + (f" ({escape(str(job['error']))})" if job.get("error") else "")
                + "</td>"
                f"<td>{escape(str(job['started_at']))}</td>"
                f"<td>{escape(str(job['finished_at']))}</td>"
                f"<td class='muted'>{escape(stats_text)}</td>"
                "</tr>"
            )
        parts.append("</tbody></table>")
    else:
        parts.append('<p class="muted">No jobs recorded yet.</p>')

    parts.extend(
        [
            "<h2>Registered roots</h2>",
            "<table>",
            "<thead><tr><th>ID</th><th>Path</th><th>Last indexed</th></tr></thead>",
            "<tbody>",
        ]
    )

    for row in data["roots"]:
        parts.append(
            "<tr>"
            f"<td>{int(row['id'])}</td>"
            f"<td><code>{escape(str(row['path']))}</code></td>"
            f"<td>{escape(str(row['last_indexed_at']))}</td>"
            "</tr>"
        )

    parts.extend(
        [
            "</tbody></table>",
            "<h2>CLI workflow</h2>",
            '<p><code>python -m facecat.index_cli add-root /path/to/photos</code></p>',
            '<p><code>python -m facecat.index_cli index-all --threads-per-gpu 4</code></p>',
            '<p><code>python -m facecat.group_cli rebuild --threads-per-gpu 4</code></p>',
            '<p class="muted">Both commands are crash-resumable: re-run the same command to continue an interrupted job.</p>',
        ]
    )

    return "".join(parts)


@app.get("/", response_class=HTMLResponse)
def home():
    return page("Face search", home_body())


@app.post("/search", response_class=HTMLResponse)
def search(file: UploadFile = File(...)):
    # Sync endpoint so FastAPI runs it in the thread pool; ONNX inference is
    # blocking and must not stall the event loop.
    data = file.file.read()
    rgb = query_image_to_rgb(data)
    # Detect at the same scale as indexing (MAX_DETECT_SIDE); full-res
    # detection on large photos yields embeddings that are not comparable to
    # the stored ones (self-match drops from ~1.0 to ~0.75).
    resized, _scale = imaging.resize_for_detector(rgb, config.MAX_DETECT_SIDE)
    embedding = ENGINE.embedding_from_best_face(resized)
    rows = search_db(embedding, config.SEARCH_LIMIT)
    body = home_body() + "<hr>" + render_search_results(rows)
    return page("Face search results", body)


@app.get("/groups", response_class=HTMLResponse)
def groups():
    rows = load_groups()
    return page("Groups", render_groups(rows))


@app.get("/groups/{group_id}", response_class=HTMLResponse)
def group_detail(group_id: int):
    group_row, members = load_group_detail(group_id)
    if group_row is None:
        return page("Group not found", f"<p>Group #{group_id} was not found.</p>")
    return page(f"Group #{group_id}", render_group_detail(group_row, members))


@app.post("/faces/{face_id}/toggle")
def toggle_face(face_id: int, next: str = Form(...)):
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "update faces set invalidated = not invalidated where id = %s returning invalidated",
                (face_id,),
            )
            row = cur.fetchone()
        conn.commit()

    if row is None:
        return RedirectResponse(url="/", status_code=303)
    print(f"[web] face {face_id} -> invalidated={bool(row['invalidated'])}")
    # Only trust root-relative redirects; anything else (an absolute URL or a
    # protocol-relative one like "//host") would be an open redirect.
    safe_next = next if next.startswith("/") and not next[:2] in ("//", "/\\") else "/"
    return RedirectResponse(url=safe_next, status_code=303)


@app.get("/admin", response_class=HTMLResponse)
def admin():
    data = load_admin()
    return page("Admin", render_admin(data))
