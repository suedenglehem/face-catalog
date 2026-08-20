from __future__ import annotations

from html import escape
from io import BytesIO

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

from . import config, db
from .vision import FaceEngine


app = FastAPI(title="Face Catalog")
app.mount("/thumbs", StaticFiles(directory=str(config.THUMBS_DIR)), name="thumbs")

ENGINE = FaceEngine()


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

        parts.append(
            '<div class="result">'
            f'<div><strong>Face ID:</strong> {int(row["face_id"])}</div>'
            f'<div><strong>Score to representative:</strong> {float(row["score_to_rep"]):.4f}</div>'
            f'<div><strong>File:</strong> <code>{escape(str(row["abs_path"]))}</code></div>'
            f'<div class="thumb-row">{face_thumb}{file_thumb}</div>'
            f'<h3>EXIF</h3>{exif_block(exif)}'
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

    return {
        "roots_count": roots_count,
        "files_count": files_count,
        "faces_count": faces_count,
        "groups_count": groups_count,
        "roots": roots,
    }


def render_admin(data: dict) -> str:
    parts = [
        "<table>",
        "<tbody>",
        f"<tr><th>Roots</th><td>{data['roots_count']}</td></tr>",
        f"<tr><th>Files</th><td>{data['files_count']}</td></tr>",
        f"<tr><th>Faces</th><td>{data['faces_count']}</td></tr>",
        f"<tr><th>Groups</th><td>{data['groups_count']}</td></tr>",
        "</tbody>",
        "</table>",
        "<h2>Registered roots</h2>",
        "<table>",
        "<thead><tr><th>ID</th><th>Path</th><th>Last indexed</th></tr></thead>",
        "<tbody>",
    ]

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
            '<p><code>python -m facecat.index_cli index-all</code></p>',
            '<p><code>python -m facecat.group_cli rebuild</code></p>',
        ]
    )

    return "".join(parts)


@app.get("/", response_class=HTMLResponse)
def home():
    return page("Face search", home_body())


@app.post("/search", response_class=HTMLResponse)
async def search(file: UploadFile = File(...)):
    data = await file.read()
    rgb = query_image_to_rgb(data)
    embedding = ENGINE.embedding_from_best_face(rgb)
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


@app.get("/admin", response_class=HTMLResponse)
def admin():
    data = load_admin()
    return page("Admin", render_admin(data))
