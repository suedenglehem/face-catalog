# Face Catalog

A searchable face catalog for local photo collections (JPEG + RAW).

## Features

- Index JPEG, PNG, TIFF, WEBP, HEIC, and many RAW formats (rawpy)
- Detect faces (InsightFace SCRFD) and store ArcFace embeddings in PostgreSQL + pgvector
- Search by uploading a face photo; results show thumbnails, file paths, EXIF data
- Invalidate or restore individual face matches from the web UI
- Group page: clusters of similar faces built on the GPUs
- Admin page: system info (GPUs), file/face counts, database disk size, last indexation/reindexation dates, job history

## Multi-GPU pipeline

Indexing and grouping use **all visible NVIDIA GPUs** in parallel. Each GPU runs an
independent ONNX Runtime CUDA session; `--threads-per-gpu` controls how many worker
threads per GPU feed it (default 4).

```
index-all:   scan -> decode/RAW-convert (CPU pool) -> detect+embed (N x T threads, one per GPU) -> persist
rebuild:     load embeddings -> brute-force kNN on each GPU (T threads/GPU) -> connected components -> groups
```

Both commands are **crash-resumable**: progress is committed to the database as work
completes. If the machine stops mid-run, re-running the same command continues from
where it left off until every file/face has been processed. Use `--fresh` (index) or
`rebuild --force` (groups) to start over.

## Quick start

1. Put your photos in a directory tree and register its root:

   ```bash
   . ./setenv.sh
   python -m facecat.index_cli add-root /path/to/photos
   ```

2. Start the database:

   ```bash
   make up          # docker compose, pgvector/pg16 on :5432
   ```

3. Index everything (resumable):

   ```bash
   python -m facecat.index_cli index-all --threads-per-gpu 4
   ```

4. Build groups:

   ```bash
   python -m facecat.group_cli rebuild --threads-per-gpu 4
   ```

5. Run the web app:

   ```bash
   make web         # http://localhost:8000
   ```

## CLI reference

### `python -m facecat.index_cli`

| Command | Description |
| --- | --- |
| `add-root PATH` | Register a directory tree to index (recursive) |
| `list-roots` | Show registered roots and their status |
| `index-all` | Index all roots; resumes an interrupted run automatically |

Flags for `index-all`:

- `--threads-per-gpu N` — worker threads per GPU (default: `$THREADS_PER_GPU` or 4)
- `--gpus "0,1"` — physical GPU ids to use (default: all visible via CUDA_VISIBLE_DEVICES)
- `--cpu-workers N` — CPU decode/RAW-conversion pool size (default: `$INDEX_CPU_WORKERS`)
- `--fresh` — wipe previous progress and start from scratch

### `python -m facecat.group_cli`

| Command | Description |
| --- | --- |
| `rebuild` | Rebuild all groups on the GPUs; resumes an interrupted run automatically |
| `show` | Print group summary |

Flags for `rebuild`:

- `--threads-per-gpu N` — worker threads per GPU (default: `$THREADS_PER_GPU` or 4)
- `--gpus "0,1"` — physical GPU ids to use
- `--force` — drop existing edges/groups and start over
- `--k N` / `--threshold F` — override `$GROUP_K` / `$GROUP_THRESHOLD`

## Environment variables (`env`)

Loaded via `. ./setenv.sh`. Key settings:

- `DATABASE_URL`, `THUMBS_DIR`
- `MODEL_NAME=buffalo_l`, `DET_SIZE=640`, `MAX_DETECT_SIDE=1600`
- `GROUP_K=20`, `GROUP_THRESHOLD=0.55`, `SEARCH_LIMIT=50`
- `CUDA_VISIBLE_DEVICES=0,1` — which physical GPUs are visible
- `THREADS_PER_GPU=4` — default thread count per GPU for both CLIs
- `INDEX_CPU_WORKERS`, `INDEX_DECODE_QUEUE`, `INDEX_GPU_QUEUE` — index pipeline tuning
- `GROUP_BLOCK_SIZE`, `KNN_CHUNK` — grouping memory tuning

## Web pages

- `/` — search by face image upload
- `/groups` — similar-face clusters with representative thumbnails
- `/admin` — system info, counts, DB size, last indexation/reindexation, job history
- `/invalidate/<face_id>` / `/restore/<face_id>` — human match invalidation