# Face Catalog

A simple searchable face catalog for local photo collections.

## Features

- Index JPEG, PNG, TIFF, WEBP, HEIC, and many RAW formats
- Detect faces and store ArcFace embeddings in PostgreSQL + pgvector
- Search by uploading a face photo
- Show thumbnails, file paths, and EXIF metadata
- Invalidate or restore individual face matches
- Simple admin page with counts and last index time

## Quick start

1. Put your photos in `./data/photos`
2. Start database and web app:

   ```bash
   make up
