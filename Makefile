.PHONY: up down logs index groups web \
	docker-build docker-up docker-down docker-logs docker-index docker-group

# --- Direct run (host venv via setenv.sh) -----------------------------------
up:
	docker compose up -d db

down:
	docker compose down

logs:
	docker compose logs -f db

# Index all registered roots (crash-resumable: re-run to continue).
index:
	. ./setenv.sh && python -m facecat.index_cli index-all --threads-per-gpu 4

# Rebuild face groups on the GPUs (crash-resumable: re-run to continue).
groups:
	. ./setenv.sh && python -m facecat.group_cli rebuild --threads-per-gpu 4

web:
	. ./setenv.sh && uvicorn facecat.webapp:app --host 0.0.0.0 --port 8000

# --- Docker (GPU) ------------------------------------------------------------
docker-build:
	docker compose build

docker-up:
	docker compose up -d db webapp

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f webapp

# Crash-resumable jobs; re-run after an interruption to continue.
docker-index:
	docker compose run --rm index

docker-group:
	docker compose run --rm group