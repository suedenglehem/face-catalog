.PHONY: up down logs index init reindex

up:
	docker compose up -d db web

down:
	docker compose down

logs:
	docker compose logs -f web index db

index:
	docker compose run --rm index python -m facecat.index_cli --once

reindex:
	docker compose run --rm index python -m facecat.index_cli --once --rebuild

cluster:
	. .venv/bin/activate && python -m facecat.cluster_cli --threshold 0.4
