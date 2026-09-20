.PHONY: install test lint lint-fix train feature-importance serve docker-build docker-run dvc-repro clean

install:
	pip install -r requirements-dev.txt
	pre-commit install

test:
	PYTHONPATH=. pytest tests/ -v

lint:
	ruff check src/ tests/

lint-fix:
	ruff check src/ tests/ --fix

train:
	PYTHONPATH=. python -m src.training.train --config configs/lstm.yaml

feature-importance:
	PYTHONPATH=. python -m src.training.feature_importance --config configs/lstm.yaml

serve:
	uvicorn src.serving.api:app --reload

dvc-repro:
	dvc repro

docker-build:
	docker build -t battery-soc-api .

docker-run:
	docker compose up

clean:
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache models data/processed data/interim
