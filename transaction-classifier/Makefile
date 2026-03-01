PROJECT_NAME=transaction_classifier

.PHONY: lint
lint:
	@uv run ruff check .
	@uv run ruff format . --check

.PHONY: build_features
build_features:
	@uv run python -m src.features

.PHONY: train
train:
	@uv run python -m src.train

.PHONY: jupyter
jupyter:
	@uv run jupyter notebook notebooks