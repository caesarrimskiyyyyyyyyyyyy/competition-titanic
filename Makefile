.DEFAULT_GOAL := final

.PHONY: sync train predict final

sync:
	uv sync --dev

train:
	uv run python -m titanic.train

predict:
	uv run python -m titanic.predict

final: train predict
