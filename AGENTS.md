# Repository Guidelines

## Project Structure & Module Organization
- `train.py`: LoRA fine-tuning for Gemma 3 on Kansai-ben dataset. Writes adapters to `gemma3-kansaiben-lora/` and checkpoints under `output-lora/`.
- `eval.py`: Runs base and LoRA-applied inference with simple prompts.
- `pyproject.toml` / `uv.lock`: Python 3.13 project managed with `uv` and PyTorch CUDA wheels.
- `.devcontainer/` + `compose.yml`: GPU-enabled container for development. Loads env from `.env`.
- `README.md`: High-level overview (fill as needed).

## Build, Test, and Development Commands
- Install deps (host): `uv sync` — resolves and installs all runtime and dev dependencies.
- Run training: `uv run python train.py` — fine-tunes and saves adapters to `gemma3-kansaiben-lora/`.
- Run eval (LoRA): `uv run python eval.py --adapter_dir ./gemma3-kansaiben-lora` — compares base vs LoRA.
- Containerized dev: `docker compose run --rm --gpus all app bash` — launches the GPU dev shell with project mounted.

## Coding Style & Naming Conventions
- Language: Python (>=3.13), 4-space indentation, UTF-8 source.
- Lint/format: `ruff check .` and `ruff format .` (use `--fix` to auto-fix). Keep imports sorted by Ruff defaults.
- Naming: snake_case for functions/vars, PascalCase for classes, UPPER_CASE for module constants (e.g., `MODEL_NAME`).

## Testing Guidelines
- No formal test suite yet. Validate changes via `eval.py` with representative prompts and record latency/quality.
- Prefer deterministic runs for comparisons: omit `--do_sample` or fix seeds in scripts.
- If adding tests, place under `tests/` and name `test_*.py`; use `pytest` if introduced.

## Commit & Pull Request Guidelines
- Commits: present-tense, concise subject (e.g., "train: mask prompt tokens", "eval: add sampling flag").
- Group related changes; avoid mixed refactors and features together.
- PRs: include purpose, summary of changes, how to run (commands used), and notable metrics (loss curves, sample outputs, GPU/driver info). Link issues if applicable.

## Security & Configuration Tips
- GPU: Compose requests NVIDIA GPUs; ensure host supports `--gpus all` and correct CUDA drivers.
- Tokens/data: Do not hardcode secrets. Use `.env` and avoid committing large artifacts; add new paths to `.gitignore`.
- Reproducibility: Keep dataset IDs, model names, and key hyperparameters configurable via top-level constants/CLI flags.

