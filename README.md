# Snake AI

Lightweight Snake reinforcement-learning project with local-only modules.

## Quickstart
```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e .[dev]
```

Run:
```bash
python -m snake_ai
python -m snake_ai.train_ai
python -m snake_ai.play_ai
```

## Structure
- `src/snake_ai/config.py`: central config
- `src/snake_ai/core/`: base gameplay and rendering
- `src/snake_ai/train/`: RL env/model/training loop
- `src/snake_ai/runtime/`: local runtime helpers
- `src/snake_ai/boards/`: obstacle generation

## Tests
```bash
pytest
```
