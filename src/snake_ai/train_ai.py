"""Training entrypoint."""

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snake_ai.train.trainer import train

__all__ = ["train"]


if __name__ == "__main__":
    train()
