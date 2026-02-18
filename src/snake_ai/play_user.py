"""Human-play entrypoint for Snake."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snake_ai.game import HumanSnakeGame
from snake_ai.logging_utils import configure_logging, log_run_context


def run_user() -> None:
    configure_logging()
    log_run_context("play-user", {})
    game = HumanSnakeGame()
    score = 0
    try:
        while True:
            game_over, score = game.play_step()
            if game_over:
                break
    finally:
        game.close()

    print("Final Score:", score)


if __name__ == "__main__":
    run_user()
