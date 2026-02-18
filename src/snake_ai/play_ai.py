"""Run a trained Snake model for quick evaluation."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
else:
    from pathlib import Path

import torch

from snake_ai.config import (
    HIDDEN_DIMENSIONS,
    MODEL_BEST_PATH,
    MODEL_PATH,
    NUM_ACTIONS,
    NUM_INPUT_FEATURES,
    resolve_show_game,
)
from snake_ai.game import TrainingSnakeGame
from snake_ai.logging_utils import configure_logging, log_run_context
from snake_ai.model import LinearQNet


class GameModelRunner:
    """Load a trained model and run greedy evaluation episodes."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model = LinearQNet(NUM_INPUT_FEATURES, HIDDEN_DIMENSIONS, NUM_ACTIONS)
        self.model.load(model_path)
        self.model.eval()
        self.game = TrainingSnakeGame(show_game=resolve_show_game(default_value=True))

    def select_action(self, state) -> list[int]:
        with torch.no_grad():
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()

        action = [0] * NUM_ACTIONS
        action[move] = 1
        return action

    def run(self, episodes: int = 10) -> None:
        total_score = 0
        best_score = 0

        for _ in range(episodes):
            self.game.reset()
            done = False

            while not done:
                state = self.game.get_state_vector()
                action = self.select_action(state)
                _, done, score = self.game.play_step(action)

            total_score += score
            best_score = max(best_score, score)

        self.game.close()
        avg_score = total_score / max(1, episodes)
        print(f"Episodes: {episodes}\tAvg Score: {avg_score:.2f}\tBest Score: {best_score}")


def run_ai(episodes: int = 10) -> None:
    configure_logging()
    model_path = MODEL_BEST_PATH if Path(MODEL_BEST_PATH).exists() else MODEL_PATH
    runner = GameModelRunner(model_path=model_path)
    log_run_context("play-ai", {"episodes": episodes, "model": model_path})
    runner.run(episodes=episodes)


if __name__ == "__main__":
    run_ai()
