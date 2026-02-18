"""Run a trained Snake model for quick evaluation."""

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from snake_ai.config import HIDDEN_DIMENSIONS, MODEL_PATH
from snake_ai.train.env import TrainingGame
from snake_ai.train.model import LinearQNet


class GameModelRunner:
    """Load a trained model and run greedy evaluation episodes."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model = LinearQNet(11, HIDDEN_DIMENSIONS, 3)
        self.model.load(model_path)
        self.model.eval()
        self.game = TrainingGame()

    def select_action(self, state):
        with torch.no_grad():
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()

        action = [0, 0, 0]
        action[move] = 1
        return action

    def run(self, episodes: int = 10):
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


def run_ai(episodes: int = 10):
    runner = GameModelRunner()
    runner.run(episodes=episodes)


if __name__ == "__main__":
    run_ai()
