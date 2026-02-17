"""Run a trained Snake model for quick evaluation."""

import torch

from config import HIDDEN_DIMENSIONS, MODEL_PATH
from game_ai_env import TrainingGame
from model import LinearQNet

class GameModelRunner:
    """Load a trained model and run greedy evaluation episodes."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model = LinearQNet(11, HIDDEN_DIMENSIONS, 3)
        self.model.load(model_path)
        self.model.eval()
        self.game = TrainingGame()

    def select_action(self, state):
        """Pick the highest-value action from the model."""
        with torch.no_grad():
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()

        action = [0, 0, 0]
        action[move] = 1
        return action

    def run(self, episodes: int = 10):
        """Play a fixed number of episodes and report average score."""
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

        avg_score = total_score / max(1, episodes)
        print(
            f"Episodes: {episodes}\t"
            f"Avg Score: {avg_score:.2f}\t"
            f"Best Score: {best_score}"
        )

if __name__ == "__main__":
    runner = GameModelRunner()
    runner.run()

