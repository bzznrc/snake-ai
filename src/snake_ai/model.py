"""Snake RL model and trainer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def _resolve_model_path(path_value: str) -> Path:
    return Path(path_value)


class LinearQNet(nn.Module):
    """Feedforward Neural Network with customizable hidden layers."""

    def __init__(self, input_size: int, hidden_layers: list[int], output_size: int):
        super().__init__()
        layers: list[nn.Module] = []
        in_size = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size

        layers.append(nn.Linear(in_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def save(self, file_path_value: str) -> None:
        file_path = _resolve_model_path(file_path_value)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load(self, file_path_value: str) -> None:
        file_path = _resolve_model_path(file_path_value)
        if file_path.exists():
            self.load_state_dict(torch.load(file_path))
            self.eval()
            print(f"Model loaded from {file_path}")
        else:
            print(f"No model found at {file_path}. Starting from scratch.")


class QTrainer:
    """Trainer class for the Q-learning model."""

    def __init__(self, model: LinearQNet, lr: float, gamma: float):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done) -> float:
        state = torch.from_numpy(np.asarray(state, dtype=np.float32))
        next_state = torch.from_numpy(np.asarray(next_state, dtype=np.float32))
        action = torch.from_numpy(np.asarray(action, dtype=np.float32))
        reward = torch.from_numpy(np.asarray(reward, dtype=np.float32))
        done = torch.from_numpy(np.asarray(done, dtype=np.bool_))

        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0)

        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            q_new = reward[idx]
            if not done[idx]:
                q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

        return float(loss.item())
