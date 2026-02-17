import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path


def _resolve_model_path(path_value: str) -> Path:
    return Path(path_value)


class LinearQNet(nn.Module):
    """Feedforward Neural Network with customizable hidden layers."""

    def __init__(self, input_size, hidden_layers, output_size):
        super().__init__()
        layers = []
        in_size = input_size

        # Dynamically create hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size

        # Output layer
        layers.append(nn.Linear(in_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)  # Returns raw logits

    def save(self, file_path_value):
        """Save the model to a file."""
        file_path = _resolve_model_path(file_path_value)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load(self, file_path_value):
        """Load the model from a file."""
        file_path = _resolve_model_path(file_path_value)
        if file_path.exists():
            self.load_state_dict(torch.load(file_path))
            self.eval()
            print(f"Model loaded from {file_path}")
        else:
            print(f"No model found at {file_path}. Starting from scratch.")

class QTrainer:
    """Trainer class for the Q-learning model."""

    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        """Perform one training step."""
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.bool)

        # Reshape if necessary
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0)

        # 1. Predict Q values with current state
        pred = self.model(state)

        # 2. Compute target Q values
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 3. Optimize the model
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

        return loss.item()  # Return the loss value
