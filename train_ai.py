"""Main training loop and DQN agent implementation for Snake."""

import random
from collections import deque

import torch
import matplotlib.pyplot as plt
from IPython import display

from config import (
    BATCH_SIZE,
    EPSILON_DECAY,
    EPSILON_MIN,
    EPSILON_START,
    GAMMA,
    HIDDEN_DIMENSIONS,
    LEARNING_RATE,
    LOAD_MODEL,
    MAX_MEMORY,
    MODEL_PATH,
    PLOT_TRAINING,
)
from bgds.visual.colors import COLOR_AQUA, COLOR_DEEP_TEAL
from game_ai_env import TrainingGame
from model import LinearQNet, QTrainer

plt.ion()

def plot_training(scores, average_scores):
    """Plot score progression during training."""
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title("Training Progress")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")
    plt.plot(scores, label="Score", color=tuple(c / 255 for c in COLOR_DEEP_TEAL))
    plt.plot(average_scores, label="Avg. Score", color=tuple(c / 255 for c in COLOR_AQUA))
    plt.legend()
    plt.show(block=False)
    plt.pause(0.1)

class DQNAgent:
    """Agent that interacts with the environment and learns from it."""

    def __init__(self):
        self.n_games = 0
        self.epsilon = EPSILON_START
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = LinearQNet(11, HIDDEN_DIMENSIONS, 3)
        if LOAD_MODEL:
            self.model.load(MODEL_PATH)
            print(f"Loaded model from {MODEL_PATH}")
        self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=GAMMA)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """Train model using a random sample from replay memory."""
        if len(self.memory) > BATCH_SIZE:
            mini_batch = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_batch = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_batch)
        return self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        """Train model using the latest transition."""
        self.trainer.train_step(state, action, reward, next_state, done)

    def select_action(self, state):
        """Select action using epsilon-greedy policy."""
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()

        action = [0, 0, 0]
        action[move] = 1
        return action

def train():
    """Train the Snake DQN agent."""
    scores = []
    average_scores = []
    total_score = 0
    record = 0

    agent = DQNAgent()
    game = TrainingGame()

    while True:
        state_old = game.get_state_vector()
        action = agent.select_action(state_old)

        reward, done, score = game.play_step(action)
        state_new = game.get_state_vector()

        agent.train_short_memory(state_old, action, reward, state_new, done)
        agent.remember(state_old, action, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            scores.append(score)
            total_score += score
            average_score = total_score / agent.n_games
            average_scores.append(average_score)

            if agent.n_games % 50 == 0:
                agent.model.save(MODEL_PATH)

            if score > record:
                record = score

            print(
                f"Game {agent.n_games}\t"
                f"Score: {score}\t"
                f"Record: {record}\t"
                f"Avg. Score: {average_score:.2f}\t"
                f"Epsilon: {agent.epsilon:.3f}"
            )

            if PLOT_TRAINING:
                plot_training(scores, average_scores)

if __name__ == "__main__":
    train()

