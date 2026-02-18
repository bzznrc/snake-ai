"""Main training loop and DQN agent implementation for Snake."""

from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from collections import deque
import logging
import random

import torch

from snake_ai.config import (
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
)
from snake_ai.runtime.helpers import configure_logging
from snake_ai.train.env import TrainingGame
from snake_ai.train.model import LinearQNet, QTrainer

LOGGER = logging.getLogger("snake_ai.train")


class DQNAgent:
    """Agent that interacts with the environment and learns from it."""

    def __init__(self):
        self.n_games = 0
        self.epsilon = EPSILON_START
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = LinearQNet(11, HIDDEN_DIMENSIONS, 3)
        if LOAD_MODEL:
            self.model.load(MODEL_PATH)
        self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=GAMMA)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_batch = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_batch = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_batch)
        return self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def select_action(self, state):
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
    configure_logging()
    scores = []
    average_scores = []
    total_score = 0
    record = 0

    agent = DQNAgent()
    game = TrainingGame()
    LOGGER.info(
        "Train / Model: %s / Load: %s / Batch: %s / Eps: %.3f->%.3f",
        MODEL_PATH,
        LOAD_MODEL,
        BATCH_SIZE,
        EPSILON_START,
        EPSILON_MIN,
    )

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

            LOGGER.info(
                "Episode=%s Score=%s Record=%s Avg=%.2f Epsilon=%.3f",
                agent.n_games,
                score,
                record,
                average_score,
                agent.epsilon,
            )


if __name__ == "__main__":
    train()
