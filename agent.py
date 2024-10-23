import torch
import random
import numpy as np
from collections import deque
from snake_game_ai import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import LinearQNet, QTrainer
import matplotlib.pyplot as plt
from IPython import display
from constants import *

plt.ion()  # Interactive mode on

def plot(scores, avg_scores):
    """Plot the scores over time."""
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    
    # Normalize RGB colors to [0, 1]
    plt.plot(scores, label='Score', color=tuple([c / 255 for c in COLOR_SNAKE_PRIMARY]))           # Dark Teal
    plt.plot(avg_scores, label='Avg. Score', color=tuple([c / 255 for c in COLOR_SNAKE_OUTLINE]))  # Turquoise
    plt.legend()
    plt.show(block=False)
    plt.pause(0.1)

class Agent:
    """Agent that interacts with the environment and learns from it."""

    def __init__(self):
        self.n_games = 0
        self.epsilon = EPSILON_START  # Initial exploration rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = LinearQNet(11, HIDDEN_LAYERS, 3)
        if LOAD_PREVIOUS_MODEL:
            self.model.load(f"{MODEL_SAVE_PREFIX}.pth")
            print(f"Loaded model from {MODEL_SAVE_PREFIX}.pth")
        self.trainer = QTrainer(self.model, lr=LR, gamma=GAMMA)

    def get_state(self, game):
        """Get the current state representation."""
        head = game.snake[0]
        # Points around the head
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        # Current direction
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # Food is left
            game.food.x > game.head.x,  # Food is right
            game.food.y < game.head.y,  # Food is up
            game.food.y > game.head.y   # Food is down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """Train the model using experiences from memory."""
        if len(self.memory) > BATCH_SIZE:
            mini_batch = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_batch = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_batch)
        loss = self.trainer.train_step(states, actions, rewards, next_states, dones)
        return loss

    def train_short_memory(self, state, action, reward, next_state, done):
        """Train the model using the latest experience."""
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """Decide on an action based on the current state."""
        # Exploration vs Exploitation
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
        if random.random() < self.epsilon:
            # Explore: random action
            move = random.randint(0, 2)
        else:
            # Exploit: choose the best action based on the model
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
        # Convert move to one-hot encoding
        action = [0, 0, 0]
        action[move] = 1
        return action

def train():
    """Train the agent."""
    scores = []
    avg_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # Get the current state
        state_old = agent.get_state(game)

        # Get the action
        action = agent.get_action(state_old)

        # Perform the action and get the new state
        reward, done, score = game.play_step(action)
        state_new = agent.get_state(game)

        # Train short memory
        agent.train_short_memory(state_old, action, reward, state_new, done)

        # Remember the experience
        agent.remember(state_old, action, reward, state_new, done)

        if done:
            # Train long memory (replay)
            game.reset()
            agent.n_games += 1
            loss = agent.train_long_memory()

            # Record scores
            scores.append(score)
            total_score += score
            avg_score = total_score / agent.n_games
            avg_scores.append(avg_score)

            # Save the model every 50 games
            if agent.n_games % 50 == 0:
                agent.model.save(f"{MODEL_SAVE_PREFIX}.pth")

            # Save the model if record is broken
            if score > record:
                record = score

            # Print training progress
            print(f'Game {agent.n_games}, Score: {score}, Record: {record}, Avg. Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}')

            # Plot the scores
            if PLOT_TRAIN:
                plot(scores, avg_scores)

if __name__ == '__main__':
    train()