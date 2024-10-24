# Snake AI - Reinforcement Learning

## Overview
Snake AI is a Python project that combines the classic Snake game with a Reinforcement Learning (RL) agent. The AI learns to control the snake, enabling it to consume food while avoiding collisions with walls, obstacles, and itself. The primary goal is to maximize the snake's score through continuous interaction and learning within the game environment. This project showcases the application of RL techniques in game development and provides a foundation for exploring more complex AI-driven game mechanics.

- Run **`game_user.py`** for the regular user-controlled game.
- Run **`agent.py`** to start the training of the AI model.

## Important Constants
- **`MAX_MEMORY`**: Maximum number of past experiences the agent can store for training.
- **`BATCH_SIZE`**: Number of experiences sampled from memory during each training step.
- **`LR` (Learning Rate)**: Rate at which the neural network updates its weights during training.
- **`HIDDEN_LAYERS`**: Defines the number and size of hidden layers in the neural network. Expressed as a list with the size of each hidden layer.
- **`GAMMA`**: Discount factor determining the importance of future rewards in the RL algorithm.
- **`NUM_OBSTACLES`**: Number of obstacles present in the game environment.
- **`WRAP_AROUND`**: Allows the snake to pass through walls and appear on the opposite side if set to True.
- **`MODEL_SAVE_PREFIX`**: Prefix tag for saving and loading trained models.
- **`LOAD_PREVIOUS_MODEL`**: Flag to determine whether to load a pre-trained model before starting training.

## Dependencies
- Python 3.10+
- Pygame
- PyTorch
- Matplotlib

## Acknowledgments
Based on [Teaching an AI to Play the Snake Game Using Reinforcement Learning](https://medium.com/@nancy.q.zhou/teaching-an-ai-to-play-the-snake-game-using-reinforcement-learning-6d2a6e8f3b1c)
