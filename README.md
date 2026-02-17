# Snake AI - Reinforcement Learning

## Overview
Snake AI is a Python project that combines the classic Snake game with a Reinforcement Learning (RL) agent. The AI learns to control the snake, enabling it to consume food while avoiding collisions with walls, obstacles, and itself. The primary goal is to maximize the snake's score through continuous interaction and learning within the game environment. This project showcases the application of RL techniques in game development and provides a foundation for exploring more complex AI-driven game mechanics.

## Shared Design System
- Visual and board constants are sourced from the sibling Bazza Game Design System package (`bgds` imports).
- For standalone environments, install it with `pip install -e ../bazza-game-design-system`.

## Install
```bash
pip install -r requirements.txt
```

## Run Instructions
- Run **`train_ai.py`** to start training the AI model.
- Run **`play_ai.py`** to evaluate a trained model.
- Run **`play_human.py`** for the user-controlled game.

## Project Layout
- `game.py`: Base game state, rendering, collisions, and board mechanics.
- `game_ai_env.py`: Training environment wrapper with rewards and state vector.
- `train_ai.py`: DQN training loop and replay-memory agent.
- `play_ai.py`: Model evaluation runner.
- `play_human.py`: Manual gameplay loop.
- `model.py`: Q-network and optimizer/training step logic.
- `config.py`: Shared configuration.

## Important Constants
- **`MAX_MEMORY`**: Maximum number of past experiences the agent can store for training.
- **`BATCH_SIZE`**: Number of experiences sampled from memory during each training step.
- **`LEARNING_RATE`**: Rate at which the neural network updates its weights during training.
- **`HIDDEN_DIMENSIONS`**: Number and size of hidden layers in the neural network.
- **`GAMMA`**: Discount factor determining the importance of future rewards in the RL algorithm.
- **`NUM_OBSTACLES`**: Number of obstacles present in the game environment.
- **`WRAP_AROUND`**: Allows the snake to pass through walls and appear on the opposite side if set to True.
- **`MODEL_PATH`**: Path used for saving and loading the trained model.
- **`LOAD_MODEL`**: Flag to determine whether to load a pre-trained model before starting training.

## Dependencies
- Python 3.10+
- See `requirements.txt`

## Acknowledgments
Based on [Teaching an AI to Play the Snake Game Using Reinforcement Learning](https://medium.com/@nancy.q.zhou/teaching-an-ai-to-play-the-snake-game-using-reinforcement-learning-6d2a6e8f3b1c)

