# Snake AI - Reinforcement Learning

## Overview
Snake AI is a Python project that integrates the classic Snake game with a Reinforcement Learning (RL) agent. The AI learns to control the snake, navigating it to consume food while avoiding collisions with walls, obstacles, and itself to maximize its score.

## Important Constants
### MAX_MEMORY
Maximum number of past experiences the agent can store for training.
### BATCH_SIZE
Number of experiences sampled from memory during each training step.
### LR (Learning Rate)
Rate at which the neural network updates its weights during training.
### HIDDEN_LAYERS
Defines the number and size of hidden layers in the neural network.
Expressed as a list with the size of each hidden layer.
### GAMMA
Discount factor determining the importance of future rewards in the RL algorithm.
### NUM_OBSTACLES
Number of obstacles present in the game environment.
### WRAP_AROUND
Allows the snake to pass through walls and appear on the opposite side if set to True.
### MODEL_SAVE_PREFIX
Prefix tag for saving and loading trained models.
### LOAD_PREVIOUS_MODEL
Flag to determine whether to load a pre-trained model before starting training.

## Acknowledgments
Based on [Teaching an AI to Play the Snake Game Using Reinforcement Learning](https://medium.com/@nancy.q.zhou/teaching-an-ai-to-play-the-snake-game-using-reinforcement-learning-6d2a6e8f3b1c)
