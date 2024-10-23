# Screen and Grid Dimensions
GRID_SIZE_X = 32          # Number of tiles horizontally
GRID_SIZE_Y = 24          # Number of tiles vertically
BLOCK_SIZE = 20           # Size of each grid tile
BB_HEIGHT = 30            # Bottom Bar height
BB_MARGIN = 20            # Margin for the bottom bar
SCREEN_WIDTH = GRID_SIZE_X * BLOCK_SIZE
SCREEN_HEIGHT = GRID_SIZE_Y * BLOCK_SIZE + BB_HEIGHT

# Game Speed
FPS = 20                  # Frames per second for the user game
AI_FPS = 180               # Frames per second for the AI game

# RGB color definitions
# RGB color definitions
COLOR_SNAKE_OUTLINE = (50, 215, 200)    # Turquoise (used as outline)
COLOR_SNAKE_PRIMARY = (30, 100, 100)    # Dark Teal (used as fill)

COLOR_FRUIT_OUTLINE = (240, 95, 95)     # Red Orange (used as outline)
COLOR_FRUIT_PRIMARY = (125, 45, 45)     # Dark Red (used as fill)

COLOR_BACKGROUND = (45, 45, 45)         # Dark Grey
COLOR_SCORE = (255, 255, 255)

COLOR_OBSTACLE_OUTLINE = (125, 125, 125)
COLOR_OBSTACLE_PRIMARY = COLOR_BACKGROUND

# Game constants
NUM_OBSTACLES = 10        # Number of obstacles to spawn in the game
MIN_SECTIONS = 2          # Minimum number of sections for obstacles
MAX_SECTIONS = 5          # Maximum number of sections for obstacles
WRAP_AROUND = True        # Allows snake to pass through walls if True

# Model constants
MODEL_SAVE_PREFIX = 'snake_32'      # Prefix tag for saving models
LOAD_PREVIOUS_MODEL = True          # Start training from a saved model if True

# Toggle for training plots
PLOT_TRAIN = False

# Agent hyperparameters
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001               # Learning rate
HIDDEN_LAYERS = [32]     # One hidden layer with 32 neurons
#HIDDEN_LAYERS = [32, 16]
GAMMA = 0.9              # Discount factor

# Epsilon parameters for exploration
EPSILON_START = 1.0
EPSILON_DECAY = 0.9995
EPSILON_MIN = 0.01