import torch

# Data Paths
DATA_DIR = "game_data"
LOGREG_PATH = "gomoku_logistic.pth"
MLP_PATH = "gomoku_mlp.pth"
CNN_PATH = "gomoku_cnn.pth"
RESNET_PATH = "gomoku_resnet.pth"

# Model Parameters
BOARD_SIZE = 15
NUM_CLASSES = BOARD_SIZE * BOARD_SIZE

# Training Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
EPOCHS = 50
EPOCHS_AUG = 20

# Hardware
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')