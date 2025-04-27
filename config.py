"""
Configuration parameters for skin lesion classification model training.

This module centralizes all configuration parameters used across the project,
including model hyperparameters, directory paths, and class-specific weights.
"""
from pathlib import Path

# Model hyperparameters
BLOCKS_TO_UNFREEZE = ['block_16']
BATCH_SIZE = 32
IMAGE_SIZE = 224
LEARNING_RATE_1 = 0.0005  # Initial fine-tuning learning rate
LEARNING_RATE_2 = 0.0001  # Cross-style fine-tuning learning rate
OVERSAMPLE_FACTOR = 4
NUM_EPOCHS = 40
NUM_CLASSES = 7

# Dataset splitting parameters
TEST_SIZE = 0.15
VAL_SIZE = 0.1765  # 15/85 split of 85%

# Directory structure
PROJECT_DIR = Path(__file__).resolve().parent.absolute()
DATA_DIR = PROJECT_DIR / "data"
TEST_DIR = DATA_DIR / "dataset_test" / "test"
TRAIN_1_DIR = DATA_DIR / "dataset_stage1" / "train"
VAL_1_DIR = DATA_DIR / "dataset_stage1" / "val"
TRAIN_2_DIR = DATA_DIR / "dataset_stage2" / "train"
VAL_2_DIR = DATA_DIR / "dataset_stage2" / "validation"
MODEL_DIR = PROJECT_DIR / "models"
PLOT_DIR = PROJECT_DIR / "plots"

# Class names mapping (for easier reference)
CLASS_NAMES = {
    0: 'akiec',
    1: 'bcc',
    2: 'bkl',
    3: 'df',
    4: 'mel',
    5: 'nv',
    6: 'vasc'
}

# Class weights for loss function
'''
Based solely on the number of images per class, the class weights would be:

akiec: 1.076
bcc: 0.898
bkl: 0.602
df: 1.909
mel: 1.964
nv: 0.221
vasc: 1.640

After some experimentation, I found that the following class weights work well:
'''
CLASS_WEIGHTS = {
    0: 1.5,  # akiec
    1: 1.0,  # bcc
    2: 1.0,  # bkl
    3: 2.5,  # df
    4: 2.0,  # mel
    5: 0.2,  # nv
    6: 2.0   # vasc
}

# Oversampling multipliers to handle class imbalance
CLASS_MULTIPLIERS = {
    'akiec': 4,
    'bcc': 2,
    'bkl': 1,
    'df': 5,
    'mel': 3,
    'nv': 1,
    'vasc': 4
}