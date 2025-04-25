
from pathlib import Path

BLOCKS_TO_UNFREEZE = ['block_16']
BATCH_SIZE = 32
IMAGE_SIZE = 224
LEARNING_RATE_1 = 0.0005
LEARNING_RATE_2 = 0.0001
OVERSAMPLE_FACTOR = 4
NUM_EPOCHS = 40

PROJECT_DIR = Path(__file__).resolve().parent.absolute()
DATA_DIR = PROJECT_DIR / "data"
TEST_DIR = DATA_DIR / "dataset_test" / "test"
TRAIN_1_DIR = DATA_DIR / "dataset_stage1" / "train"
VAL_1_DIR = DATA_DIR / "dataset_stage1" / "val"
TRAIN_2_DIR = DATA_DIR / "dataset_stage2" / "train"
VAL_2_DIR = DATA_DIR / "dataset_stage2" / "validation"
MODEL_DIR = PROJECT_DIR / "models"

'''
Based solely on the number of images per class, the class weights are:

akiec: 1.076
bcc: 0.898
bkl: 0.602
df: 1.909
mel: 1.964
nv: 0.221
vasc: 1.640

After some experimentation, I found that the following class weights work well:
'''
# CLASS_WEIGHTS = {
#     'akiec': 1.5,
#     'bcc': 1.0,
#     'bkl': 1.0,
#     'df': 2.5,
#     'mel': 2.0,
#     'nv': 0.2,
#     'vasc': 2.0
# }
CLASS_WEIGHTS = {
    0: 1.5,
    1: 1.0,
    2: 1.0,
    3: 2.5,
    4: 2.0,
    5: 0.2,
    6: 2.0
}

'''
The same holds for the class multiplying factors for the oversampling:
'''
CLASS_MULTIPLIERS = {
    'akiec': 4,
    'bcc': 2,
    'bkl': 1,
    'df': 5,
    'mel': 3,
    'nv': 1,
    'vasc': 4
}