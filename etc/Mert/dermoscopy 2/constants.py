import os

DERM7_FOLDER = "/path/to/derm7pt/"
DERM7_META = os.path.join(DERM7_FOLDER, "meta", "meta.csv")
TRAIN_IDX = os.path.join(DERM7_FOLDER, "meta", "train_indexes.csv")
VAL_IDX = os.path.join(DERM7_FOLDER, "meta", "valid_indexes.csv")

DATA_DIR = "/path/to/ham10000"
ISIC_FOLDER = "/path/to/isic"

# From the DDI paper. (https://arxiv.org/abs/2111.08006)

# google drive paths to models
MODEL_WEB_PATHS = {
'HAM10000':'https://drive.google.com/uc?id=1ToT8ifJ5lcWh8Ix19ifWlMcMz9UZXcmo',
}
