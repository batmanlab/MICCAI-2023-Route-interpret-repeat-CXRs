import os
import torch
import pandas as pd
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# HAM10k can be downloaded from
# from https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

# Download and Place in the folder below, it should contain the following:
# /path/ham10k/HAM10000_images_part_1
# /path/ham10k/HAM10000_images_part_2
# /path/ham10k/HAM10000_metadata.csv
DATA_DIR = "/path/ham10k/"


class DermDataset(Dataset):
    def __init__(self, df, preprocess=None):
        self.df = df
        self.preprocess = preprocess

    def __len__(self):
        return (len(self.df))

    def __getitem__(self, index):
        X = Image.open(self.df['path'].iloc[index])
        y = torch.tensor(int(self.df['y'].iloc[index]))
        if self.preprocess:
            X = self.preprocess(X)
        return X, y


def load_ham_data(args, preprocess):
    np.random.seed(args.seed)
    benign_malignant = {
        'nv': 'benign',
        'mel': 'malignant',
        'bkl': 'benign',
        'bcc': 'malignant',
        'akiec': 'benign',
        'vasc': 'benign',
        'df': 'benign'}
    class_to_idx = {"benign": 0, "malignant": 1}
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    df = pd.read_csv(os.path.join(DATA_DIR, 'HAM10000_metadata.csv'))
    all_image_paths = glob(os.path.join(DATA_DIR, '*', '*.jpg'))
    id_to_path = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_paths}

    df['path'] = df['image_id'].map(lambda id: id_to_path[id])
    df['benign_or_malignant'] = df["dx"].map(lambda id: benign_malignant[id])
    df['y'] = df["benign_or_malignant"].map(lambda id: class_to_idx[id])

    _, df_val = train_test_split(df, test_size=0.20, random_state=args.seed, stratify=df["dx"])
    df_train = df[~df.image_id.isin(df_val.image_id)]
    trainset = DermDataset(df_train, preprocess)
    valset = DermDataset(df_val, preprocess)
    print(f"Train, Val: {df_train.shape}, {df_val.shape}")
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.num_workers)

    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader, idx_to_class

