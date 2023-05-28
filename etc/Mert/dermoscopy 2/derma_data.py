from glob import glob
import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from constants import *


class DermDataset(Dataset):
    def __init__(self, df, preprocess=None):
        self.df = df
        self.preprocess = preprocess
    
    def __len__(self):
        return(len(self.df))
    
    def __getitem__(self, index):
        X = Image.open(self.df['path'].iloc[index])
        y = torch.tensor(int(self.df['y'].iloc[index]))
        if self.preprocess:
            X = self.preprocess(X)
        return X,y


def load_ham_data(args, preprocess):
    np.random.seed(args.seed)
    id_to_lesion = {
    'nv': 'Melanocytic nevi',
    'mel': 'dermatofibroma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'}

    benign_malignant = {
    'nv': 'benign',
    'mel': 'malignant',
    'bkl': 'benign',
    'bcc': 'malignant',
    'akiec': 'benign',
    'vasc': 'benign',
    'df': 'benign'}

    df = pd.read_csv(os.path.join(DATA_DIR,'HAM10000_metadata.csv'))
    all_image_paths = glob(os.path.join(DATA_DIR, '*', '*.jpg'))
    id_to_path = {os.path.splitext(os.path.basename(x))[0] : x for x in all_image_paths}

    def path_getter(id):
        if id in id_to_path:
            return id_to_path[id] 
        else:
            return  "-1"
    
    df['path'] = df['image_id'].map(path_getter)
    df = df[df.path != "-1"] 
    df['dx_name'] = df['dx'].map(lambda id: id_to_lesion[id])
    df['benign_or_malignant'] = df["dx"].map(lambda id: benign_malignant[id])
    class_to_idx = {"benign": 0, "malignant": 1}

    df['y'] = df["benign_or_malignant"].map(lambda id: class_to_idx[id])

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
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


def load_isic(args, preprocess, n_train=2000, n_val=400):
    df = pd.read_csv(os.path.join(ISIC_FOLDER, 'train.csv'))
    df['path'] = df['image_name'].map(lambda name: os.path.join(ISIC_FOLDER, "train", name + '.jpg'))
    df['y'] = df['target']

    files = os.listdir(os.path.join(ISIC_FOLDER, "train"))
    files = [os.path.join(ISIC_FOLDER, "train", f) for f in files]
    df = df[df.path.isin(files)]
    
    df_pos = df[df.y == 1]
    df_neg = df[df.y == 0]

    _, df_val_pos = train_test_split(df_pos, test_size=0.20, random_state=args.seed)
    _, df_val_neg = train_test_split(df_neg, test_size=0.20, random_state=args.seed)
    df_train_pos = df_pos[~df_pos.path.isin(df_val_pos.path)]
    df_train_neg = df_neg[~df_neg.path.isin(df_val_neg.path)]

    df_train_pos = df_train_pos.sample(n_train//5, random_state=args.seed)
    df_train_neg = df_train_neg.sample(4*n_train//5, random_state=args.seed)

    df_val_pos = df_val_pos.sample(n_val//5, random_state=args.seed)
    df_val_neg = df_val_neg.sample(4*n_val//5, random_state=args.seed)

    df_train = pd.concat([df_train_pos, df_train_neg])
    df_val = pd.concat([df_val_pos, df_val_neg])

    trainset = DermDataset(df_train, preprocess)
    valset = DermDataset(df_val, preprocess)

    print(f"Train, Val: {df_train.shape}, {df_val.shape}")
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          shuffle=True, num_workers=args.num_workers)
    
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                      shuffle=False, num_workers=args.num_workers)

    class_to_idx = {"benign": 0, "malignant": 1}
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return train_loader, val_loader, idx_to_class
