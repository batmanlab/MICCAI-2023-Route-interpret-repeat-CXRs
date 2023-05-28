import pandas as pd
import torch
import os
from torchvision import transforms
from PIL import Image
import argparse
import pickle
import numpy as np
from derma_models import get_model
from sklearn.svm import SVC
from constants import *
from concept_utils import learn_concepts, get_embedding


preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class EasyDataset():
    def __init__(self, images, base_dir=os.path.join(DERM7_FOLDER, "images"), transform=preprocess,
                        image_key="derm"):
        self.images = images
        self.transform = transform
        self.base_dir = base_dir
        self.image_key = image_key
    
    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.images.iloc[idx]
        img_path = os.path.join(self.base_dir, row[self.image_key])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", default=50, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--C", nargs="+", type=float, default=[0.01])
    parser.add_argument("--out-dir", default="./outputs/", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--model-name", default="HAM10000", type=str)
    return parser.parse_args()


@torch.no_grad()
def main(args):
    df = pd.read_csv(DERM7_META)
    train_indexes = list(pd.read_csv(TRAIN_IDX)['indexes'])
    val_indexes = list(pd.read_csv(VAL_IDX)['indexes'])
    print(df.columns)
    df["TypicalPigmentNetwork"] = df.apply(lambda row: {"absent": 0, "typical": 1, "atypical": -1}[row["pigment_network"]] ,axis=1)
    df["AtypicalPigmentNetwork"] = df.apply(lambda row: {"absent": 0, "typical": -1, "atypical": 1}[row["pigment_network"]] ,axis=1)

    df["RegularStreaks"] = df.apply(lambda row: {"absent": 0, "regular": 1, "irregular": -1}[row["streaks"]] ,axis=1)
    df["IrregularStreaks"] = df.apply(lambda row: {"absent": 0, "regular": -1, "irregular": 1}[row["streaks"]] ,axis=1)

    df["RegressionStructures"] = df.apply(lambda row: (1-int(row["regression_structures"] == "absent")) ,axis=1)

    df["RegularDG"] = df.apply(lambda row: {"absent": 0, "regular": 1, "irregular": -1}[row["dots_and_globules"]] ,axis=1)
    df["IrregularDG"] = df.apply(lambda row: {"absent": 0, "regular": -1, "irregular": 1}[row["dots_and_globules"]] ,axis=1)

    df["BWV"] = df.apply(lambda row: {"absent": 0, "present": 1}[row["blue_whitish_veil"]] ,axis=1)

    df = df.iloc[train_indexes+val_indexes]

    concepts = ["DarkSkinColor", "BWV", "RegularDG", "IrregularDG", "RegressionStructures", "IrregularStreaks",
                   "RegularStreaks", "AtypicalPigmentNetwork", "TypicalPigmentNetwork"]
    
    concept_libs = {C: {} for C in args.C}
    model, model_bottom, _ = get_model(args, args.model_name)

    np.random.seed(args.seed)
    for c_name in concepts: 
        print("\t Learning: ",  c_name)
        pos_df = df[df[c_name] == 1]
        neg_df = df[df[c_name] == 0]
        base_dir = os.path.join(DERM7_FOLDER, "images")

        print(pos_df.shape, neg_df.shape)
        
        if (pos_df.shape[0] < 2*args.n_samples) or (neg_df.shape[0] < 2*args.n_samples):
            print("\t Not enough samples! Sampling with replacement")
            pos_df = pos_df.sample(2*args.n_samples, replace=True)
            neg_df = neg_df.sample(2*args.n_samples, replace=True)
        else:
            pos_df = pos_df.sample(2*args.n_samples)
            neg_df = neg_df.sample(2*args.n_samples)
        
        pos_ds = EasyDataset(pos_df, base_dir=base_dir, image_key="derm")
        neg_ds = EasyDataset(neg_df, base_dir=base_dir, image_key="derm")
        pos_loader = torch.utils.data.DataLoader(pos_ds,
                                                 batch_size=2*args.n_samples,
                                                 shuffle=True,
                                                 num_workers=args.num_workers)

        neg_loader = torch.utils.data.DataLoader(neg_ds,
                                                 batch_size=2*args.n_samples,
                                                 shuffle=True,
                                                 num_workers=args.num_workers)
        
        # Get CAV for each concept using positive/negative image split
        cav_info = learn_concepts(pos_loader, neg_loader, model_bottom, args.n_samples, args.C, device="cuda")
        
        # Store CAV train acc, val acc, margin info for each regularization parameter and each concept
        for C in args.C:
            concept_libs[C][c_name] = cav_info[C]
            print(c_name, C, cav_info[C][1], cav_info[C][2])
    
    # Save CAV results 
    os.makedirs(args.out_dir, exist_ok=True)
    for C in concept_libs.keys():
        lib_path = os.path.join(args.out_dir, f"derma_{args.model_name}_{C}_{args.n_samples}.pkl")
        with open(lib_path, "wb") as f:
            pickle.dump(concept_libs[C], f)
        print(f"Saved to: {lib_path}")        


if __name__ == "__main__":
    args = config()
    main(args)