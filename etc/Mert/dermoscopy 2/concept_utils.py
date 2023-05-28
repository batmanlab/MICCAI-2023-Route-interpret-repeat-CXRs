from collections import defaultdict
import numpy as np
import torch
from sklearn.svm import SVC
from tqdm import tqdm


class EasyDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class ConceptBank:
    def __init__(self, concept_dict, device):
        all_vectors, concept_names, all_intercepts = [], [], []
        all_margin_info = defaultdict(list)
        for k, (tensor, _, _, intercept, margin_info) in concept_dict.items():
            all_vectors.append(tensor)
            concept_names.append(k)
            all_intercepts.append(np.array(intercept).reshape(1, 1))
            for key, value in margin_info.items():
                if key != "train_margins":
                    all_margin_info[key].append(np.array(value).reshape(1, 1))
        for key, val_list in all_margin_info.items():
            margin_tensor = torch.tensor(np.concatenate(
                val_list, axis=0), requires_grad=False).float().to(device)
            all_margin_info[key] = margin_tensor

        self.concept_info = EasyDict()
        self.concept_info.margin_info = EasyDict(dict(all_margin_info))
        self.concept_info.vectors = torch.tensor(np.concatenate(all_vectors, axis=0), requires_grad=False).float().to(
            device)
        self.concept_info.norms = torch.norm(
            self.concept_info.vectors, p=2, dim=1, keepdim=True).detach()
        self.concept_info.intercepts = torch.tensor(np.concatenate(all_intercepts, axis=0),
                                                    requires_grad=False).float().to(device)
        self.concept_info.concept_names = concept_names
        print("Concept Bank is initialized.")

    def __getattr__(self, item):
        return self.concept_info[item]

@torch.no_grad()
def get_embedding(loader, model, n_samples=100, device="cuda"):
    """
    Args:
        loader ([type]): Data loader returning only the images
        model ([type]): Backbone
        n_samples (int, optional): Number of samples to extract the activations
        device (str, optional): Device to use. Defaults to "cpu".

    Returns:
        [type]: Activations as a numpy array.
    """
    activations = None
    with torch.no_grad():
        for image in tqdm(loader):
            image = image.to(device)
            try:
                batch_act = model.encode_image(image).squeeze().detach().cpu().numpy()
            
            except:
                batch_act = model(image).squeeze().detach().cpu().numpy()
            if activations is None:
                activations = batch_act
            else:
                activations = np.concatenate([activations, batch_act], axis=0)
            if activations.shape[0] >= n_samples:
                return activations[:n_samples]
        raise ValueError(f"Insufficient number of samples: {activations.shape}. Desired: {n_samples}")


def get_cavs(X_train, y_train, X_val, y_val, C):
    """Extract the concept activation vectors and the corresponding stats

    Args:
        X_train, y_train, X_val, y_val: NumPy arrays to learn the concepts with.
        C: Regularizer for the SVM. 
    """
    svm = SVC(C=C, kernel="linear")
    svm.fit(X_train, y_train)
    train_acc = svm.score(X_train, y_train)
    test_acc = svm.score(X_val, y_val)
    train_margin = ((np.dot(svm.coef_, X_train.T) + svm.intercept_) / np.linalg.norm(svm.coef_)).T
    margin_info = {"max": np.max(train_margin),
                   "min": np.min(train_margin),
                   "pos_mean":  np.nanmean(train_margin[train_margin > 0]),
                   "pos_std": np.nanstd(train_margin[train_margin > 0]),
                   "neg_mean": np.nanmean(train_margin[train_margin < 0]),
                   "neg_std": np.nanstd(train_margin[train_margin < 0]),
                   "q_90": np.quantile(train_margin, 0.9),
                   "q_10": np.quantile(train_margin, 0.1),
                   "pos_count": X_train.shape[0]//2,
                   "neg_count": X_train.shape[0]//2,
                   }
    concept_info = (svm.coef_, train_acc, test_acc, svm.intercept_, margin_info)
    return concept_info


def learn_concepts(pos_loader, neg_loader, model_bottom, n_samples, C, train_ratio=0.5, device="cuda"):
    """Learning CAVs and related margin stats.
    Args:
        pos_loader (torch.utils.data.DataLoader): A PyTorch DataLoader yielding positive samples for each concept
        neg_loader (torch.utils.data.DataLoader): A PyTorch DataLoader yielding negative samples for each concept
        model_bottom (nn.Module): Mode
        n_samples (int): Number of positive samples to use while learning the concept.
        C (float): Regularization parameter for the SVM. Possibly multiple options.
        device (str, optional): Device to use while extracting activations. Defaults to "cuda".

    Returns:
        dict: Concept information, including the CAV and margin stats.
    """
    print("Extracting Embeddings: ")
    pos_act = get_embedding(pos_loader, model_bottom, n_samples=2*n_samples, device=device)
    neg_act = get_embedding(neg_loader, model_bottom, n_samples=2*n_samples, device=device)
    train_idx = int(train_ratio*pos_act.shape[0])
    X_train = np.concatenate([pos_act[:train_idx], neg_act[:train_idx]], axis=0)
    X_val = np.concatenate([pos_act[train_idx:], neg_act[train_idx:]], axis=0)
    y_train = np.concatenate([np.ones(train_idx), np.zeros(train_idx)], axis=0)
    y_val = np.concatenate([np.ones(X_val.shape[0]//2), np.zeros(X_val.shape[0]//2)], axis=0)
    print("\t Learning CAVS")
    concept_info = {}
    for c in C:
        concept_info[c] = get_cavs(X_train, y_train, X_val, y_val, c)
        print(f"\t Reg-C: {c}, Training Acc: {concept_info[c][1]}, Validation Acc: {concept_info[c][2]}")
    return concept_info
