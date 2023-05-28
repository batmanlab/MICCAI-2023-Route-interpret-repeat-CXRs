import torch.nn as nn
import torch
from torchvision import models, transforms
import os
from tqdm import tqdm
import numpy as np


def get_model(args, use_pretrained=True, model_name="resnet50", full_model=False):
    if model_name.lower() == "ham10000":
        from derma_models import get_model
        model, model_bottom, model_top = get_model(args, model_name.lower())
        preprocess = transforms.Compose([
                        transforms.Resize(299),
                        transforms.CenterCrop(299),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                      ])
        if args.dataset == "isic":
            model = None
    else:
        raise ValueError(model_name)

    if full_model:
        return model, model_bottom, preprocess
    else:
        return model_bottom, preprocess


class PosthocLinearCBM(nn.Module):
    def __init__(self, concept_bank, idx_to_class=None, n_classes=5):
        """
        PosthocCBM Linear Layer. 
        Takes an embedding as the input, outputs class-level predictions using only concept margins.
        Args:
            concept_bank (ConceptBank): [description]
            idx_to_class (dict, optional): A mapping from the output indices to the class names. Defaults to None.
            n_classes (int, optional): Number of classes in the classification problem. Defaults to 5.
        """
        super(PosthocLinearCBM, self).__init__()
        # Get the concept information from the bank
        self.cavs = concept_bank.vectors
        self.intercepts = concept_bank.intercepts
        self.norms = concept_bank.norms
        self.names = concept_bank.concept_names.copy()
        self.n_concepts = self.cavs.shape[0]

        self.n_classes = n_classes
        # Will be used to plot classifier weights nicely
        self.idx_to_class = idx_to_class if idx_to_class else {i: i for i in range(self.n_classes)}

        # A single linear layer will be used as the classifier
        self.classifier = nn.Linear(self.n_concepts, self.n_classes)

    def compute_dist(self, emb):
        # Computing the geometric margin to the decision boundary specified by CAV.
        margins = (torch.matmul(self.cavs, emb.T) +
           self.intercepts) / (self.norms)
        return margins.T

    def forward(self, emb, return_dist=False):
        x = self.compute_dist(emb)
        out = self.classifier(x)
        if return_dist:
            return out, x
        return out
    
    def trainable_params(self):
        return self.classifier.parameters()
    
    def classifier_weights(self):
        return self.classifier.weight
    
    def set_weights(self, weights, bias):
        self.classifier.weight.data = torch.tensor(weights).to(self.classifier.weight.device)
        self.classifier.bias.data = torch.tensor(bias).to(self.classifier.weight.device)
        return 1

    def analyze_classifier(self, k=5, print_lows=False):
        weights = self.classifier.weight.clone().detach()
        output = []
        for idx, cls in self.idx_to_class.items():
            cls_weights = weights[idx]
            topk_vals, topk_indices = torch.topk(cls_weights, k=k)
            topk_indices = topk_indices.detach().cpu().numpy()
            topk_concepts = [self.names[j] for j in topk_indices]
            analysis_str = [f"Class : {cls}"]
            for j, c in enumerate(topk_concepts):
                analysis_str.append(f"\t {j+1} - {c}: {topk_vals[j]:.3f}")
            analysis_str = "\n".join(analysis_str)
            output.append(analysis_str)

            if print_lows:
                topk_vals, topk_indices = torch.topk(-cls_weights, k=k)
                topk_indices = topk_indices.detach().cpu().numpy()
                topk_concepts = [self.names[j] for j in topk_indices]
                analysis_str = [f"Class : {cls}"]
                for j, c in enumerate(topk_concepts):
                    analysis_str.append(f"\t {j+1} - {c}: {-topk_vals[j]:.3f}")
                analysis_str = "\n".join(analysis_str)
                output.append(analysis_str)

        analysis = "\n".join(output)
        return analysis


