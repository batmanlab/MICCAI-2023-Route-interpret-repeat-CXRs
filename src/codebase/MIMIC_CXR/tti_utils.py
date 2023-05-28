import ast
import pickle

import numpy as np
import torch

import utils
from Explainer.models.Gated_Logic_Net import Gated_Logic_Net
from Explainer.models.explainer import Explainer


def get_model(moie_config, args, device, moie_checkpoint):
    if args.model == "MoIE":
        moIE = Gated_Logic_Net(
            moie_config.input_size_pi,
            moie_config.concept_names,
            moie_config.labels,
            moie_config.hidden_nodes,
            moie_config.conceptizator,
            moie_config.temperature_lens,
        ).to(device)
        moIE.load_state_dict(torch.load(moie_checkpoint)["state_dict"])
        moIE.eval()
        return moIE
    else:
        model = Explainer(
            n_concepts=len(moie_config.concept_names), n_classes=len(moie_config.labels),
            explainer_hidden=moie_config.hidden_nodes,
            conceptizator=moie_config.conceptizator, temperature=moie_config.temperature_lens,
        ).to(device)
        model.load_state_dict(torch.load(moie_checkpoint)["state_dict"])
        model.eval()
        return model


def perform_tti(args, df, moie_checkpoint, moie_config_path, alpha_norm_path):
    print("Performing interventions")
    device = utils.get_device()
    moie_config = pickle.load(open(moie_config_path, "rb"))
    moie_config.concept_names = args.concept_names
    moIE = get_model(moie_config, args, device, moie_checkpoint)
    labels = moie_config.labels
    tensor_alpha_norm = torch.load(alpha_norm_path)

    y = torch.FloatTensor()
    y_preds = torch.FloatTensor()
    y_pred_intervene = torch.FloatTensor()

    with torch.no_grad():
        for idx in range(df.shape[0]):
            concept_proba = ast.literal_eval(df.loc[idx, "all_concept_proba"])
            concept_gt = ast.literal_eval(df.loc[idx, "all_concept_gt"])
            target_class = torch.tensor(int(df.loc[idx, "ground_truth"])).reshape(1)
            y_hat = torch.tensor(int(df.loc[idx, "g_pred"])).reshape(1)
            y_hat_logits = torch.tensor(ast.literal_eval(df.loc[idx, "g_pred_logit"])).reshape(1, -1)
            test_tensor_concepts = torch.tensor(np.array(concept_proba), dtype=torch.float)
            tensor_attrs = torch.tensor(np.array(concept_gt), dtype=torch.float)

            top_concepts = torch.topk(tensor_alpha_norm[y_hat], args.topK)[1]
            concept_s = test_tensor_concepts
            concept_s[top_concepts] = tensor_attrs[top_concepts]

            if args.model == "MoIE":
                y_pred_ex, _, _ = moIE(concept_s.unsqueeze(0).to(device))
            else:
                y_pred_ex = moIE(concept_s.unsqueeze(0).to(device))

            pred_in = y_pred_ex.argmax(dim=1).item()

            y = torch.cat((y, target_class), dim=0)
            y_preds = torch.cat((y_preds, torch.nn.Softmax(dim=1)(y_hat_logits)), dim=0)
            y_pred_intervene = torch.cat((y_pred_intervene, torch.nn.Softmax(dim=1)(y_pred_ex.cpu())), dim=0)
            # print(
            #     f"id: {idx}, "
            #     f"Ground Truth: {labels[target_class]}, "
            #     f"Predicted(g) : {y_hat}, "
            #     f"Predicted(alt) : {pred_in}, "
            # )

    return y, y_preds, y_pred_intervene


def zero_out_concepts_tti(args, df, moie_checkpoint, moie_config_path, alpha_norm_path):
    print("Zeroing out concepts due to interventions")
    device = utils.get_device()
    moie_config = pickle.load(open(moie_config_path, "rb"))
    moie_config.concept_names = args.concept_names
    moIE = get_model(moie_config, args, device, moie_checkpoint)
    labels = moie_config.labels
    tensor_alpha_norm = torch.load(alpha_norm_path)

    y = torch.FloatTensor()
    y_preds = torch.FloatTensor()
    y_pred_intervene = torch.FloatTensor()

    with torch.no_grad():
        for idx in range(df.shape[0]):
            concept_proba = ast.literal_eval(df.loc[idx, "all_concept_proba"])
            concept_gt = ast.literal_eval(df.loc[idx, "all_concept_gt"])
            target_class = torch.tensor(int(df.loc[idx, "ground_truth"])).reshape(1)
            y_hat = torch.tensor(int(df.loc[idx, "g_pred"])).reshape(1)
            y_hat_logits = torch.tensor(ast.literal_eval(df.loc[idx, "g_pred_logit"])).reshape(1, -1)
            test_tensor_concepts = torch.tensor(np.array(concept_proba), dtype=torch.float)
            tensor_attrs = torch.tensor(np.array(concept_gt), dtype=torch.float)

            top_concepts = torch.topk(tensor_alpha_norm[y_hat], args.topK)[1]
            concept_s = test_tensor_concepts
            concept_s[top_concepts] = 0

            if args.model == "MoIE":
                y_pred_ex, _, _ = moIE(concept_s.unsqueeze(0).to(device))
            else:
                y_pred_ex = moIE(concept_s.unsqueeze(0).to(device))

            pred_in = y_pred_ex.argmax(dim=1).item()

            y = torch.cat((y, target_class), dim=0)
            y_preds = torch.cat((y_preds, torch.nn.Softmax(dim=1)(y_hat_logits)), dim=0)
            y_pred_intervene = torch.cat((y_pred_intervene, torch.nn.Softmax(dim=1)(y_pred_ex.cpu())), dim=0)
            # print(
            #     f"id: {idx}, "
            #     f"Ground Truth: {labels[target_class]}, "
            #     f"Predicted(g) : {y_hat}, "
            #     f"Predicted(alt) : {pred_in}, "
            # )

    return y, y_preds, y_pred_intervene
