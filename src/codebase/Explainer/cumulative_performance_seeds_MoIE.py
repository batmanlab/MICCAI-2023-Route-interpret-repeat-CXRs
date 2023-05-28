import os.path

import sklearn.metrics as metrics
import torch
import numpy as np

from Explainer.output_paths import paths


def get_cum_performance(args):
    soft_hard_concepts = "soft" if args.soft == "y" else "hard"
    accuracy_arr = []
    auroc_arr = []
    recall_arr = []
    precision_arr = []
    f1_arr = []
    print(f"Getting performance results for the dataset: {args.dataset}")
    print("#######################################################################################################")
    for seed in range(args.seed):
        print(f"===============>>> Seed: {seed} <<<===============")
        tensor_preds = torch.FloatTensor()
        tensor_y = torch.FloatTensor()

        for _iter in range(args.iterations):
            _iter += 1
            base_path = paths[f"{args.dataset}_{args.arch}"][f"iter{_iter}"]["base_path"]
            prev_path = paths[f"{args.dataset}_{args.arch}"][f"iter{_iter}"]["prev_path"]
            output = paths[f"{args.dataset}_{args.arch}"][f"iter{_iter}"]["output"]
            full_path = os.path.join(args.output, args.dataset, f"{soft_hard_concepts}_concepts", f"seed_{seed}",
                                     "explainer", base_path, prev_path, f"iter{_iter}", "explainer",
                                     output)
            gt_file = paths[f"{args.dataset}_{args.arch}"][f"iter{_iter}"]["gt_file"]
            pred_file = paths[f"{args.dataset}_{args.arch}"][f"iter{_iter}"]["pred_file"]

            print(f">>>>> Iteration_{_iter} : {full_path}")

            tensor_preds = torch.cat((tensor_preds, torch.load(os.path.join(full_path, pred_file))), dim=0)
            tensor_y = torch.cat((tensor_y, torch.load(os.path.join(full_path, gt_file))), dim=0)

        y_np = tensor_y.numpy()
        preds_np = tensor_preds.argmax(dim=1).numpy()
        preds_proba = torch.nn.Softmax(dim=1)(tensor_preds).numpy()[:, 1]
        print(f"y shape: {y_np.shape}")
        print(f"preds_np shape: {preds_np.shape}")
        print(f"preds_proba shape: {preds_proba.shape}")
        accuracy_arr.append(metrics.accuracy_score(y_np, preds_np))
        if args.dataset == "HAM10k" or args.dataset == "SIIM-ISIC" or args.dataset == "mimic_cxr":
            auroc_arr.append(metrics.roc_auc_score(y_np, preds_proba))
            recall_arr.append(metrics.recall_score(y_np, preds_np))
            precision_arr.append(metrics.precision_score(y_np, preds_np))
            f1_arr.append(metrics.f1_score(y_np, preds_np))
        else:
            recall_arr.append(metrics.recall_score(y_np, preds_np, average="macro"))
            precision_arr.append(metrics.precision_score(y_np, preds_np, average="macro"))
            f1_arr.append(metrics.f1_score(y_np, preds_np, average='macro'))
            auroc_arr.append(0.5)

    print("#######################################################################################################")

    np_acc = np.array(accuracy_arr)
    np_auroc = np.array(auroc_arr)
    np_recall = np.array(recall_arr)
    np_precision = np.array(precision_arr)
    np_f1 = np.array(f1_arr)

    print(f"Accuracy Mean (std dev): {np.mean(np_acc)} ({np.std(np_acc)})")
    print(f"Auroc Mean (std dev): {np.mean(np_auroc)} ({np.std(np_auroc)})")
    print(f"Recall Mean (std dev): {np.mean(np_recall)} ({np.std(np_recall)})")
    print(f"Precision Mean (std dev): {np.mean(np_precision)} ({np.std(np_precision)})")
    print(f"F1 Mean (std dev): {np.mean(np_f1)} ({np.std(np_f1)})")

    output = os.path.join(args.output, args.dataset, f"{soft_hard_concepts}_concepts", "performance_metrics", args.arch)
    os.makedirs(output, exist_ok=True)

    with open(os.path.join(output, "accuracy.npy"), 'wb') as f:
        np.save(f, np_acc)
    with open(os.path.join(output, "auroc.npy"), 'wb') as f:
        np.save(f, np_auroc)
    with open(os.path.join(output, "recall.npy"), 'wb') as f:
        np.save(f, np_recall)
    with open(os.path.join(output, "precision.npy"), 'wb') as f:
        np.save(f, np_precision)
    with open(os.path.join(output, "f1.npy"), 'wb') as f:
        np.save(f, np_f1)


    print(f"Save at: {output}")
