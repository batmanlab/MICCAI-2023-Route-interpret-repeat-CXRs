import argparse
import json
import os
import pickle
import random
import sys

import numpy as np
import pandas as pd
import torch

import MIMIC_CXR.mimic_cxr_utils as FOL_mimic

sys.path.append(os.path.abspath("/ocean/projects/asc170022p/shg121/PhD/ICLR-2022"))


def config():
    parser = argparse.ArgumentParser(description='Get important concepts masks')
    parser.add_argument('--base_path', metavar='DIR',
                        default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022',
                        help='path to checkpoints')
    parser.add_argument('--output', metavar='DIR',
                        default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out',
                        help='path to output logs')

    parser.add_argument('--disease', type=str, default="effusion", help='dataset name')
    parser.add_argument('--seed', type=int, default=0, help='Seed')
    parser.add_argument('--iterations', type=int, default="1", help='total number of iteration')
    parser.add_argument('--model', default='MoIE', type=str, help='MoIE')
    parser.add_argument('--icml', default='n', type=str, help='ICML or MICCAI')

    return parser.parse_args()


def calculate_performance(disease, iterations, output, json_file, _seed, dataset_path):
    random.seed(_seed)
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    results = FOL_mimic.get_residual_outputs(
        iterations, json_file, output, disease, _seed, dataset_path
    )
    print("")
    return results, dataset_path


def main():
    args = config()
    _disease = args.disease
    _iters = args.iterations
    _seed = args.seed
    _output = args.output
    if args.icml == "y":
        print("=====================>>>>> Calculating performance for ICML paper <<<<<<=====================")
        dataset_path = f"{_output}/mimic_cxr/t/lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4/densenet121/{_disease}/dataset_g"
        output = f"/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out/mimic_cxr/explainer/{_disease}"
        json_file = os.path.join(args.base_path, "codebase", "MIMIC_CXR", "paths_mimic_cxr_icml.json")
    else:
        dataset_path = f"{_output}/mimic_cxr/t/lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4/densenet121/{_disease}/dataset_g"
        output = f"{_output}/mimic_cxr/soft_concepts/seed_{_seed}/explainer/{_disease}"
        json_file = os.path.join(args.base_path, "codebase", "MIMIC_CXR", "paths_mimic_cxr.json")

    results, path_to_dump = calculate_performance(_disease, _iters, output, json_file, _seed, dataset_path)
    print(results)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(path_to_dump, f"Residual_results.csv"))
    print(path_to_dump)
    print(df.iloc[:, 0:5])
    pickle.dump(results, open(os.path.join(path_to_dump, f"total_results_{_disease}.pkl"), "wb"))
    print(path_to_dump)


if __name__ == '__main__':
    main()
