import argparse
import json
import os
import pickle
import sys

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
    parser.add_argument('--iteration', type=int, default="1", help='expert id')
    parser.add_argument('--model', default='MoIE', type=str, help='MoIE')

    return parser.parse_args()


def get_explanations(disease, root, iteration, output_path, dataset_path, path_to_dump, mode="test"):
    x_to_bool = 0.5

    device, configs = FOL_mimic.setup(output_path)
    df_master_sel = FOL_mimic.load_master_csv(configs, mode)
    (
        tensor_alpha, tensor_alpha_norm, tensor_concept_mask, mask_by_pi, out_put_g_pred,
        out_put_bb_pred, out_put_target, proba_concepts, ground_truth_concepts, concept_names
    ) = FOL_mimic.get_outputs(iteration, configs, output_path, dataset_path, mode)

    results = FOL_mimic.compute_performance_metrics(
        out_put_g_pred, out_put_bb_pred, out_put_target, mask_by_pi, configs
    )
    pickle.dump(results, open(os.path.join(path_to_dump, f"{mode}_performance_metrics_expert_{iteration}.pkl"), "wb"))

    moie = FOL_mimic.get_moie(configs, concept_names, iteration, disease, root, device)
    selected_ids = (mask_by_pi == 1).nonzero(as_tuple=True)[0]
    selected_ids_list = selected_ids.tolist()
    test_tensor_concepts_bool = (proba_concepts.cpu() > x_to_bool).to(torch.float)
    _feature_names = [f"feature{j:010}" for j in range(test_tensor_concepts_bool.size(1))]

    print("Creating Explanations")
    percentile_selection = 99
    ii = 0
    results_arr = []
    for _idx in selected_ids_list:
        ii += 1
        print(f"===>> {ii} <<===")
        results = FOL_mimic.compute_explanations_per_sample(
            iteration,
            _idx,
            df_master_sel,
            _feature_names,
            out_put_g_pred,
            out_put_bb_pred,
            out_put_target,
            test_tensor_concepts_bool,
            tensor_alpha_norm,
            percentile_selection,
            concept_names,
            moie,
            proba_concepts,
            ground_truth_concepts,
            device
        )
        results_arr.append(results)
        print(
            f" {[results['idx']]}, predicted: {configs.labels[results['g_pred']]}, target: {configs.labels[results['ground_truth']]}"
        )
        print(f" {configs.labels[results['g_pred']]} <=> {results['actual_explanations']}")
    return results_arr


def main():
    args = config()
    _disease = args.disease
    _iter = args.iteration
    _seed = args.seed
    json_file = os.path.join(args.base_path, "codebase", "MIMIC_CXR", "paths_mimic_cxr.json")
    with open(json_file) as _file:
        paths = json.load(_file)
    root = paths[_disease]["MoIE_paths"][f"iter{_iter}"]["base_path"]
    print(root)
    output_path = f"{args.output}/mimic_cxr/soft_concepts/seed_{_seed}/explainer/{_disease}/{root}/iter{_iter}/g/selected/auroc"
    dataset_path = f"{args.output}/mimic_cxr/t/lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4/densenet121/{_disease}/dataset_g"
    path_to_dump = os.path.join(output_path, "FOLs")
    os.makedirs(path_to_dump, exist_ok=True)

    print("######################## Test Explanations Start ########################")
    test_results_arr = get_explanations(_disease, root, _iter, output_path, dataset_path, path_to_dump, mode="test")
    pickle.dump(test_results_arr, open(os.path.join(path_to_dump, f"test_FOL_results_expert_{_iter}.pkl"), "wb"))
    test_results_df = pd.DataFrame.from_dict(test_results_arr, orient='columns')
    test_results_df.to_csv(os.path.join(path_to_dump, f"test_results_expert_{_iter}.csv"))
    print("######################## Test Explanations End ########################")

    print("######################## Val Explanations Start ########################")
    val_results_arr = get_explanations(_disease, root, _iter, output_path, dataset_path, path_to_dump, mode="val")
    pickle.dump(val_results_arr, open(os.path.join(path_to_dump, f"val_FOL_results_expert_{_iter}.pkl"), "wb"))
    val_results_df = pd.DataFrame.from_dict(val_results_arr, orient='columns')
    val_results_df.to_csv(os.path.join(path_to_dump, f"val_results_expert_{_iter}.csv"))
    print("######################## Val Explanations End ########################")

    print("######################## Train Explanations Start ########################")
    train_results_arr = get_explanations(_disease, root, _iter, output_path, dataset_path, path_to_dump, mode="train")
    pickle.dump(train_results_arr, open(os.path.join(path_to_dump, f"train_FOL_results_expert_{_iter}.pkl"), "wb"))
    train_results_df = pd.DataFrame.from_dict(train_results_arr, orient='columns')
    train_results_df.to_csv(os.path.join(path_to_dump, f"train_results_expert_{_iter}.csv"))
    print("######################## Train Explanations End ########################")
    print(path_to_dump)


if __name__ == '__main__':
    main()
