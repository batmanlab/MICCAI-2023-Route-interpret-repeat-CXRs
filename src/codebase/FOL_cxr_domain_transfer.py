import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch

import MIMIC_CXR.mimic_cxr_utils as FOL_mimic
from Explainer.models.Gated_Logic_Net import Gated_Logic_Net

sys.path.append(os.path.abspath("/ocean/projects/asc170022p/shg121/PhD/ICLR-2022"))


def config():
    parser = argparse.ArgumentParser(description='Get important concepts masks')
    parser.add_argument('--base_path', metavar='DIR',
                        default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022',
                        help='path to checkpoints')
    parser.add_argument('--checkpoints', metavar='DIR',
                        default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints',
                        help='path to checkpoints')
    parser.add_argument('--output', metavar='DIR',
                        default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out',
                        help='path to output logs')

    parser.add_argument('--disease', type=str, default="effusion", help='dataset name')
    parser.add_argument('--seed', type=int, default=0, help='Seed')
    parser.add_argument('--tot_samples', type=int, default=15000, help='Seed')
    parser.add_argument('--iteration', type=int, default=3, help='total number of iteration')
    parser.add_argument('--model', default='MoIE', type=str, help='MoIE')
    parser.add_argument('--icml', default='n', type=str, help='ICML or MICCAI')
    parser.add_argument('--target_dataset', default='stanford_cxr', type=str, help='target dataset')
    parser.add_argument('--cov', type=float, default=0.4)
    parser.add_argument('--initialize_w_mimic', default="n", type=str, metavar='TYPE')
    parser.add_argument('--train_phi', default="n", type=str, metavar='TYPE')

    return parser.parse_args()


def get_explanations(
        checkpoint, target_dataset, cov, tot_samples, disease, root, iteration, output_path, output_mimic,
        dataset_path,
        path_to_dump, mode="test"
):
    x_to_bool = 0.5
    device, configs = FOL_mimic.setup(output_mimic)
    configs.model = "MoIE"
    (
        tensor_alpha, tensor_alpha_norm, tensor_concept_mask, mask_by_pi, out_put_g_pred,
        out_put_bb_pred, out_put_target, proba_concepts, ground_truth_concepts, concept_names
    ) = FOL_mimic.get_outputs(iteration, configs, output_path, dataset_path, mode, domain_transfer=True)

    moie = Gated_Logic_Net(
        configs.input_size_pi, concept_names, configs.labels, configs.hidden_nodes,
        configs.conceptizator, configs.temperature_lens,
    ).to(device)

    # moie_checkpoint ="/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/mimic_cxr/soft_concepts/seed_0/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4/{configs.checkpoint_model[-1]}"
    moie_checkpoint = Path(f"{checkpoint}/best_model.pth.tar")
    print(moie_checkpoint)
    moie.load_state_dict(torch.load(moie_checkpoint)["state_dict"])
    moie.eval()
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
            None,
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
    output_mimic = f"{args.output}/mimic_cxr/soft_concepts/seed_{_seed}/explainer/{_disease}/{root}/iter{_iter}/g/selected/auroc"
    root = f"{root}_sample_{args.tot_samples}"
    dataset_path = f"{args.output}/mimic_cxr/t/lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4/densenet121/{_disease}/dataset_g"
    output_path = f"{args.output}/{args.target_dataset}/soft_concepts/seed_{_seed}/{_disease}/{root}/iter{_iter}/g/selected/auroc"
    checkpoint = f"{args.checkpoints}/{args.target_dataset}/soft_concepts/seed_{_seed}/{_disease}/{root}/iter{_iter}/g/selected/auroc"
    if args.cov != 0:
        output_path = f"{output_path}_cov_{args.cov}"
        checkpoint = f"{checkpoint}_cov_{args.cov}"
    if args.initialize_w_mimic == "y":
        output_path = f"{output_path}_initialize_w_mimic_{args.initialize_w_mimic}"
        checkpoint = f"{checkpoint}_initialize_w_mimic_{args.initialize_w_mimic}"

    path_to_dump = os.path.join(output_path, "FOLs")
    os.makedirs(path_to_dump, exist_ok=True)

    print("######################## Test Explanations Start ########################")
    test_results_arr = get_explanations(
        checkpoint, args.target_dataset, args.cov, args.tot_samples, _disease, root, _iter, output_path,
        output_mimic,
        dataset_path, path_to_dump, mode="test"
    )
    pickle.dump(test_results_arr, open(os.path.join(path_to_dump, f"test_FOL_results_expert_{_iter}.pkl"), "wb"))
    test_results_df = pd.DataFrame.from_dict(test_results_arr, orient='columns')
    test_results_df.to_csv(os.path.join(path_to_dump, f"test_results_expert_{_iter}_samples_{args.tot_samples}.csv"))
    print("######################## Test Explanations End ########################")


if __name__ == '__main__':
    main()
