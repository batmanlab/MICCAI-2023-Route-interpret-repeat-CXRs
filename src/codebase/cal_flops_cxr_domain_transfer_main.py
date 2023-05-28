import argparse
import os
import sys

from Explainer.experiments_explainer_cxr_domain_transfer import train_glt

sys.path.append(os.path.abspath("/ocean/projects/asc170022p/shg121/PhD/ICLR-2022"))


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', metavar='DIR',
                        default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022',
                        help='path to checkpoints')
    parser.add_argument('--image_dir_path', metavar='DIR',
                        default='/ocean/projects/asc170022p/shared/Data/chestXRayDatasets/StanfordCheXpert',
                        help='image path in ocean')
    parser.add_argument('--image_source_dir', metavar='DIR', type=str, default='CheXpert-v1.0-small',
                        help='dataset directory')
    parser.add_argument('--image_col_header', metavar='DIR', type=str, default='Path',
                        help='dataset directory')
    parser.add_argument('--checkpoints', metavar='DIR',
                        default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints',
                        help='path to checkpoints')
    parser.add_argument('--output', metavar='DIR',
                        default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out',
                        help='path to output logs')
    parser.add_argument('--logs', metavar='DIR',
                        default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/log',
                        help='path to tensorboard logs')
    parser.add_argument('--chexpert_names', nargs='+',
                        default=["No_Finding", "Enlarged_Cardiomediastinum", "Cardiomegaly", "Lung_Opacity",
                                 "Lung_Lesion",
                                 "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Effusion",
                                 "Pleural_Other", "Fracture", "Support Devices"]
                        )
    parser.add_argument('--batch-size', default=16, type=int, metavar='N', help='batch size BB')
    parser.add_argument('--flattening-type', type=str, default="flatten", help='flatten or adaptive or maxpool')
    parser.add_argument('--uncertain', default=1, type=int, help='number of epochs warm up.')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--epochs', type=int, default=500, help='epochs')
    parser.add_argument('--disease', type=str, default="effusion", help='disease name')
    parser.add_argument('--arch', type=str, default="densenet121", help='arch')
    parser.add_argument('--tot_samples', type=int, default="1000", help='tot_samples')
    parser.add_argument('--image_size', default=512, type=int, help='image_size.')
    parser.add_argument('--crop_size', default=512, type=int, help='image_size.')
    parser.add_argument('--model', default='MoIE', type=str, help='MoIE or Baseline_CBM_logic or Baseline_PCBM_logic')
    parser.add_argument('--target_dataset', default='stanford_cxr', type=str, help='dataset')
    parser.add_argument('--source_dataset', type=str, default="mimic_cxr", help='source dataset name')
    parser.add_argument('--source-checkpoint-t-path', type=str,
                        default="lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4",
                        help='dataset folder of concepts')
    parser.add_argument('--target-checkpoint-t-path', type=str,
                        default="lr_0.1_epochs_90_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4",
                        help='dataset folder of concepts')
    parser.add_argument('--checkpoint-model', metavar='file', nargs="+",
                        default=['xxxx'],
                        help='checkpoint file of the model GatedLogicNet')
    parser.add_argument('--channels', default=3, type=int, help='channels ')
    parser.add_argument('--optim', type=str, default="SGD", help='optimizer of GLT')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--train_phi', default="n", type=str, metavar='TYPE')
    parser.add_argument('--profile', default="n", type=str, metavar='TYPE')
    parser.add_argument('--selection_threshold', default=0.5, type=float,
                        help='selection threshold of the selector for the test/val set')

    return parser.parse_args()


if __name__ == "__main__":
    args = config()
    args.expert_to_train = "explainer"
    args.iter = 1
    args.cov = 0.4
    args.initialize_w_mimic = "y"
    print("**************************************** Iter 1 ****************************************")
    flops1_y = train_glt(args)
    print()
    print(f"flops1_y: {flops1_y}")
    print()
    print("**************************************** Iter 1 ****************************************")
    print()

    lm = 128
    args.expert_to_train = "explainer"
    if args.disease == "effusion":
        lm = 96.0
    elif args.disease == "cardiomegaly":
        lm = 1024.0
    elif args.disease == "edema":
        lm = 128.0

    args.iter = 2
    args.cov = 0.3
    args.initialize_w_mimic = "y"
    args.prev_covs = [0.4]
    args.prev_chk_pt_explainer_folder = [
        "{soft_hard_filter}_concepts/seed_{seed}/" + args.disease + "/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_" + str(
            lm) + "_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4_sample_" + str(
            args.tot_samples)]
    print(args.prev_chk_pt_explainer_folder)
    args.checkpoint_model = ["best_model.pth.tar"]
    print("**************************************** Iter 2 ****************************************")
    flops2_y = train_glt(args)
    print()
    print(f"flops2_y: {flops2_y}")
    print()
    print("**************************************** Iter 2 ****************************************")
    print()

    args.iter = 3
    args.cov = 0.3
    args.initialize_w_mimic = "y"
    args.prev_covs = [0.4, 0.3]
    args.prev_chk_pt_explainer_folder = [
        "{soft_hard_filter}_concepts/seed_{seed}/" + args.disease + "/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_" + str(
            lm) + "_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4_sample_" + str(
            args.tot_samples),
        "{soft_hard_filter}_concepts/seed_{seed}/" + args.disease + "/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_" + str(
            lm) + "_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4_sample_" + str(
            args.tot_samples)]
    print(args.prev_chk_pt_explainer_folder)
    args.checkpoint_model = ["best_model.pth.tar", "best_model.pth.tar"]
    print("**************************************** Iter 3 ****************************************")
    flops3 = train_glt(args)
    print()
    print(f"flops3: {flops3}")
    print()
    print("**************************************** Iter 3 ****************************************")
    print()

    args.expert_to_train = "residual"
    args.source_checkpoint_bb_path = "lr_0.01_epochs_60_loss_CE"
    args.source_checkpoint_bb = "g_best_model_epoch_8.pth.tar" if args.disease == "effusion" else "g_best_model_epoch_4.pth.tar"

    print("**************************************** Iter 3 residual ****************************************")
    flops_r = train_glt(args)
    print()
    print(f"flops_r: {flops_r}")
    print()
    print("**************************************** Iter 3 residual ****************************************")
    print()

    args.expert_to_train = "explainer"
    args.cov = 0.4
    args.initialize_w_mimic = "n"
    print("**************************************** Iter 1 ****************************************")
    flops1_n = train_glt(args)
    print()
    print(f"flops1_n: {flops1_n}")
    print()
    print("**************************************** Iter 1 ****************************************")
    print()

    args.expert_to_train = "explainer"
    args.iter = 2
    args.cov = 0.3
    args.prev_covs = [0.4]
    args.initialize_w_mimic = "n"
    args.prev_chk_pt_explainer_folder = [
        "{soft_hard_filter}_concepts/seed_{seed}/" + args.disease + "/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_" + str(
            lm) + "_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4_sample_" + str(
            args.tot_samples)]
    print(args.prev_chk_pt_explainer_folder)
    args.checkpoint_model = ["best_model.pth.tar"]
    print("**************************************** Iter 2 ****************************************")
    flops2_n = train_glt(args)
    print()
    print(f"flops2_n: {flops2_n}")
    print()
    print("**************************************** Iter 2 ****************************************")
    print()

    print("****************************************************************************************")
    print("*************************** Flops with fine tuning G and Pi ****************************")
    print(f"MoIE: {flops1_y + flops2_y + flops3}")
    print(f"MoIE + R : {flops1_y + flops2_y + flops3 + flops_r}")

    print("**************************** Flops with fine tuning only Pi ****************************")
    print(f"MoIE: {flops1_n + flops2_n + flops3}")
    print(f"MoIE + R : {flops1_n + flops2_n + flops3 + flops_r}")
    print("****************************************************************************************")
