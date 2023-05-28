import argparse
import os
import sys

from Explainer.experiments_explainer_cxr_domain_transfer import train_glt, test_glt

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
    parser.add_argument('--iter', type=int, default="1", help='iteration')
    parser.add_argument('--cov', type=float, default=0, help='coverage')
    parser.add_argument('--tot_samples', type=int, default="1000", help='tot_samples')
    parser.add_argument('--image_size', default=512, type=int, help='image_size.')
    parser.add_argument('--crop_size', default=512, type=int, help='image_size.')
    parser.add_argument('--model', default='MoIE', type=str, help='MoIE')
    parser.add_argument('--target_dataset', default='stanford_cxr', type=str, help='dataset')
    parser.add_argument('--source_dataset', type=str, default="mimic_cxr", help='source dataset name')
    parser.add_argument('--source-checkpoint-t-path', type=str,
                        default="lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4",
                        help='dataset folder of concepts')
    parser.add_argument('--target-checkpoint-t-path', type=str,
                        default="lr_0.1_epochs_90_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4",
                        help='dataset folder of concepts')
    parser.add_argument('--target_checkpoint', default='xxxx', type=str, help='target_checkpoint')
    parser.add_argument('--prev_chk_pt_explainer_folder', nargs='+', type=str,
                        default="xxxx",
                        help='chkpt explainer')
    parser.add_argument('--checkpoint-model', metavar='file', nargs="+",
                    default=['xxxx'],
                    help='checkpoint file of the model GatedLogicNet')
    parser.add_argument('--channels', default=3, type=int, help='channels ')
    parser.add_argument('--optim', type=str, default="SGD", help='optimizer of GLT')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--train_phi', default="n", type=str, metavar='TYPE')
    parser.add_argument('--initialize_w_mimic', default="n", type=str, metavar='TYPE')
    parser.add_argument('--selection_threshold', default=0.5, type=float,
                    help='selection threshold of the selector for the test/val set')
    parser.add_argument('--prev_covs', nargs='+', default=[0.4, 0.3, 0.3])
    return parser.parse_args()


if __name__ == "__main__":
    args = config()
    test_glt(args)
