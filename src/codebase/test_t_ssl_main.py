import argparse
import os
import sys

from BB.experiments_t_ssl import train_t, test_t

sys.path.append(os.path.abspath("/ocean/projects/asc170022p/shg121/PhD/ICLR-2022"))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def config():
    parser = argparse.ArgumentParser(description='Get important concepts masks')
    parser.add_argument('--image_dir_path', metavar='DIR',
                        default='/ocean/projects/asc170022p/shared/Data/chestXRayDatasets/StanfordCheXpert',
                        help='image path in ocean')
    parser.add_argument('--image_source_dir', metavar='DIR', type=str, default='CheXpert-v1.0-small',
                        help='dataset directory')
    parser.add_argument('--image_col_header', metavar='DIR', type=str, default='Path',
                        help='dataset directory')
    parser.add_argument('--logs', metavar='DIR',
                        default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/log',
                        help='path to tensorboard logs')
    parser.add_argument('--checkpoints', metavar='DIR',
                        default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints',
                        help='path to checkpoints')
    parser.add_argument('--output', metavar='DIR',
                        default='/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/out',
                        help='path to checkpoints')

    parser.add_argument('--target_dataset', type=str, default="stanford_cxr", help='target dataset name')
    parser.add_argument('--source_dataset', type=str, default="mimic_cxr", help='source dataset name')
    parser.add_argument('--root', type=str, default="BB/lr_0.01_epochs_10_loss_CE", help='root')
    parser.add_argument('--disease', type=str, default="edema", help='disease')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--print-freq', default=20, type=int,
                        metavar='N', help='display frequence (default: 20)')
    parser.add_argument('--arch', type=str, default="densenet121", help='Arch')
    parser.add_argument('--train_csv_file_name', type=str, default="train.csv", help='train_csv_file_name')
    parser.add_argument('--valid_csv_file_name', type=str, default="valid.csv", help='valid_csv_file_name')
    parser.add_argument('--seed', type=int, default="0", help='seed')
    parser.add_argument('--channels', default=3, type=int, help='channels ')
    parser.add_argument('--concept_threshold', default=0.7, type=int, help='concept_threshold')
    parser.add_argument('--tot_samples', type=int, required=True, help='tot_samples')
    parser.add_argument('--image_size', default=512, type=int, help='image_size.')
    parser.add_argument('--crop_size', default=512, type=int, help='image_size.')
    parser.add_argument('--uncertain', default=1, type=int, help='uncertain.')
    # Technical details
    parser.add_argument('--pool1', metavar='ARCH', default='average',
                        help='type of pooling layer for net1. the options are: average, max, log-sum-exp')
    parser.add_argument('--pool2', metavar='ARCH', default='average',
                        help='type of pooling layer for net2. the options are: average, max, log-sum-exp')
    parser.add_argument('--pretrained', type=bool, default=True, help='pretrained imagenet')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--is-parallel', default=False, type=str2bool,
                        help='use data parallel', metavar='BOOL')
    parser.add_argument('--checkpoint-epochs', default=1, type=int,
                        metavar='EPOCHS',
                        help='checkpoint frequency in epochs, 0 to turn checkpointing off (default: 1)')
    parser.add_argument('-g', '--gpu', default=0, type=int, metavar='N',
                        help='gpu number (default: 0)')

    # Optimization
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='max learning rate')
    parser.add_argument('--loss', default="mse", type=str, metavar='TYPE',
                        choices=['soft'])
    parser.add_argument('--optim', default="sgd", type=str, metavar='TYPE',
                        choices=['sgd', 'adam'])
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='use nesterov momentum', metavar='BOOL')

    # LR schecular
    parser.add_argument('--lr-scheduler', default="cos", type=str, metavar='TYPE',
                        choices=['cos', 'multistep', 'none'])
    parser.add_argument('--min-lr', '--minimum-learning-rate', default=1e-7, type=float,
                        metavar='LR', help='minimum learning rate')

    parser.add_argument('--gamma', default=0.1, type=float,
                        help='factor of learning rate decay')

    # Pseudo-Label
    parser.add_argument('--T1', default=100, type=float, metavar='M',
                        help='T1')
    parser.add_argument('--T2', default=600, type=float, metavar='M',
                        help='T1')
    parser.add_argument('--af', default=0.3, type=float, metavar='M',
                        help='af')
    parser.add_argument(
        '--layer', type=str, default="features_denseblock4", help='features_denseblock3 or features_denseblock4 for phi'
    )
    parser.add_argument('--flattening-type', type=str, default="flatten", help='flatten or adaptive or maxpool')
    parser.add_argument('--source-checkpoint-bb', metavar='file', required=True, help='checkpoint file of BB')
    parser.add_argument(
        '--source-checkpoint-bb_path', metavar='file', default="lr_0.01_epochs_60_loss_CE", required=True,
        help='checkpoint file of BB'
    )
    parser.add_argument('--source-checkpoint-t', metavar='file', required=True, help='checkpoint file of t of mimic')
    parser.add_argument('--source-checkpoint-t-path', type=str, required=True,
                        default="lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4",
                        help='dataset folder of concepts')

    parser.add_argument('--landmark-names-spec', nargs='+',
                        default=['trachea', 'left_hilar', 'right_hilar', 'hilar_unspec', 'left_pleural',
                                 'right_pleural', 'pleural_unspec', 'heart_size', 'heart_border', 'left_diaphragm',
                                 'right_diaphragm', 'diaphragm_unspec', 'retrocardiac', 'lower_left_lobe',
                                 'upper_left_lobe',
                                 'lower_right_lobe', 'middle_right_lobe', 'upper_right_lobe', 'left_lower_lung',
                                 'left_mid_lung', 'left_upper_lung',
                                 'left_apical_lung', 'left_lung_unspec', 'right_lower_lung', 'right_mid_lung',
                                 'right_upper_lung', 'right_apical_lung',
                                 'right_lung_unspec', 'lung_apices', 'lung_bases', 'left_costophrenic',
                                 'right_costophrenic', 'costophrenic_unspec',
                                 'cardiophrenic_sulcus', 'mediastinal', 'spine', 'rib', 'right_atrium',
                                 'right_ventricle',
                                 'aorta', 'svc',
                                 'interstitium', 'parenchymal', 'cavoatrial_junction', 'stomach', 'clavicle'])
    parser.add_argument('--abnorm-obs-concepts', nargs='+',
                        default=['effusion', 'opacity', 'edema', 'atelectasis', 'tube', 'consolidation',
                                 'process', 'abnormality', 'enlarge', 'tip', 'low',
                                 'line', 'congestion', 'catheter', 'cardiomegaly', 'fracture', 'air',
                                 'tortuous', 'lead', 'disease', 'calcification', 'prominence',
                                 'device', 'engorgement', 'picc', 'clip', 'elevation', 'nodule', 'wire', 'fluid',
                                 'degenerative', 'pacemaker', 'thicken', 'marking', 'scar',
                                 'blunt', 'loss', 'widen', 'collapse', 'density', 'emphysema', 'aerate', 'mass',
                                 'crowd',
                                 'infiltrate', 'obscure', 'deformity', 'hernia',
                                 'drainage', 'distention', 'shift', 'stent', 'pressure', 'lesion', 'finding',
                                 'borderline',
                                 'hardware', 'dilation', 'chf', 'redistribution', 'aspiration'])

    parser.add_argument('--train_phi', type=str, default="n", help='flatten or adaptive or maxpool')
    return parser.parse_args()


if __name__ == '__main__':
    args = config()
    test_t(args)
