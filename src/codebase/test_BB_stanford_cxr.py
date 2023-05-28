import argparse

from BB.experiments_BB_cxrs import train, test

parser = argparse.ArgumentParser(description='PyTorch Chest-Xray Training')

parser.add_argument('--exp-dir', metavar='DIR',
                    default='/ocean/projects/asc170022p/yuke/PythonProject/MIMIC-CXR-SSL/exp/mimic_cxr/debug',
                    help='path to images')
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
                    help='path to output logs')

parser.add_argument('--chexpert_names', nargs='+',
                    default=["No_Finding", "Enlarged_Cardiomediastinum", "Cardiomegaly", "Lung_Opacity", "Lung_Lesion",
                             "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Effusion",
                             "Pleural_Other", "Fracture", "Support Devices"]
                    )

parser.add_argument('--full-anatomy-names', nargs='+',
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
                             'cardiophrenic_sulcus', 'mediastinal', 'spine', 'clavicle', 'rib', 'stomach',
                             'right_atrium', 'right_ventricle', 'aorta', 'svc',
                             'interstitium', 'parenchymal', 'cavoatrial_junction', 'cardiopulmonary', 'pulmonary',
                             'lung_volumes', 'unspecified', 'other'])
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
                             'cardiophrenic_sulcus', 'mediastinal', 'spine', 'rib', 'right_atrium', 'right_ventricle',
                             'aorta', 'svc',
                             'interstitium', 'parenchymal', 'cavoatrial_junction', 'stomach', 'clavicle'])
parser.add_argument('--landmark-names-unspec', nargs='+',
                    default=['cardiopulmonary', 'pulmonary', 'lung_volumes', 'unspecified', 'other'])
parser.add_argument('--full-obs', nargs='+',
                    default=['normal', 'clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free',
                             'effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis', 'tube', 'consolidation',
                             'process', 'abnormality', 'enlarge', 'tip', 'low',
                             'pneumonia', 'line', 'congestion', 'catheter', 'cardiomegaly', 'fracture', 'air',
                             'tortuous', 'lead', 'disease', 'calcification', 'prominence',
                             'device', 'engorgement', 'picc', 'clip', 'elevation', 'expand', 'nodule', 'wire', 'fluid',
                             'degenerative', 'pacemaker', 'thicken', 'marking', 'scar',
                             'hyperinflate', 'blunt', 'loss', 'widen', 'collapse', 'density', 'emphysema', 'aerate',
                             'mass', 'crowd', 'infiltrate', 'obscure', 'deformity', 'hernia',
                             'drainage', 'distention', 'shift', 'stent', 'pressure', 'lesion', 'finding', 'borderline',
                             'hardware', 'dilation', 'chf', 'redistribution', 'aspiration',
                             'tail_abnorm_obs', 'excluded_obs'])
parser.add_argument('--norm-obs', nargs='+',
                    default=['normal', 'clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free',
                             'expand', 'hyperinflate'])
parser.add_argument('--abnorm-obs', nargs='+',
                    default=['effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis', 'tube', 'consolidation',
                             'process', 'abnormality', 'enlarge', 'tip', 'low',
                             'pneumonia', 'line', 'congestion', 'catheter', 'cardiomegaly', 'fracture', 'air',
                             'tortuous', 'lead', 'disease', 'calcification', 'prominence',
                             'device', 'engorgement', 'picc', 'clip', 'elevation', 'nodule', 'wire', 'fluid',
                             'degenerative', 'pacemaker', 'thicken', 'marking', 'scar',
                             'blunt', 'loss', 'widen', 'collapse', 'density', 'emphysema', 'aerate', 'mass', 'crowd',
                             'infiltrate', 'obscure', 'deformity', 'hernia',
                             'drainage', 'distention', 'shift', 'stent', 'pressure', 'lesion', 'finding', 'borderline',
                             'hardware', 'dilation', 'chf', 'redistribution', 'aspiration'])
parser.add_argument('--tail-abnorm-obs', nargs='+', default=['tail_abnorm_obs'])
parser.add_argument('--excluded-obs', nargs='+', default=['excluded_obs'])
parser.add_argument('--labels', nargs='+', default=['0 (No Pneumothorax)', '1 (Pneumothorax)'])
parser.add_argument('--selected-obs', nargs='+', default=['pneumothorax'])
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
parser.add_argument('--channels', default=3, type=int, help='channels ')
parser.add_argument('--pretrained', type=bool, default=True, help='pretrained imagenet')
parser.add_argument('--dataset', type=str, default="stanford_cxr", help='dataset')
parser.add_argument('--train_csv_file_name', type=str, default="train.csv", help='train_csv_file_name')
parser.add_argument('--valid_csv_file_name', type=str, default="valid.csv", help='valid_csv_file_name')
parser.add_argument('--uncertain', default=1, type=int, help='number of epochs warm up.')
parser.add_argument('--batch_size', default=8, type=int, help='batch_size.')
parser.add_argument('--image_size', default=512, type=int, help='image_size.')
parser.add_argument('--crop_size', default=512, type=int, help='image_size.')
parser.add_argument('--resize', default=512, type=int, help='number of epochs warm up.')
parser.add_argument('--arch', metavar='ARCH', default='densenet121', help='PyTorch image models')
parser.add_argument('--pool1', metavar='ARCH', default='average',
                    help='type of pooling layer for net1. the options are: average, max, log-sum-exp')
parser.add_argument('--pool2', metavar='ARCH', default='average',
                    help='type of pooling layer for net2. the options are: average, max, log-sum-exp')
parser.add_argument('--lr', default=0.01, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--epochs', default=5, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--loss', default='CE', help='observation loss type.')
parser.add_argument('--weight-decay', default=0.0001, type=float, metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--checkpoint-stanford', metavar='file', default='best_model_epoch_xx.pth.tar',
                    help='checkpoint file of stanford cxr')
parser.add_argument('--checkpoint-mimic-cxr', metavar='file', default='best_model_epoch_xx.pth.tar',
                    help='checkpoint file of mimic-cxr')


def main():
    print("Train BB for STANFORD_CXR")
    args = parser.parse_args()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    test(args)


if __name__ == '__main__':
    main()
