import ast
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.nn.functional import one_hot
from torch.utils.data import Dataset

from dataset.utils_mimic_cxr import get_aug_bbox


class MIMICCXRDataset(Dataset):

    def __init__(
            self, args, radgraph_sids, radgraph_adj_mtx, mode, transform=None, model_type="bb",
            file_name_concept=None, file_name_y=None, network_type="CNN", feature_path="None"
    ):

        self.network_type = network_type
        self.feature_path = feature_path
        self.args = args
        self.mode = mode
        self.transform = transform
        self.model_type = model_type
        self.file_name_concept = file_name_concept
        self.file_name_y = file_name_y
        # read master dataset
        df_master = pd.read_csv(self.args.img_chexpert_file)
        df_master['study_id'] = df_master['study_id'].apply(str)

        # read NVIDIA disease bbox
        df_bbox = pd.read_csv(self.args.nvidia_bounding_box_file)

        # read ImaGenome landmark bbox
        if self.mode == 'bbox':
            self.dict_imagenome_bbox = pickle.load(
                open(self.args.imagenome_bounding_box_file + 'bbox_imagenome_bbox.pkl', "rb"))
        else:
            self.dict_imagenome_bbox = pickle.load(
                open(self.args.imagenome_bounding_box_file + 'all_imagenome_bbox.pkl', "rb"))
        self.dict_landmark_mapping = json.load(open(self.args.imagenome_radgraph_landmark_mapping_file))

        # read radgraph landmark PNU labels
        self.arr_radgraph_sids = radgraph_sids  # N_sids
        self.arr_radgraph_adj = radgraph_adj_mtx  # N_sids * 51 * 75

        # create random splits 80%, 10%, 10%
        subject_ids = np.array(df_master['subject_id'].unique())  # ~65K patients
        np.random.seed(0)
        np.random.shuffle(subject_ids)
        k1 = int(len(subject_ids) * 0.8)
        k2 = int(len(subject_ids) * 0.9)
        self.train_subject_ids = list(subject_ids[:k1])
        self.valid_subject_ids = list(subject_ids[k1:k2])
        self.test_subject_ids = list(subject_ids[k2:])

        # dataset
        idx1 = df_master['ViewPosition'].isin(['AP', 'PA'])  # frontal view CXR only
        idx2 = df_master['study_id'].isin(list(self.arr_radgraph_sids))  # must have RadGraph labels
        idx3 = df_master['dicom_id'].isin(list(df_bbox['dicom_id'].unique()))  # leave NVIDIA bbox out for evaluation
        if self.mode == 'train':
            idx4 = df_master['subject_id'].isin(self.train_subject_ids)  # is in training dataset
            idx = idx1 & idx2 & (~idx3) & idx4
        elif self.mode == 'valid':
            idx4 = df_master['subject_id'].isin(self.valid_subject_ids)  # is in validate dataset
            idx = idx1 & idx2 & (~idx3) & idx4
        elif self.mode == 'test':
            idx4 = df_master['subject_id'].isin(self.test_subject_ids)  # is in test dataset
            idx = idx1 & idx2 & (~idx3) & idx4
        elif self.mode == 'bbox':
            idx = idx1 & idx2 & idx3
        else:
            raise Exception('Invalid split mode.')

        # selected master dataset
        self.df_master_sel = df_master[idx]

        self.dicom_ids = list(self.df_master_sel['dicom_id'].unique())
        self.study_ids = list(self.df_master_sel['study_id'].unique())

        # define self.df_bbox, make sure its dicom id is in the selected master table
        self.df_bbox = df_bbox[df_bbox['dicom_id'].isin(self.dicom_ids)]
        # convert bbox from string to list
        self.df_bbox['bbox'] = self.df_bbox['bbox'].apply(ast.literal_eval)
        self.df_bbox = self.df_bbox.groupby(['dicom_id', 'name', 'report', 'label_text'])['bbox'].agg(
            lambda x: +x).reset_index()
        self.df_bbox['bbox'] = self.df_bbox['bbox'].apply(np.array)
        self.df_bbox['bbox'] = self.df_bbox['bbox'].apply(lambda x: x.tolist())

        # truncate data to only a subset for debugging
        if self.args.mini_data is not None:
            self.dicom_ids = self.dicom_ids[:self.args.mini_data]

        # column index of chexpert labels
        self.attr_idxs = [self.df_master_sel.columns.tolist().index(a) for a in self.args.chexpert_names]
        # attr columns ['No Finding', ..., 'Support Devices']; note AP/PA remains with NAs for Lateral pictures
        self.df_master_sel[self.args.chexpert_names] = self.df_master_sel[self.args.chexpert_names].fillna(0)
        # fill -1 as 0 (U-zeros)
        self.df_master_sel[self.args.chexpert_names] = self.df_master_sel[self.args.chexpert_names].replace(-1, 0)

        # get specified landmark index in the adjacent matrix
        self.landmark_spec_idx = [self.args.full_anatomy_names.index(x) for x in self.args.landmark_names_spec]
        self.landmark_unspec_idx = [self.args.full_anatomy_names.index(x) for x in self.args.landmark_names_unspec]

        # get norm_obs, abnorm_obs, tail_abnorm_obs, excluded_obs index in the adjacent matrix
        self.norm_obs_idx = [self.args.full_obs.index(x) for x in self.args.norm_obs]
        self.abnorm_obs_idx = [self.args.full_obs.index(x) for x in self.args.abnorm_obs]
        self.tail_abnorm_obs_idx = [self.args.full_obs.index(x) for x in self.args.tail_abnorm_obs]
        self.excluded_obs_idx = [self.args.full_obs.index(x) for x in self.args.excluded_obs]
        self.selected_obs_idx = [self.args.full_obs.index(x) for x in self.args.selected_obs]

        # get the initial landmark (spec.) labels (landmark is abnormal only when abnorm_obs or tail_abnorm_obs = 1.0)
        self.landmark_spec_label_org = self.arr_radgraph_adj[:, self.landmark_spec_idx, :]
        self.landmark_spec_label_org = self.landmark_spec_label_org[:, :, (
                                                                                  self.abnorm_obs_idx + self.tail_abnorm_obs_idx)]  # landmark is treated abnormal when any abnormal obs present, including devices
        # self.landmark_spec_label_org = self.landmark_spec_label_org[:, :, self.selected_obs_idx] # landmakr is treated abnorm only when selected obs present, excluding devices
        self.landmark_spec_label_org = self.landmark_spec_label_org.max(axis=2)  # N_sids * 46
        self.landmark_spec_label_imputed = self.landmark_spec_label_org.copy()
        self.landmark_spec_label_imputed[self.landmark_spec_label_imputed == -1] = 0

        # get the initial observation (abnorm) labels (observation is present on the image, if any anatomical landmark (spec, or unspec) = 1.0)
        self.selected_obs_label_org = self.arr_radgraph_adj[:, :, self.selected_obs_idx].max(axis=1)  # N_sids * 65
        self.selected_obs_label_imputed = self.selected_obs_label_org.copy()
        self.selected_obs_label_imputed[self.selected_obs_label_imputed == -1] = 0

        # get the adj-mtx that for spec. landmark and abnorm obs
        self.adj_mtx_spec_abnorm_org = self.arr_radgraph_adj[:, self.landmark_spec_idx, :]
        self.adj_mtx_spec_abnorm_org = self.adj_mtx_spec_abnorm_org[:, :, self.selected_obs_idx]  # N_sides * 46 * 65
        self.adj_mtx_spec_abnorm_imputed = self.adj_mtx_spec_abnorm_org
        self.adj_mtx_spec_abnorm_imputed[self.adj_mtx_spec_abnorm_imputed == -1] = 0

        # compute label weights
        self.chexpert_weights = torch.FloatTensor(self.df_master_sel[self.args.chexpert_names].mean(axis=0))  # 14
        self.landmark_spec_imputed_weights = torch.FloatTensor(self.landmark_spec_label_imputed.mean(axis=0))  # 46
        self.selected_obs_imputed_weights = torch.FloatTensor(self.selected_obs_label_imputed.mean(axis=0))  # 65

        # compute label PNU weights
        obs_p_idx = self.selected_obs_label_org == 1
        self.selected_obs_p_weights = torch.tensor(obs_p_idx.sum(axis=0) / obs_p_idx.shape[0])
        obs_u_idx = self.selected_obs_label_org == -1
        self.selected_obs_u_weights = torch.tensor(obs_u_idx.sum(axis=0) / obs_u_idx.shape[0])
        self.mode = "val" if mode == "valid" else mode
        if model_type == "t":
            self.abnorm_obs_concepts_idx = [self.args.full_obs.index(x) for x in self.args.abnorm_obs_concepts]
            self.full_obs_label_org = self.arr_radgraph_adj[:, :, self.abnorm_obs_concepts_idx].max(
                axis=1)  # N_sids * 65
            self.full_obs_label_imputed = self.full_obs_label_org.copy()
            self.full_obs_label_imputed[self.full_obs_label_imputed == -1] = 0
            self.full_obs_imputed_weights = torch.FloatTensor(self.full_obs_label_imputed.mean(axis=0))

            full_obs_p_idx = self.full_obs_label_org == 1
            self.full_obs_p_weights = torch.tensor(full_obs_p_idx.sum(axis=0) / full_obs_p_idx.shape[0])
            full_obs_u_idx = self.full_obs_label_org == -1
            self.full_obs_u_weights = torch.tensor(full_obs_u_idx.sum(axis=0) / full_obs_u_idx.shape[0])
        elif model_type == "g":
            self.concepts = torch.load(file_name_concept)
            self.y = torch.load(file_name_y)
            self.y_one_hot = one_hot(self.y.to(torch.long)).to(torch.float)

    def __getitem__(self, idx):
        # 1. get the sample key, i.e., DICOM ID
        dicom_id = self.dicom_ids[idx]
        study_id = list(self.df_master_sel.loc[self.df_master_sel['dicom_id'] == dicom_id, 'study_id'])[0]
        idx_sid = list(self.arr_radgraph_sids).index(str(study_id))

        # 2. select and load image
        img_pth_old = self.df_master_sel.iloc[idx, 4]
        img_pth_components = img_pth_old.split("/")
        img_pth = os.path.join(
            self.args.image_path_ocean_shared,
            *img_pth_components[11:]
        )
        image = Image.open(img_pth).convert('RGB')
        raw_transform = transforms.Compose([
            transforms.Resize(self.args.resize),
            transforms.CenterCrop(self.args.resize),
            transforms.ToTensor()
        ])
        raw_image = raw_transform(image)
        if self.transform is not None:
            image = self.transform(image)

        # 3. extract labels
        # chexpert labels
        chexpert_label = self.df_master_sel.iloc[idx, self.attr_idxs].values.astype(np.float32)
        chexpert_label = torch.FloatTensor(chexpert_label)
        chexpert_inverse_weight = torch.where(chexpert_label == 0, self.chexpert_weights, 1 - self.chexpert_weights)

        # landmark weak labels (binary)
        landmark_spec_label = torch.FloatTensor(self.landmark_spec_label_imputed[idx_sid, :])  # 46
        landmarks_spec_inverse_weight = torch.where(landmark_spec_label == 0, self.landmark_spec_imputed_weights,
                                                    1 - self.landmark_spec_imputed_weights)
        landmark_spec_label_pnu = torch.FloatTensor(self.landmark_spec_label_org[idx_sid, :])

        # observation class
        selected_obs_label = torch.FloatTensor(self.selected_obs_label_imputed[idx_sid, :])  # 65
        selected_obs_inverse_weight = torch.where(selected_obs_label == 0, self.selected_obs_p_weights,
                                                  1 - self.selected_obs_p_weights)
        selected_obs_label_pnu = torch.FloatTensor(self.selected_obs_label_org[idx_sid, :])

        # 4. extract adjacent matrix
        adj_mtx = torch.FloatTensor(self.adj_mtx_spec_abnorm_imputed[idx_sid, :, :])

        # 5. extract NVIDIA bbox
        if self.mode == 'bbox':
            img_pth = list(self.df_master_sel[self.df_master_sel['dicom_id'] == dicom_id]['path'])[0]
            label_text = list(self.df_bbox[self.df_bbox['dicom_id'] == dicom_id]['label_text'])[0]
            disease = list(self.df_bbox[self.df_bbox['dicom_id'] == dicom_id]['name'])[0]
            bbox = list(self.df_bbox[self.df_bbox['dicom_id'] == dicom_id]['bbox'])[0]

            # get augmented bbox
            bbox_aug_lst = []
            if type(bbox[0]) == list:
                for bb_org in bbox:
                    bb_aug = get_aug_bbox(img_pth, (bb_org, 'xywh'), self.args.resize)
                    bbox_aug_lst.append(bb_aug)
            else:
                bbox_aug_lst = get_aug_bbox(img_pth, (bbox, 'xywh'), self.args.resize)

            bbox_aug = str(bbox_aug_lst)
        else:
            disease = ''
            label_text = ''
            bbox_aug = ''
            img_pth = ''

        # 6. extract ImaGenome bboxs
        landmark_bbox = np.zeros(
            (self.args.N_landmarks_spec, 4))  # ImaGenome bbox is (x1, y1, x2, y2) on a 512 * 512 image
        for k, v in self.dict_landmark_mapping.items():
            j = list(self.dict_landmark_mapping.keys()).index(k)
            l = self.args.landmark_names_spec.index(
                v)  # find the index of corresponding landmark name in defined RadGraph landmark specs
            try:
                landmark_bbox[l] = self.dict_imagenome_bbox[dicom_id][j, :]
            except:
                pass

        landmark_bbox = torch.FloatTensor(landmark_bbox)
        if self.model_type == "bb":
            return dicom_id, image, adj_mtx, \
                   chexpert_label, chexpert_inverse_weight, \
                   landmark_spec_label, landmarks_spec_inverse_weight, landmark_spec_label_pnu, \
                   selected_obs_label, \
                   selected_obs_inverse_weight, selected_obs_label_pnu, \
                   disease, label_text, bbox_aug, img_pth, \
                   landmark_bbox
        elif self.model_type == "t" and self.network_type == "VIT":
            full_obs_label = torch.FloatTensor(self.full_obs_label_imputed[idx_sid, :])  # 65
            full_obs_inverse_weight = torch.where(full_obs_label == 0, self.full_obs_p_weights,
                                                  1 - self.full_obs_p_weights)
            full_obs_label_pnu = torch.FloatTensor(self.full_obs_label_org[idx_sid, :])
            if self.args.flattening_type == "vit_flatten":
                vit_features = torch.load(os.path.join(
                    self.feature_path, f"{self.mode}_features_VIT_full", f"features_{idx}.pth.tar")
                )
            else:
                vit_features = torch.load(os.path.join(
                    self.feature_path, f"{self.mode}_features_VIT", f"features_{idx}.pth.tar")
                )
            densenet_features = torch.load(os.path.join(
                self.feature_path, f"{self.mode}_features", f"features_{idx}.pth.tar")
            )
            return dicom_id, image, vit_features, densenet_features, raw_image, adj_mtx, \
                   chexpert_label, chexpert_inverse_weight, \
                   landmark_spec_label, landmarks_spec_inverse_weight, landmark_spec_label_pnu, \
                   selected_obs_label, selected_obs_inverse_weight, selected_obs_label_pnu, \
                   full_obs_label, full_obs_inverse_weight, full_obs_label_pnu, \
                   disease, label_text, bbox_aug, img_pth, \
                   landmark_bbox

        elif self.model_type == "t" and self.network_type == "CNN":
            full_obs_label = torch.FloatTensor(self.full_obs_label_imputed[idx_sid, :])  # 65
            full_obs_inverse_weight = torch.where(full_obs_label == 0, self.full_obs_p_weights,
                                                  1 - self.full_obs_p_weights)
            full_obs_label_pnu = torch.FloatTensor(self.full_obs_label_org[idx_sid, :])
            return dicom_id, image, raw_image, raw_image, raw_image, adj_mtx, \
                   chexpert_label, chexpert_inverse_weight, \
                   landmark_spec_label, landmarks_spec_inverse_weight, landmark_spec_label_pnu, \
                   selected_obs_label, selected_obs_inverse_weight, selected_obs_label_pnu, \
                   full_obs_label, full_obs_inverse_weight, full_obs_label_pnu, \
                   disease, label_text, bbox_aug, img_pth, \
                   landmark_bbox

        elif self.model_type == "g":
            return dicom_id, image, adj_mtx, \
                   chexpert_label, chexpert_inverse_weight, \
                   landmark_spec_label, landmarks_spec_inverse_weight, landmark_spec_label_pnu, \
                   selected_obs_label, selected_obs_inverse_weight, selected_obs_label_pnu, \
                   disease, label_text, bbox_aug, img_pth, \
                   landmark_bbox, self.concepts[idx], self.y[idx], self.y_one_hot[idx]

    def __len__(self):
        return len(self.dicom_ids)


class Dataset_mimic_for_explainer(Dataset):
    def __init__(
            self,
            iteration,
            mode,
            metric,
            expert,
            dataset_path,
            bb_logits_path=None
    ):
        if iteration == 1 and expert is not "baseline":
            self.bb_logits = torch.load(os.path.join(dataset_path, f"{mode}_logits_bb.pt")).reshape(-1, 2)
        elif iteration > 1 and expert is not "baseline":
            self.bb_logits = torch.load(os.path.join(bb_logits_path, f"{mode}_out_put_preds_residual.pt"))

        self.logits_concept_x = torch.load(os.path.join(dataset_path, f"{mode}_select_logits_{metric}_concepts.pt"))
        self.attributes_gt = torch.load(os.path.join(dataset_path, f"{mode}_select_{metric}_attributes.pt"))
        self.proba_concept_x = torch.load(os.path.join(dataset_path, f"{mode}_select_proba_{metric}_concepts.pt"))
        self.y = torch.load(os.path.join(dataset_path, f"{mode}_class_labels.pt"))
        self.y_one_hot = one_hot(self.y.to(torch.long)).to(torch.float)
        self.concepts = torch.load(os.path.join(dataset_path, f"{mode}_select_{metric}_attributes.pt"))
        self.expert = expert
        self.mode = mode
        self.dataset_path = dataset_path

    def __getitem__(self, item):
        if self.expert == "explainer":
            return (
                self.bb_logits[item],
                self.logits_concept_x[item],
                self.proba_concept_x[item],
                self.attributes_gt[item],
                self.y[item],
                self.y_one_hot[item],
                self.concepts[item]
            )
        elif self.expert == "residual":
            return (
                torch.load(
                    os.path.join(
                        self.dataset_path, f"{self.mode}_transformed_images", f"transformed_img_{item}.pth.tar"
                    )
                ),
                torch.load(os.path.join(self.dataset_path, f"{self.mode}_raw_images", f"raw_img_{item}.pth.tar")),
                torch.load(os.path.join(self.dataset_path, f"{self.mode}_features", f"features_{item}.pth.tar")),
                self.bb_logits[item],
                self.logits_concept_x[item],
                self.proba_concept_x[item],
                self.attributes_gt[item],
                self.y[item],
                self.y_one_hot[item],
                self.concepts[item]
            )
        elif self.expert == "baseline":
            return (
                # self.bb_logits[item],
                self.logits_concept_x[item],
                self.proba_concept_x[item],
                self.attributes_gt[item],
                self.y[item],
                self.y_one_hot[item],
                self.concepts[item]
            )

    def __len__(self):
        return self.y.size(0)
