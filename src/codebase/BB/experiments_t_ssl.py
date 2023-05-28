import os
import pickle
import random
import time
import warnings
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from BB.models.BB_DenseNet121 import DenseNet121
from BB.models.t import Logistic_Regression_t
from BB.models.t_domain_transfer import T_Domain_Transfer
from Logger.logger_mimic_cxr import Logger_MIMIC_CXR
from dataset.dataset_stanford_cxr import Stanford_CXR_T_SSL, Stanford_CXR

warnings.filterwarnings("ignore")
from pathlib import Path


def setup(args, mode):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    batch_size = 0
    train_shuffle = False
    if mode == "train":
        batch_size = args.batch_size
        train_shuffle = True
    elif mode == "test":
        batch_size = 1
        train_shuffle = False

    args.N_labels = 2
    root = f"lr_{args.lr}_epochs_{args.epochs}_loss_BCE_W_flattening_type_{args.flattening_type}_layer_{args.layer}"
    chk_pt_path_bb_mimic = Path(
        f"{args.checkpoints}/{args.source_dataset}/BB/{args.source_checkpoint_bb_path}/{args.arch}/{args.disease}/{args.source_checkpoint_bb}"
    )
    chk_pt_path_t_mimic = Path(
        f"{args.checkpoints}/{args.source_dataset}/t/{args.source_checkpoint_t_path}/{args.arch}/{args.disease}/{args.source_checkpoint_t}"
    )

    output_path = Path(
        f"{args.output}/{args.target_dataset}/t/{root}/{args.arch}/sample_size_{args.tot_samples}_train_phi_{args.train_phi}/{args.disease}"
    ) if args.train_phi == "y" else Path(
        f"{args.output}/{args.target_dataset}/t/{root}/{args.arch}/sample_size_{args.tot_samples}/{args.disease}"
    )
    tb_logs_path_t = Path(
        f"{args.logs}/{args.target_dataset}/t/{root}/{args.arch}/sample_size_{args.tot_samples}_train_phi_{args.train_phi}/{args.disease}"
    ) if args.train_phi == "y" else Path(
        f"{args.logs}/{args.target_dataset}/t/{root}/{args.arch}/sample_size_{args.tot_samples}/{args.disease}"
    )
    chk_pt_path_t = Path(
        f"{args.checkpoints}/{args.target_dataset}/t/{root}/{args.arch}/sample_size_{args.tot_samples}_train_phi_{args.train_phi}/{args.disease}"
    ) if args.train_phi == "y" else Path(
        f"{args.checkpoints}/{args.target_dataset}/t/{root}/{args.arch}/sample_size_{args.tot_samples}/{args.disease}"
    )

    os.makedirs(tb_logs_path_t, exist_ok=True)
    os.makedirs(chk_pt_path_t, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    device = utils.get_device()
    print(f"Device: {device}")

    print("##################### Paths #####################")
    print(tb_logs_path_t)
    print(chk_pt_path_t)
    print(output_path)
    print("#################################################")

    bb_mimic = DenseNet121(args, layer=args.layer).cuda()
    bb_mimic.load_state_dict(torch.load(chk_pt_path_bb_mimic)['state_dict'])
    bb_mimic.eval()

    # dataset_path = Path(f"{args.output}/mimic_cxr/t/{args.dataset_folder_concepts}/{args.arch}/{args.disease}/dataset_g")
    # concept_names = pickle.load(open(os.path.join(dataset_path, f"selected_concepts_{args.metric}.pkl"), "rb"))

    t_mimic = Logistic_Regression_t(
        ip_size=get_input_size_t(args, bb_mimic), op_size=len(args.landmark_names_spec) + len(args.abnorm_obs_concepts),
        flattening_type=args.flattening_type
    ).cuda()
    t_mimic.load_state_dict(torch.load(chk_pt_path_t_mimic)['state_dict'])
    t_mimic.eval()

    if args.train_phi == "n":
        print("********************** Training only T ********************** ")
        t_new = Logistic_Regression_t(
            ip_size=get_input_size_t(args, bb_mimic),
            op_size=len(args.landmark_names_spec) + len(args.abnorm_obs_concepts),
            flattening_type=args.flattening_type
        ).cuda()
    elif args.train_phi == "y":
        print("********************** Training Phi and T both ********************** ")
        t_new = T_Domain_Transfer(
            args, chk_pt_path_bb_mimic, ip_size=get_input_size_t(args, bb_mimic),
            op_size=len(args.landmark_names_spec) + len(args.abnorm_obs_concepts)
        ).cuda()

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        normalize
    ])
    val_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        normalize
    ])

    args.csv_file = f"master_tot_{args.tot_samples}.csv"
    args.dataset = args.target_dataset
    train_dataset = Stanford_CXR_T_SSL(args=args, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=train_shuffle, num_workers=args.workers, pin_memory=True
    )

    val_dataset = Stanford_CXR(args, args.valid_csv_file_name, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return chk_pt_path_t, tb_logs_path_t, output_path, bb_mimic, t_mimic, t_new, train_loader, val_loader


def train_t(args):
    start = time.time()
    chk_pt_path_t, tb_logs_path_t, output_path, bb_mimic, t_mimic, t_new, train_loader, val_loader = setup(args,
                                                                                                           mode="train")
    done = time.time()
    elapsed = done - start
    print("Time to setup: " + str(elapsed) + " secs")

    if args.train_phi == "n":
        optimizer = torch.optim.SGD(
            [
                {'params': list(t_new.linear.parameters()), 'lr': args.lr,
                 'weight_decay': args.weight_decay, 'momentum': args.momentum
                 }
            ])
    elif args.train_phi == "y":
        optimizer = torch.optim.SGD(
            [
                {'params': list(t_new.features.parameters()), 'lr': args.lr,
                 'weight_decay': args.weight_decay, 'momentum': args.momentum},
                {'params': list(t_new.linear.parameters()), 'lr': args.lr,
                 'weight_decay': args.weight_decay, 'momentum': args.momentum}
            ]
        )

    final_parameters = OrderedDict(
        arch=[args.arch],
        dataset=[args.target_dataset],
        now=[datetime.today().strftime('%Y-%m-%d-%HH-%MM-%SS')]
    )
    scheduler = utils.create_lr_scheduler(optimizer, args)

    run_id = utils.get_runs(final_parameters)[0]
    run_manager = Logger_MIMIC_CXR(
        1, 0, args.start_epoch, chk_pt_path_t, tb_logs_path_t, output_path, train_loader, val_loader,
        len(args.landmark_names_spec) + len(args.abnorm_obs_concepts), model_type="t"
    )

    run_manager.begin_run(run_id)
    for epoch in range(args.epochs):
        run_manager.begin_epoch()
        t_new.train()
        if scheduler is not None:
            scheduler.step()
        with tqdm(total=len(train_loader)) as t:
            for i, data in enumerate(train_loader):
                (image, disease_label, image_names) = data
                bs = image.size(0)
                if args.gpu is not None:
                    image = image.cuda(args.gpu, non_blocking=True)
                    disease_label = disease_label.to(torch.int32).cuda(args.gpu, non_blocking=True)
                features, _, _ = bb_mimic(image)
                proba_concepts_mimic = torch.sigmoid(t_mimic(features))
                target_concepts_sup = (proba_concepts_mimic > 0.5).float()
                proba_concept_sup = None
                if args.train_phi == "n":
                    proba_concept_sup = torch.sigmoid(t_new(features))
                elif args.train_phi == "y":
                    proba_concept_sup = torch.sigmoid(t_new(image))
                mask_sup = (~disease_label.eq(-1)).float().view(-1, 1)
                n_sup = torch.sum(mask_sup)
                if n_sup > 0:
                    loss_sup = torch.sum(
                        (
                                mask_sup.repeat(1, target_concepts_sup.size(1)) *
                                F.binary_cross_entropy(proba_concept_sup, target_concepts_sup, reduction='none')
                        )
                    ) / (n_sup * target_concepts_sup.size(1))
                else:
                    loss_sup = 0
                with torch.no_grad():
                    target_concepts_pseudo = (proba_concept_sup > 0.5).float()
                mask_unsup = disease_label.eq(-1).float().view(-1, 1)
                n_unsup = torch.sum(mask_unsup)

                if n_unsup > 0:
                    loss_unsup = torch.sum(
                        (
                                mask_unsup.repeat(1, target_concepts_pseudo.size(1)) *
                                F.binary_cross_entropy(proba_concept_sup, target_concepts_pseudo, reduction='none')
                        )
                    ) / (n_unsup * target_concepts_sup.size(1))
                else:
                    loss_unsup = 0

                train_loss = loss_sup + unlabeled_weight(args, epoch) * loss_unsup
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                run_manager.track_train_loss(train_loss.item())
                run_manager.track_total_train_correct_multilabel_per_epoch(proba_concept_sup, target_concepts_sup)
                t.set_postfix(
                    epoch='{0}'.format(epoch),
                    training_loss='{:05.3f}'.format(run_manager.epoch_train_loss))
                t.update()

        t_new.eval()
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for i, data in enumerate(val_loader):
                    (image, disease_label, image_names) = data
                    if args.gpu is not None:
                        image = image.cuda(args.gpu, non_blocking=True)
                        disease_label = disease_label.to(torch.int32).cuda(args.gpu, non_blocking=True)

                    features, _, _ = bb_mimic(image)
                    proba_concepts_mimic = torch.sigmoid(t_mimic(features))
                    target_concepts_sup = (proba_concepts_mimic > 0.5).float()
                    proba_concept_sup = None
                    if args.train_phi == "n":
                        proba_concept_sup = torch.sigmoid(t_new(features))
                    elif args.train_phi == "y":
                        proba_concept_sup = torch.sigmoid(t_new(image))

                    val_loss = F.binary_cross_entropy(proba_concept_sup, target_concepts_sup, reduction='mean')
                    run_manager.track_val_loss(val_loss.item())
                    run_manager.track_total_val_correct_multilabel_per_epoch(proba_concept_sup, target_concepts_sup)
                    run_manager.track_val_bb_outputs(out_class=proba_concept_sup, val_y=target_concepts_sup)
                    t.set_postfix(
                        epoch='{0}'.format(epoch),
                        validation_loss='{:05.3f}'.format(run_manager.epoch_val_loss)
                    )
                    t.update()

        run_manager.end_epoch(t_new, optimizer, multi_label=True)
        print(f"Epoch: [{epoch + 1}/{args.epochs}] "
              f"Train_loss: {round(run_manager.get_final_train_loss(), 4)} "
              f"Train_Accuracy: {round(run_manager.get_final_train_accuracy(), 4)} (%) "
              f"Val_loss: {round(run_manager.get_final_val_loss(), 4)} "
              f"Best_auroc: {round(run_manager.best_auroc, 4)} (0-1) "
              f"Val_Accuracy: {round(run_manager.val_accuracy, 4)} (%)  "
              f"Val_AUROC: {round(run_manager.val_auroc, 4)} (0-1) "
              f"Val_AURPC: {round(run_manager.val_aurpc, 4)} (0-1) "
              f"Epoch_Duration: {round(run_manager.get_epoch_duration(), 4)} secs")

    run_manager.end_run()


def test_t(args):
    start = time.time()
    chk_pt_path_t, tb_logs_path_t, output_path, bb, t_mimic, t_new, train_loader, val_loader = setup(args, mode="test")
    done = time.time()
    elapsed = done - start
    print("Time to setup: " + str(elapsed) + " secs")
    t_new.load_state_dict(torch.load(chk_pt_path_t / "best_model.pth.tar")['state_dict'])
    t_new.eval()
    validate_and_save(args, val_loader, t_mimic, t_new, bb, output_path, mode="test")
    validate_and_save(args, train_loader, t_mimic, t_new, bb, output_path, mode="train")


def validate_and_save(args, loader, t_mimic, t_new, bb, output_path, mode):
    proba_concepts_x = torch.FloatTensor().cuda()
    attributes = torch.FloatTensor().cuda()
    with torch.no_grad():
        with tqdm(total=len(loader)) as t:
            for i, data in enumerate(loader):
                (image, disease_label, image_names) = data
                if args.gpu is not None:
                    image = image.cuda(args.gpu, non_blocking=True)
                    disease_label = disease_label.to(torch.int32).cuda(args.gpu, non_blocking=True)
                features, _, _ = bb(image)
                proba_concepts_mimic = torch.sigmoid(t_mimic(features))
                target_concepts_sup = (proba_concepts_mimic > 0.5).float()
                proba_concept_sup = None
                if args.train_phi == "n":
                    proba_concept_sup = torch.sigmoid(t_new(features))
                elif args.train_phi == "y":
                    proba_concept_sup = torch.sigmoid(t_new(image))

                attributes = torch.cat((attributes, target_concepts_sup), dim=0)
                proba_concepts_x = torch.cat((proba_concepts_x, proba_concept_sup), dim=0)

                t.set_postfix(batch_id='{0}'.format(i))
                t.update()

    proba_concepts_x = proba_concepts_x.cpu()
    attributes = attributes.cpu()
    print(f"====> Saved proba concepts_x size: {proba_concepts_x.size()}")
    print(f"====> Saved attributes size: {attributes.size()}")

    utils.save_tensor(
        path=output_path / f"{mode}_sample_size_{args.tot_samples}_attributes.pt", tensor_to_save=attributes
    )
    utils.save_tensor(
        path=output_path / f"{mode}_sample_size_{args.tot_samples}_proba_concepts.pt", tensor_to_save=proba_concepts_x
    )

    all_concepts = args.landmark_names_spec + args.abnorm_obs_concepts
    out_auroc = utils.compute_AUROC(attributes, proba_concepts_x, len(all_concepts))

    auroc_mean = np.array(out_auroc).mean()
    print("<<< Model Test Results: AUROC >>>")
    print("MEAN", ": {:.4f}".format(auroc_mean))

    chk_pt_path_t_mimic = Path(
        f"{args.output}/{args.source_dataset}/t/{args.source_checkpoint_t_path}/{args.arch}/{args.disease}/dataset_g"
    )
    with open(os.path.join(chk_pt_path_t_mimic, "selected_concepts_auroc.pkl"), "rb") as input_file:
        concepts = pickle.load(input_file)
    print(f"Sub selected concepts from MIMIC-CXR: {len(concepts)}")
    idx_list = []
    for concept in concepts:
        idx_list.append(all_concepts.index(concept))

    attributes_sub_select = attributes[:, idx_list]
    proba_concepts_x_sub_select = proba_concepts_x[:, idx_list]
    utils.save_tensor(
        path=output_path / f"{mode}_sample_size_{args.tot_samples}_sub_select_attributes.pt",
        tensor_to_save=attributes_sub_select
    )
    utils.save_tensor(
        path=output_path / f"{mode}_sample_size_{args.tot_samples}_sub_select_proba_concepts.pt",
        tensor_to_save=proba_concepts_x_sub_select
    )


def get_input_size_t(args, model_bb):
    if args.arch == "densenet121":
        if args.flattening_type == "flatten":
            return (
                       model_bb.fc1.weight.shape[1] if args.layer == "features_denseblock4" else
                       int(model_bb.fc1.weight.shape[1] / 2)
                   ) * 16 * 16
        elif args.flattening_type == "adaptive":
            return model_bb.fc1.weight.shape[1] if args.layer == "features_denseblock4" \
                else int(model_bb.fc1.weight.shape[1] / 2)
        elif args.flattening_type == "max_pool":
            return model_bb.fc1.weight.shape[1] if args.layer == "features_denseblock4" \
                else int(model_bb.fc1.weight.shape[1] / 2)
    elif args.arch == "ViT-B_32_densenet":
        # return 1024
        return 17 * 1024


def unlabeled_weight(args, cur_epoch):
    alpha = 0.0
    if cur_epoch > args.T1:
        alpha = (cur_epoch - args.T1) / (args.T2 - args.T1) * args.af
        if cur_epoch > args.T2:
            alpha = args.af
    return alpha
