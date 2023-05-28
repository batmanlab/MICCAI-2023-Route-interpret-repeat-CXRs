import os
import pickle
import random
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn.metrics as sk_metrics
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm

import utils
from BB.models.BB_DenseNet121 import DenseNet121
from Explainer.profiler import fit_BB_domain_transfer_cxr_profiler
from Logger.logger_mimic_cxr import Logger_MIMIC_CXR
from dataset.dataset_nih import Nih_test_dataset, Nih_train_val_dataset
from dataset.dataset_stanford_cxr import Stanford_CXR, Stanford_CXR_Domain_transfer


def get_loaders_stanford_cxr(args, normalize, mode):
    batch_size = 0
    train_shuffle = None
    if mode == "train":
        batch_size = args.batch_size
        train_shuffle = True
    elif mode == "test":
        batch_size = 1
        train_shuffle = False
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(args.resize),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        normalize
    ])
    val_transform = transforms.Compose([
        transforms.Resize(args.resize),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        normalize
    ])

    if args.domain_transfer == "n":
        train_ds = Stanford_CXR(args, args.train_csv_file_name, transform=train_transform)
        train_data_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=train_shuffle, num_workers=4
        )
    else:
        train_ds = Stanford_CXR_Domain_transfer(args, transform=train_transform)
        train_data_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=train_shuffle, num_workers=4
        )

    val_ds = Stanford_CXR(args, args.valid_csv_file_name, transform=val_transform)
    val_data_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_data_loader, val_data_loader


def initial_setup_nih(args, mode):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if mode == "train":
        batch_size = args.batch_size
        train_shuffle = True
    elif mode == "test":
        batch_size = 1
        train_shuffle = False

    print("###############################################")
    args.N_landmarks_spec = len(args.landmark_names_spec)
    args.N_selected_obs = len(args.selected_obs)
    args.N_labels = len(args.labels)
    root = f"lr_{args.lr}_epochs_{args.epochs}_loss_{args.loss}_transform_mimic"
    if len(args.selected_obs) == 1:
        disease_folder = args.selected_obs[0]
    else:
        disease_folder = f"{args.selected_obs[0]}_{args.selected_obs[1]}"
    args.disease = disease_folder
    chk_pt_path = os.path.join(args.checkpoints, args.dataset, "BB", root, args.arch, disease_folder)
    output_path = os.path.join(args.output, args.dataset, "BB", root, args.arch, disease_folder)
    tb_logs_path = os.path.join(args.logs, args.dataset, "BB", f"{root}_{args.arch}_{disease_folder}")
    os.makedirs(chk_pt_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(tb_logs_path, exist_ok=True)

    print('******************** Paths ********************')
    print(chk_pt_path)
    print(output_path)
    print(tb_logs_path)
    print('******************** Paths ********************')

    device = utils.get_device()
    print(f"Device: {device}")
    print(output_path)
    pickle.dump(args, open(os.path.join(output_path, f"{args.dataset}_train_configs.pkl"), "wb"))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(args.resize),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        normalize
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(args.resize),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        normalize
    ])

    # transform = transforms.Compose([transforms.ToPILImage(),
    #                                 transforms.Resize(args.resize),
    #                                 transforms.ToTensor(),
    #                                 normalize])

    start = time.time()
    config = {}
    config["pkl_dir_path"] = os.path.join(args.pkl_dir_path, "pickles")
    config["test_val_dir_path"] = args.pkl_dir_path
    config["train_val_df_pkl_path"] = args.train_val_df_pkl_path
    config["test_df_pkl_path"] = args.test_df_pkl_path
    config["disease_classes_pkl_path"] = args.disease_classes_pkl_path
    config["models_dir"] = args.models_dir
    Nih_train_ds = Nih_train_val_dataset(args.image_dir_path, config, transform=train_transform)
    train_percentage = 0.8
    train_dataset, val_dataset = torch.utils.data.random_split(
        Nih_train_ds, [
            int(len(Nih_train_ds) * train_percentage),
            len(Nih_train_ds) - int(len(Nih_train_ds) * train_percentage)
        ])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    Nih_test_ds = Nih_test_dataset(args.image_dir_path, config, transform=val_transform)
    test_loader = torch.utils.data.DataLoader(Nih_test_ds, batch_size=batch_size, shuffle=False)
    print(f"Train_val size: {len(Nih_train_ds)}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Test size: {len(Nih_test_ds)}")
    done = time.time()
    elapsed = done - start
    print("Time to load the dataset: " + str(elapsed) + " secs")
    return train_loader, val_loader, test_loader, chk_pt_path, tb_logs_path, output_path, disease_folder, root


def initial_setup_stanford(args, mode):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    print("###############################################")
    args.N_landmarks_spec = len(args.landmark_names_spec)
    args.N_selected_obs = len(args.selected_obs)
    args.N_labels = len(args.labels)
    root = f"lr_{args.lr}_epochs_{args.epochs}_loss_{args.loss}"
    if len(args.selected_obs) == 1:
        disease_folder = args.selected_obs[0]
    else:
        disease_folder = f"{args.selected_obs[0]}_{args.selected_obs[1]}"
    args.disease = disease_folder

    if args.domain_transfer == "n":
        model = DenseNet121(args)
        chk_pt_path = os.path.join(args.checkpoints, args.dataset, "BB", root, args.arch, disease_folder)
        output_path = os.path.join(args.output, args.dataset, "BB", root, args.arch, disease_folder)
        tb_logs_path = os.path.join(args.logs, args.dataset, "BB", f"{root}_{args.arch}_{disease_folder}")
        os.makedirs(chk_pt_path, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(tb_logs_path, exist_ok=True)

    else:
        chk_pt_path_mimic = os.path.join(
            args.checkpoints, "mimic_cxr", "BB", "lr_0.01_epochs_60_loss_CE", args.arch, disease_folder
        )
        checkpoint_mimic = torch.load(os.path.join(chk_pt_path_mimic, args.checkpoint_mimic_cxr))
        model = DenseNet121(args)
        model.load_state_dict(checkpoint_mimic['state_dict'])
        chk_pt_path = os.path.join(args.checkpoints, args.dataset, "BB_domain_shift_w_mimic", root, args.arch,
                                   disease_folder)
        output_path = os.path.join(args.output, args.dataset, "BB_domain_shift_w_mimic", root, args.arch,
                                   disease_folder)
        tb_logs_path = os.path.join(args.logs, args.dataset, "BB_domain_shift_w_mimic",
                                    f"{root}_{args.arch}_{disease_folder}")
        os.makedirs(chk_pt_path, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(tb_logs_path, exist_ok=True)

    print('******************** Paths ********************')
    print(chk_pt_path)
    print(output_path)
    print(tb_logs_path)
    device = utils.get_device()
    print(f"Device: {device}")
    print(output_path)
    pickle.dump(args, open(os.path.join(output_path, f"{args.dataset}_train_configs.pkl"), "wb"))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    start = time.time()
    train_data_loader, val_data_loader = get_loaders_stanford_cxr(args, normalize, mode)
    done = time.time()
    elapsed = done - start
    print("Time to load the dataset: " + str(elapsed) + " secs")
    return train_data_loader, val_data_loader, model, chk_pt_path, tb_logs_path, output_path, disease_folder, root


def train(args):
    if args.dataset == "stanford_cxr":
        (
            train_data_loader, val_data_loader, model, chk_pt_path, tb_logs_path, output_path, disease_folder, root
        ) = initial_setup_stanford(args, mode="train")
    elif args.dataset == "nih":
        (
            train_data_loader, val_data_loader, test_data_loader, chk_pt_path, tb_logs_path, output_path,
            disease_folder, root
        ) = initial_setup_nih(args, mode="train")
        val_data_loader = test_data_loader
        model = DenseNet121(args)

    final_parameters = OrderedDict(
        arch=[args.arch],
        dataset=[args.dataset],
        now=[datetime.today().strftime('%Y-%m-%d-%HH-%MM-%SS')]
    )

    run_id = utils.get_runs(final_parameters)[0]
    run_manager = Logger_MIMIC_CXR(
        1, 0, 0, chk_pt_path, tb_logs_path, output_path, train_data_loader, val_data_loader, args.N_labels,
        model_type="bb"
    )
    optimizer = torch.optim.SGD(
        [{'params': list(model.backbone.parameters()), 'lr': args.lr,
          'weight_decay': args.weight_decay, 'momentum': args.momentum},
         {'params': list(model.fc1.parameters()), 'lr': args.lr,
          'weight_decay': args.weight_decay, 'momentum': args.momentum}
         ])

    if args.profile == "n":
        fit(args, model, optimizer, train_data_loader, val_data_loader, run_manager, run_id)
    else:
        fit_BB_domain_transfer_cxr_profiler(args, model, optimizer, train_data_loader, val_data_loader,)


def test(args):
    train_data_loader, val_data_loader, test_data_loader, disease_folder, output_path = None, None, None, None, None
    model_stanford = None
    if args.dataset == "stanford_cxr":
        (
            train_data_loader, val_data_loader, model_stanford, chk_pt_path, tb_logs_path, output_path, disease_folder,
            root
        ) = initial_setup_stanford(args, mode="test")
        checkpoint_stanford = torch.load(os.path.join(chk_pt_path, args.checkpoint_stanford))
        model_stanford.load_state_dict(checkpoint_stanford['state_dict'])
        model_stanford.cuda()
        model_stanford.eval()
    elif args.dataset == "nih":
        (
            train_data_loader, val_data_loader, test_data_loader, chk_pt_path, tb_logs_path, output_path,
            disease_folder, root
        ) = initial_setup_nih(args, mode="test")
        chk_pt_path_stanford = os.path.join(
            args.checkpoints, "stanford_cxr", "BB", "lr_0.01_epochs_10_loss_CE", args.arch, disease_folder
        )
        checkpoint_stanford = torch.load(os.path.join(chk_pt_path_stanford, args.checkpoint_stanford))
        model_stanford = DenseNet121(args)
        model_stanford.load_state_dict(checkpoint_stanford['state_dict'])
        model_stanford.cuda()
        model_stanford.eval()

    chk_pt_path_mimic = os.path.join(
        args.checkpoints, "mimic_cxr", "BB", "lr_0.01_epochs_60_loss_CE", args.arch, disease_folder
    )
    checkpoint_mimic = torch.load(os.path.join(chk_pt_path_mimic, args.checkpoint_mimic_cxr))
    model_mimic_cxr = DenseNet121(args)
    model_mimic_cxr.load_state_dict(checkpoint_mimic['state_dict'])
    model_mimic_cxr.cuda()
    model_mimic_cxr.eval()

    if test_data_loader is not None:
        validate(args, model_stanford, model_mimic_cxr, test_data_loader, output_path, disease_folder, mode="test")

    if train_data_loader is not None:
        validate(args, model_stanford, model_mimic_cxr, train_data_loader, output_path, disease_folder, mode="train")

    if val_data_loader is not None:
        validate(args, model_stanford, model_mimic_cxr, val_data_loader, output_path, disease_folder, mode="val")


def validate(args, model_stanford, model_mimic_cxr, loader, output_path, disease, mode):
    out_put_GT = torch.FloatTensor().cuda()
    out_put_predict_stanford = torch.FloatTensor().cuda()
    out_put_predict_mimic = torch.FloatTensor().cuda()
    idx_list = []
    image_names = []
    with torch.no_grad():
        with tqdm(total=len(loader)) as t:
            for idx, data in enumerate(loader):
                (images, labels, image_name) = data
                disease_idx = list(map(str.lower, args.chexpert_names)).index(args.disease.lower())
                labels = labels[:, disease_idx]

                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)

                if torch.cuda.is_available():
                    labels = labels.cuda(args.gpu, non_blocking=True).view(-1).to(torch.long)
                _, _, logits_stanford = model_stanford(images)
                _, _, logits_mimic = model_mimic_cxr(images)

                idx_list.append(idx)
                image_names.append(image_name[0])
                out_put_predict_stanford = torch.cat((out_put_predict_stanford, logits_stanford), dim=0)
                out_put_predict_mimic = torch.cat((out_put_predict_mimic, logits_mimic), dim=0)
                out_put_GT = torch.cat((out_put_GT, labels), dim=0)

                t.set_postfix(batch_id='{0}'.format(idx + 1))
                t.update()

    print(out_put_predict_stanford.size())
    out_put_predict_stanford = out_put_predict_stanford.reshape(-1, 2)
    out_put_predict_mimic = out_put_predict_mimic.reshape(-1, 2)
    out_put_GT_np = out_put_GT.cpu().numpy()
    out_put_stanford_np = out_put_predict_stanford.argmax(dim=1).cpu().numpy()
    out_put_mimic_np = out_put_predict_mimic.argmax(dim=1).cpu().numpy()
    proba_stanford_1 = torch.nn.Softmax(dim=1)(out_put_predict_stanford)[:, 1].cpu().numpy()
    proba_mimic_1 = torch.nn.Softmax(dim=1)(out_put_predict_mimic)[:, 1].cpu().numpy()

    auroc_stanford = sk_metrics.roc_auc_score(out_put_GT_np, proba_stanford_1)
    aurpc_stanford = sk_metrics.average_precision_score(out_put_GT_np, proba_stanford_1)
    auroc_mimic = sk_metrics.roc_auc_score(out_put_GT_np, proba_mimic_1)
    aurpc_mimic = sk_metrics.average_precision_score(out_put_GT_np, proba_mimic_1)
    print(f"************** Evaluating Stanford CXR for {disease} ***************")
    print(f"AUROC using stanford model: {auroc_stanford}")
    print(f"AURPC using stanford model: {aurpc_stanford}")
    print(f"AUROC using mimic-cxr model: {auroc_mimic}")
    print(f"AURPC using mimic-cxr model: {aurpc_mimic}")

    torch.save(out_put_GT.cpu(), os.path.join(output_path, f"{mode}_GT.pth.tar"))
    torch.save(out_put_predict_stanford.cpu(), os.path.join(output_path, f"{mode}_out_put_predict_stanford.pth.tar"))
    torch.save(out_put_predict_mimic.cpu(), os.path.join(output_path, f"{mode}_out_put_predict_mimic.pth.tar"))

    data = {
        'idx': idx_list,
        'GT': out_put_GT_np.tolist(),
        'pred_w_stanford_model': out_put_stanford_np.tolist(),
        'pred_w_mimic_model': out_put_mimic_np.tolist(),
        'proba_class_index_1_w_stanford_model': proba_stanford_1.tolist(),
        'proba_class_index_1_w_mimic_model': proba_mimic_1.tolist(),
        'image_names': image_names
    }
    data_df = pd.DataFrame.from_dict(data)
    data_df.to_csv(os.path.join(output_path, f"{mode}_results.csv"), index=False)
    print(output_path)


def fit(args, model, optimizer, train_loader, val_loader, run_manager, run_id):
    model.cuda()
    run_manager.begin_run(run_id)
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        run_manager.begin_epoch()
        # switch to train mode
        model.train()
        with tqdm(total=len(train_loader)) as t:
            for i, data in enumerate(train_loader):
                (images, labels, image_name) = data
                disease_idx = list(map(str.lower, args.chexpert_names)).index(args.disease.lower())
                if args.domain_transfer == "n":
                    labels = labels[:, disease_idx]
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)

                if torch.cuda.is_available():
                    labels = labels.cuda(args.gpu, non_blocking=True).view(-1).to(torch.long)
                features, pooled_features, logits = model(images)
                train_loss = F.cross_entropy(logits, labels, reduction='mean')

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                run_manager.track_train_loss(train_loss.item())
                run_manager.track_total_train_correct_per_epoch(logits, labels)
                t.set_postfix(
                    epoch='{0}'.format(epoch),
                    training_loss='{:05.3f}'.format(run_manager.epoch_train_loss))
                t.update()
        model.eval()
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for i, data in enumerate(val_loader):
                    (images, labels, image_name) = data
                    disease_idx = list(map(str.lower, args.chexpert_names)).index(args.disease.lower())
                    labels = labels[:, disease_idx]
                    if args.gpu is not None:
                        images = images.cuda(args.gpu, non_blocking=True)

                    if torch.cuda.is_available():
                        labels = labels.cuda(args.gpu, non_blocking=True).view(-1).to(torch.long)

                    features, pooled_features, logits = model(images)
                    val_loss = F.cross_entropy(logits, labels, reduction='mean')

                    run_manager.track_val_loss(val_loss.item())
                    run_manager.track_total_val_correct_per_epoch(logits, labels)
                    run_manager.track_val_bb_outputs(out_class=logits, val_y=labels)
                    t.set_postfix(
                        epoch='{0}'.format(epoch),
                        validation_loss='{:05.3f}'.format(run_manager.epoch_val_loss)
                    )
                    t.update()

        run_manager.end_epoch(model, optimizer, multi_label=False)

        print(f"Epoch: [{epoch + 1}/{args.epochs}] "
              f"Train_loss: {round(run_manager.get_final_train_loss(), 4)} "
              f"Train_Accuracy: {round(run_manager.get_final_train_accuracy(), 4)} (%) "
              f"Val_loss: {round(run_manager.get_final_val_loss(), 4)} "
              f"Best_Val_AUROC: {round(run_manager.best_auroc, 4)}  "
              f"Val_Accuracy: {round(run_manager.val_accuracy, 4)} (%)  "
              f"Val_AUROC: {round(run_manager.val_auroc, 4)} (0-1) "
              f"Val_AURPC: {round(run_manager.val_aurpc, 4)} (0-1) "
              f"Epoch_Duration: {round(run_manager.get_epoch_duration(), 4)} secs")

    run_manager.end_run()


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 6 epochs"""
    lr = args.lr * (0.33 ** (epoch // 12))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
