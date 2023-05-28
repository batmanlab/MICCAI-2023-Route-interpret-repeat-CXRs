import json
import os
import pickle
import warnings

import torchvision.transforms as transforms
from tqdm import tqdm

import MIMIC_CXR.mimic_cxr_utils as mimic_cxr_utils
from BB.models.BB_DenseNet121 import DenseNet121
from Explainer.experiments_explainer_mimic_cxr import get_selected_idx_for_g, get_selected_idx_for_residual
from Explainer.loss_F import Selective_CE_Loss_Domain_Transfer, entropy_loss, KD_Residual_Loss_domain_transfer
from Explainer.models.Gated_Logic_Net import Gated_Logic_Net
from Explainer.models.residual import Residual
from Explainer.profiler import fit_explainer_domain_transfer_cxr, fit_residual_domain_transfer_cxr
from Explainer.utils_explainer import get_glts_soft_seed_domain_transfer
from Logger.logger_mimic_cxr import Logger_MIMIC_CXR
from dataset.dataset_stanford_cxr import Stanford_CXR_MoIE_train, Stanford_CXR_MoIE_val

warnings.filterwarnings("ignore")
import random
import numpy as np
import torch
import utils
from pathlib import Path


def get_previous_pi_vals(iteration, glt_list, concepts):
    pi = []
    for i in range(iteration - 1):
        _, out_select, _ = glt_list[i](concepts)
        pi.append(out_select)

    return pi


def get_moie_configs(args):
    json_file = Path(f"{args.base_path}/codebase/MIMIC_CXR/paths_mimic_cxr.json")
    with open(json_file) as _file:
        paths = json.load(_file)
    root = paths[args.disease]["MoIE_paths"][f"iter{args.iter}"]["base_path"]
    g_chk_pt = paths[args.disease]["MoIE_paths"][f"iter{args.iter}"]["checkpoint_g"]
    moie_config_path = Path(
        f"{args.output}/mimic_cxr/soft_concepts/seed_{args.seed}/explainer/{args.disease}/{root}/iter{args.iter}/g/selected/auroc/"
    )
    moie_checkpoint = Path(
        f"{args.checkpoints}/mimic_cxr/soft_concepts/seed_{args.seed}/explainer/{args.disease}/{root}/iter{args.iter}/g/selected/auroc/{g_chk_pt}"
    )

    device, moie_configs = mimic_cxr_utils.setup(moie_config_path, residual=False)
    return device, moie_configs, moie_checkpoint, root


def setup_target_paths(args, root):
    expert = "g" if args.expert_to_train == "explainer" else "residual"
    g_output_path = Path(
        f"{args.output}/{args.target_dataset}/soft_concepts/seed_{args.seed}/{args.disease}/{root}_sample_{args.tot_samples}_train_phi_{args.train_phi}/iter{args.iter}/{expert}/selected/auroc"
    ) if args.train_phi == "y" else Path(
        f"{args.output}/{args.target_dataset}/soft_concepts/seed_{args.seed}/{args.disease}/{root}_sample_{args.tot_samples}/iter{args.iter}/{expert}/selected/auroc"
    )
    g_tb_logs_path = Path(
        f"{args.logs}/{args.target_dataset}/soft_concepts/seed_{args.seed}/{args.disease}/{root}_sample_{args.tot_samples}_train_phi_{args.train_phi}/iter{args.iter}/{expert}/selected/auroc"
    ) if args.train_phi == "y" else Path(
        f"{args.logs}/{args.target_dataset}/soft_concepts/seed_{args.seed}/{args.disease}/{root}_sample_{args.tot_samples}/iter{args.iter}/{expert}/selected/auroc"
    )
    g_chk_pt_path = Path(
        f"{args.checkpoints}/{args.target_dataset}/soft_concepts/seed_{args.seed}/{args.disease}/{root}_sample_{args.tot_samples}_train_phi_{args.train_phi}/iter{args.iter}/{expert}/selected/auroc"
    ) if args.train_phi == "y" else Path(
        f"{args.checkpoints}/{args.target_dataset}/soft_concepts/seed_{args.seed}/{args.disease}/{root}_sample_{args.tot_samples}/iter{args.iter}/{expert}/selected/auroc"
    )

    print("############### Paths ###############")
    if args.cov != 0:
        g_output_path = f"{g_output_path}_cov_{args.cov}"
        g_tb_logs_path = f"{g_tb_logs_path}_cov_{args.cov}"
        g_chk_pt_path = f"{g_chk_pt_path}_cov_{args.cov}"

    if args.initialize_w_mimic == "y":
        g_output_path = f"{g_output_path}_initialize_w_mimic_{args.initialize_w_mimic}"
        g_tb_logs_path = f"{g_tb_logs_path}_initialize_w_mimic_{args.initialize_w_mimic}"
        g_chk_pt_path = f"{g_chk_pt_path}_initialize_w_mimic_{args.initialize_w_mimic}"

    print(g_output_path)
    print(g_chk_pt_path)
    print(g_tb_logs_path)

    os.makedirs(g_chk_pt_path, exist_ok=True)
    os.makedirs(g_tb_logs_path, exist_ok=True)
    os.makedirs(g_output_path, exist_ok=True)
    pickle.dump(args, open(os.path.join(g_output_path, "train_explainer_configs.pkl"), "wb"))
    print("############### Paths ###############")

    return g_output_path, g_chk_pt_path, g_tb_logs_path


def get_dataloaders(args, mode):
    batch_size = 0
    train_shuffle = None
    if mode == "train":
        batch_size = args.batch_size
        train_shuffle = True
    elif mode == "test":
        batch_size = 1
        train_shuffle = False

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
    chk_pt_path_t_mimic = Path(
        f"{args.output}/{args.source_dataset}/t/{args.source_checkpoint_t_path}/{args.arch}/{args.disease}/dataset_g"
    )
    with open(os.path.join(chk_pt_path_t_mimic, "selected_concepts_auroc.pkl"), "rb") as input_file:
        concepts = pickle.load(input_file)
    print(f"Sub selected concepts from MIMIC-CXR: {len(concepts)}")
    dataset_path = Path(
        f"{args.output}/{args.target_dataset}/t/{args.target_checkpoint_t_path}/{args.arch}/sample_size_{args.tot_samples}/{args.disease}"
    ) if args.train_phi == "n" else Path(
        f"{args.output}/{args.target_dataset}/t/{args.target_checkpoint_t_path}/{args.arch}/sample_size_{args.tot_samples}_train_phi_{args.train_phi}/{args.disease}"
    )
    args.csv_file = f"master_tot_{args.tot_samples}.csv"
    args.dataset = args.target_dataset
    print(f"=======>> Concepts loaded from: {dataset_path}")
    train_dataset = Stanford_CXR_MoIE_train(args=args, dataset_path=dataset_path, mode="train",
                                            transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=train_shuffle, num_workers=4, pin_memory=True
    )

    val_dataset = Stanford_CXR_MoIE_val(args=args, dataset_path=dataset_path, mode="test", transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, concepts


def get_models(args, moie_configs, moie_checkpoint, device):
    glt_list = []
    if args.iter > 1:
        args.soft = "y"
        args.metric = "auroc"
        args.input_size_pi = moie_configs.input_size_pi
        args.labels = moie_configs.labels
        args.conceptizator = moie_configs.conceptizator
        args.temperature_lens = moie_configs.temperature_lens
        glt_list = get_glts_soft_seed_domain_transfer(args.iter, args, args.seed, device, dataset=args.dataset)

    moie_source = Gated_Logic_Net(
        moie_configs.input_size_pi, args.concept_names, moie_configs.labels, moie_configs.hidden_nodes,
        moie_configs.conceptizator, moie_configs.temperature_lens,
    ).to(device)
    print(f"=======>> G for iteration {args.iter} is loaded from {moie_checkpoint}")
    moie_source.load_state_dict(torch.load(moie_checkpoint)["state_dict"])
    moie_source.eval()

    moie_target = Gated_Logic_Net(
        moie_configs.input_size_pi, args.concept_names, moie_configs.labels, moie_configs.hidden_nodes,
        moie_configs.conceptizator, moie_configs.temperature_lens,
    ).to(device)
    if args.initialize_w_mimic == "y":
        moie_target.load_state_dict(torch.load(moie_checkpoint)["state_dict"])

    return moie_source, moie_target, glt_list


def setup(args, mode):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device, moie_configs, moie_checkpoint, root = get_moie_configs(args)
    g_output_path, g_chk_pt_path, g_tb_logs_path = setup_target_paths(args, root)
    train_loader, val_loader, concepts = get_dataloaders(args, mode)
    print(f"****************** Training the explainer for iteration: {args.iter}, seed: {args.seed} ******************")
    args.concept_names = concepts
    moie_source, moie_target, glt_list = get_models(args, moie_configs, moie_checkpoint, device)
    return (
        moie_configs, moie_source, moie_target, device, g_output_path, g_chk_pt_path, g_tb_logs_path,
        train_loader, val_loader, root, glt_list
    )


def test_glt(args):
    (
        moie_configs, moie_source, moie_target, device, g_output_path, g_chk_pt_path, g_tb_logs_path,
        train_loader, val_loader, root, glt_list
    ) = setup(args, mode="test")

    pickle.dump(moie_configs, open(os.path.join(g_output_path, "test_explainer_configs.pkl"), "wb"))
    moie_checkpoint = os.path.join(g_chk_pt_path, args.target_checkpoint)
    print(f"=======>> Pi for iteration {args.iter} is loaded from {moie_checkpoint}")
    moie_target.load_state_dict(torch.load(moie_checkpoint)["state_dict"])
    moie_target.eval()
    if args.cov == 0:
        coverage = moie_configs.cov
    else:
        coverage = args.cov

    print("******************* Predicting and saving the outputs *******************")
    predict(args, moie_source, moie_target, val_loader, g_output_path, mode="test")
    save_results_selected_by_pi(
        args,
        args.iter,
        val_loader,
        moie_source,
        moie_target,
        args.selection_threshold,
        g_output_path,
        device,
        mode="test",
        higher_iter_params={
            "glt_list": glt_list,
        },
    )


def predict(args, moie_source, moie_target, loader, output_path, mode):
    out_put_sel_proba = torch.FloatTensor().cuda()
    out_put_class = torch.FloatTensor().cuda()
    out_put_target = torch.FloatTensor().cuda()
    proba_concept = torch.FloatTensor().cuda()

    with torch.no_grad():
        with tqdm(total=len(loader)) as t:
            for batch_id, data in enumerate(loader):
                image, disease_label, proba_concept_x, attributes_gt, image_names = data
                if torch.cuda.is_available():
                    proba_concept_x = proba_concept_x.cuda(args.gpu, non_blocking=True)
                    disease_label = disease_label.cuda(args.gpu, non_blocking=True).to(torch.long)
                    disease_idx = list(map(str.lower, args.chexpert_names)).index(args.disease.lower())
                    disease_label = disease_label.view(1, -1)[:, disease_idx]

                if args.initialize_w_mimic == "n":
                    out_class, _, out_aux = moie_source(proba_concept_x)
                    _, out_select, _ = moie_target(proba_concept_x)

                else:
                    out_class, out_select, out_aux = moie_target(proba_concept_x)

                out_put_sel_proba = torch.cat((out_put_sel_proba, out_select), dim=0)
                out_put_class = torch.cat((out_put_class, out_class), dim=0)
                out_put_target = torch.cat((out_put_target, disease_label), dim=0)
                proba_concept = torch.cat((proba_concept, proba_concept_x), dim=0)

                t.set_postfix(batch_id="{0}".format(batch_id))
                t.update()

    out_put_sel_proba = out_put_sel_proba.cpu()
    out_put_class_pred = out_put_class.cpu()
    out_put_target = out_put_target.cpu()
    proba_concept = proba_concept.cpu()

    print(f"out_put_sel_proba size: {out_put_sel_proba.size()}")
    print(f"out_put_class_pred size: {out_put_class_pred.size()}")
    print(f"out_put_target size: {out_put_target.size()}")
    print(f"proba_concept size: {proba_concept.size()}")

    os.makedirs(os.path.join(output_path, "model_outputs"), exist_ok=True)
    utils.save_tensor(
        path=os.path.join(output_path, "model_outputs", f"{mode}_out_put_sel_proba.pt"),
        tensor_to_save=out_put_sel_proba
    )
    utils.save_tensor(
        path=os.path.join(output_path, "model_outputs", f"{mode}_out_put_class_pred.pt"),
        tensor_to_save=out_put_class_pred
    )
    utils.save_tensor(
        path=os.path.join(output_path, "model_outputs", f"{mode}_out_put_target.pt"), tensor_to_save=out_put_target
    )
    utils.save_tensor(
        path=os.path.join(output_path, "model_outputs", f"{mode}_proba_concept.pt"), tensor_to_save=proba_concept
    )


def save_results_selected_by_pi(
        args,
        iteration,
        data_loader,
        moie_source,
        moie_target,
        selection_threshold,
        output_path,
        device,
        mode,
        higher_iter_params
):
    glt_list = None
    residual = None
    if iteration > 1:
        glt_list = higher_iter_params["glt_list"]

    mask_by_pi = torch.FloatTensor().cuda()
    gt_tensor = torch.FloatTensor().cuda()
    pred_tensor = torch.FloatTensor().cuda()
    # tensor_conceptizator_threshold = torch.FloatTensor().cuda()
    tensor_concept_mask = torch.FloatTensor().cuda()
    tensor_alpha = torch.FloatTensor().cuda()
    tensor_alpha_norm = torch.FloatTensor().cuda()
    with torch.no_grad():
        with tqdm(total=len(data_loader)) as t:
            for batch_id, data in enumerate(data_loader):
                image, disease_label, proba_concept_x, attributes_gt, image_names = data
                if torch.cuda.is_available():
                    proba_concept_x = proba_concept_x.cuda(args.gpu, non_blocking=True)
                    disease_label = disease_label.cuda(args.gpu, non_blocking=True).view(-1).to(torch.long)
                    disease_idx = list(map(str.lower, args.chexpert_names)).index(args.disease.lower())
                    disease_label = disease_label.view(1, -1)[:, disease_idx]

                pi_list = None
                if iteration > 1:
                    pi_list = get_previous_pi_vals(iteration, glt_list, proba_concept_x)

                if args.initialize_w_mimic == "n":
                    prediction_out, _, auxiliary_out, concept_mask, alpha, alpha_norm, conceptizator = moie_source(
                        proba_concept_x, test=True)
                    _, selection_out, _ = moie_target(proba_concept_x)
                else:
                    prediction_out, selection_out, auxiliary_out, concept_mask, alpha, alpha_norm, conceptizator = moie_target(
                        proba_concept_x, test=True)

                condition = get_selected_idx_for_g(iteration, selection_out, selection_threshold, device, pi_list)
                mask = torch.where(
                    condition,
                    torch.ones_like(selection_out),
                    torch.zeros_like(selection_out),
                )
                mask_by_pi = torch.cat((mask_by_pi, mask), dim=0)
                gt_tensor = torch.cat((gt_tensor, disease_label), dim=0)
                pred_tensor = torch.cat((pred_tensor, prediction_out), dim=0)
                tensor_concept_mask = concept_mask
                tensor_alpha = alpha
                tensor_alpha_norm = alpha_norm
                t.set_postfix(batch_id="{0}".format(batch_id))
                t.update()

    gt_tensor = gt_tensor
    pred_tensor_class = pred_tensor.argmax(dim=1)
    pred_tensor_proba = torch.nn.Softmax()(pred_tensor)[:, 1]
    print(gt_tensor.size(), pred_tensor_class.size(), pred_tensor_proba.size())
    print(f"All acc: {(gt_tensor == pred_tensor_class).sum(dim=0) / gt_tensor.size(0)}")
    print(f"All auroc: {utils.compute_AUC(gt_tensor, pred_tensor_proba)}")

    sel = mask_by_pi
    print(torch.sum(mask_by_pi))
    print(mask_by_pi.size())
    g_gt = torch.masked_select(gt_tensor.view(-1, 1), sel.bool())
    g_pred_tensor_class = torch.masked_select(pred_tensor_class.view(-1, 1), sel.bool())
    g_pred_tensor_proba = torch.masked_select(pred_tensor_proba.view(-1, 1), sel.bool())

    print(g_gt.size(), g_pred_tensor_class.size(), g_pred_tensor_proba.size())
    print(f"g acc: {(g_gt == g_pred_tensor_class).sum(dim=0) / g_gt.size(0)}")
    print(f"g auroc: {utils.compute_AUC(g_gt, g_pred_tensor_proba)}")

    mask_by_pi = mask_by_pi.cpu()
    tensor_concept_mask = tensor_concept_mask.cpu()
    tensor_alpha = tensor_alpha.cpu()
    tensor_alpha_norm = tensor_alpha_norm.cpu()

    print("Output sizes: ")
    print(f"mask_by_pi size: {mask_by_pi.size()}")
    print("Model-specific sizes: ")
    # print(f"tensor_conceptizator_threshold: {tensor_conceptizator_threshold}")
    print(f"tensor_concept_mask size: {tensor_concept_mask.size()}")
    print(f"tensor_alpha size: {tensor_alpha.size()}")
    print(f"tensor_alpha_norm size: {tensor_alpha_norm.size()}")

    os.makedirs(os.path.join(output_path, "g_outputs"), exist_ok=True)
    utils.save_tensor(
        path=os.path.join(output_path, "g_outputs", f"{mode}_mask_by_pi.pt"), tensor_to_save=mask_by_pi
    )

    utils.save_tensor(
        path=os.path.join(output_path, "g_outputs", f"{mode}_tensor_concept_mask.pt"),
        tensor_to_save=tensor_concept_mask
    )

    utils.save_tensor(
        path=os.path.join(output_path, "g_outputs", f"{mode}_tensor_alpha.pt"),
        tensor_to_save=tensor_alpha
    )
    utils.save_tensor(
        path=os.path.join(output_path, "g_outputs", f"{mode}_tensor_alpha_norm.pt"), tensor_to_save=tensor_alpha_norm
    )


def train_glt(args):
    if args.train_phi == "n":
        print("************************** Training MoIE with concepts by learning T only  **************************")
    elif args.train_phi == "y":
        print("********************** Training MoIE with concepts by learning Phi and T both  **********************")

    (
        moie_configs, moie_source, moie_target, device, g_output_path, g_chk_pt_path, g_tb_logs_path,
        train_loader, val_loader, root, glt_list
    ) = setup(args, mode="train")
    if args.expert_to_train == "explainer":
        return train_explainer(
            args, moie_configs, moie_target, moie_source, g_chk_pt_path, g_tb_logs_path, g_output_path, train_loader,
            val_loader, glt_list, device
        )
    elif args.expert_to_train == "residual":
        return train_residual(
            args, root, moie_target, moie_configs, g_chk_pt_path, g_tb_logs_path, g_output_path, train_loader,
            val_loader, glt_list, device
        )


def train_residual(
        args, root, moie_target, moie_configs, g_chk_pt_path, g_tb_logs_path, g_output_path, train_loader,
        val_loader, glt_list, device
):
    n_class = 2
    moie_checkpoint = Path(
        f"{args.checkpoints}/mimic_cxr/soft_concepts/seed_{args.seed}/explainer/{args.disease}/{root}/iter{args.iter}/residual/selected/auroc/best_model.pth.tar"
    )
    residual = Residual(args.dataset, True, len(args.labels), args.arch).to(device)
    print(moie_checkpoint)

    residual.load_state_dict(torch.load(moie_checkpoint)["state_dict"])
    chk_pt_path_bb_mimic = Path(
        f"{args.checkpoints}/{args.source_dataset}/BB/{args.source_checkpoint_bb_path}/{args.arch}/{args.disease}/{args.source_checkpoint_bb}"
    )
    moie_configs.epochs_residual = 2
    moie_configs.N_labels = n_class
    moie_configs.disease = args.disease
    bb_mimic = DenseNet121(moie_configs, layer=moie_configs.layer).cuda()
    bb_mimic.load_state_dict(torch.load(chk_pt_path_bb_mimic)['state_dict'])
    bb_mimic.eval()

    CE = torch.nn.CrossEntropyLoss(reduction="none")
    kd_Loss = KD_Residual_Loss_domain_transfer(args.iter, CE)
    optimizer = torch.optim.SGD(
        residual.parameters(), lr=moie_configs.lr_residual, momentum=moie_configs.momentum_residual,
        weight_decay=moie_configs.weight_decay_residual
    )
    moie_target_checkpoint = Path(
        f"{args.checkpoints}/mimic_cxr/soft_concepts/seed_{args.seed}/explainer/{args.disease}/{root}/iter{args.iter}/g/selected/auroc/best_model.pth.tar"
    )
    moie_target.load_state_dict(torch.load(moie_target_checkpoint)["state_dict"])
    moie_target.eval()
    logger = Logger_MIMIC_CXR(
        args.iter, 0, 0, g_chk_pt_path, g_tb_logs_path, g_output_path,
        train_loader, val_loader, n_class, model_type="residual", device=device
    )

    moie_configs.chexpert_names = args.chexpert_names
    if args.profile == "y":
        return fit_residual_domain_transfer_cxr(
            moie_configs,
            bb_mimic,
            args.iter,
            moie_target,
            glt_list,
            residual,
            optimizer,
            train_loader,
            val_loader,
            kd_Loss,
            os.path.join("residual"),
            moie_configs.selection_threshold,
            device
        )

    else:
        fit_residual(
            moie_configs,
            bb_mimic,
            args.iter,
            moie_target,
            glt_list,
            residual,
            optimizer,
            train_loader,
            val_loader,
            kd_Loss,
            logger,
            os.path.join("residual"),
            moie_configs.selection_threshold,
            device
        )

        train_loader, val_loader, concepts = get_dataloaders(args, mode="test")
        output_path_model_outputs = os.path.join(g_output_path, "model_outputs")
        output_path_residual_outputs = os.path.join(g_output_path, "residual_outputs")
        os.makedirs(output_path_model_outputs, exist_ok=True)
        os.makedirs(output_path_residual_outputs, exist_ok=True)
        predict_residual(args, bb_mimic, residual, val_loader, output_path_model_outputs, mode="test")
        print("!! Saving test loader only selected by g!!")
        pickle.dump(moie_configs, open(os.path.join(g_output_path, "test_configs.pkl"), "wb"))
        save_results_selected_residual_by_pi(
            args,
            bb_mimic,
            args.iter,
            residual,
            moie_target,
            val_loader,
            args.selection_threshold,
            output_path_residual_outputs,
            device,
            mode="test",
            higher_iter_params={
                "glt_list": glt_list
            }
        )

        return 0


def predict_residual(args, bb_mimic, residual, loader, output_path_model_outputs, mode="test"):
    print(f"Mode: {mode}")
    out_put_preds_residual = torch.FloatTensor().cuda()
    out_put_preds_bb = torch.FloatTensor().cuda()
    out_put_target = torch.FloatTensor().cuda()

    with torch.no_grad():
        with tqdm(total=len(loader)) as t:
            for batch_id, data in enumerate(loader):
                image, disease_label, proba_concept_x, attributes_gt, image_names = data
                if torch.cuda.is_available():
                    image = image.cuda(args.gpu, non_blocking=True)
                    proba_concept_x = proba_concept_x.cuda(args.gpu, non_blocking=True)
                    disease_label = disease_label.cuda(args.gpu, non_blocking=True).view(-1).to(torch.long)
                disease_idx = list(map(str.lower, args.chexpert_names)).index(args.disease.lower())
                disease_label = disease_label.view(1, -1)[:, disease_idx]

                features, _, _ = bb_mimic(image)
                residual_student_logits = residual(features)

                out_put_preds_residual = torch.cat((out_put_preds_residual, residual_student_logits), dim=0)
                out_put_target = torch.cat((out_put_target, disease_label), dim=0)

                t.set_postfix(
                    batch_id='{0}'.format(batch_id + 1))
                t.update()

    out_put_preds_residual = out_put_preds_residual.cpu()
    out_put_preds_bb = out_put_preds_residual.cpu()
    out_put_target = out_put_target.cpu()

    print(f"out_put_preds_residual size: {out_put_preds_residual.size()}")
    print(f"out_put_preds_bb size: {out_put_preds_bb.size()}")
    print(f"out_put_target size: {out_put_target.size()}")

    utils.save_tensor(
        path=os.path.join(output_path_model_outputs, f"{mode}_out_put_preds_residual.pt"),
        tensor_to_save=out_put_preds_residual
    )

    utils.save_tensor(
        path=os.path.join(output_path_model_outputs, f"{mode}_out_put_preds_bb.pt"),
        tensor_to_save=out_put_preds_bb
    )

    utils.save_tensor(
        path=os.path.join(output_path_model_outputs, f"{mode}_out_put_target.pt"),
        tensor_to_save=out_put_target
    )


def save_results_selected_residual_by_pi(
        args,
        bb_mimic,
        iteration,
        residual,
        glt_model,
        data_loader,
        selection_threshold,
        output_path,
        device,
        mode,
        higher_iter_params
):
    glt_list = None
    if iteration > 1:
        glt_list = higher_iter_params["glt_list"]

    mask_by_pi = torch.FloatTensor().cuda()
    positive_selected = 0
    negative_selected = 0
    with torch.no_grad():
        with tqdm(total=len(data_loader)) as t:
            for batch_id, data in enumerate(data_loader):
                image, disease_label, proba_concept_x, attributes_gt, image_names = data
                if torch.cuda.is_available():
                    image = image.cuda(args.gpu, non_blocking=True)
                    proba_concept_x = proba_concept_x.cuda(args.gpu, non_blocking=True)
                    disease_label = disease_label.cuda(args.gpu, non_blocking=True).view(-1).to(torch.long)
                disease_idx = list(map(str.lower, args.chexpert_names)).index(args.disease.lower())
                disease_label = disease_label.view(1, -1)[:, disease_idx]

                out_class, out_select, out_aux = glt_model(proba_concept_x)
                features, _, _ = bb_mimic(image)
                residual_student_logits = residual(features)

                pi_list = None
                if iteration > 1:
                    pi_list = get_previous_pi_vals(iteration, glt_list, proba_concept_x)

                if iteration == 1:
                    idx_selected = (out_select < 0.5).nonzero(as_tuple=True)[0]
                else:
                    condition = torch.full(pi_list[0].size(), True).to(device)
                    for proba in pi_list:
                        condition = condition & (proba < selection_threshold)
                    idx_selected = (condition & (out_select < selection_threshold)).nonzero(as_tuple=True)[0]

                if idx_selected.size(0) > 0:
                    positive_selected += disease_label[idx_selected].sum(dim=0).item()
                    negative_selected += disease_label[idx_selected].size(0) - disease_label[idx_selected].sum(
                        dim=0).item()

                condition = get_selected_idx_for_residual(
                    iteration, out_select, selection_threshold, device, pi_list
                )
                mask = torch.where(
                    condition,
                    torch.ones_like(out_select),
                    torch.zeros_like(out_select),
                )
                mask_by_pi = torch.cat((mask_by_pi, mask), dim=0)
                t.set_postfix(
                    batch_id='{0}'.format(batch_id + 1))
                t.update()

    mask_by_pi = mask_by_pi.cpu()

    print("Output sizes: ")
    print(f"mask_by_pi size: {mask_by_pi.size()}")
    print(f"Selected by Residual: {mask_by_pi.sum(dim=0)}")
    print(f"n_pos_pred_residual: {positive_selected} || n_neg_pred_residual: {negative_selected} ")

    utils.save_tensor(
        path=os.path.join(output_path, f"{mode}_mask_by_pi.pt"), tensor_to_save=mask_by_pi
    )


def fit_residual(
        args,
        bb_mimic,
        iteration,
        moie_target,
        glt_list,
        residual,
        optimizer,
        train_loader,
        val_loader,
        kd_Loss,
        logger,
        run_id,
        selection_threshold,
        device
):
    logger.begin_run(run_id)
    for epoch in range(args.epochs_residual):
        logger.begin_epoch()
        residual.train()
        with tqdm(total=len(train_loader)) as t:
            for batch_id, data in enumerate(train_loader):
                image, disease_label, proba_concept_x, attributes_gt, image_names = data
                if torch.cuda.is_available():
                    image = image.cuda(args.gpu, non_blocking=True)
                    proba_concept_x = proba_concept_x.cuda(args.gpu, non_blocking=True)
                    disease_label = disease_label.cuda(args.gpu, non_blocking=True).view(-1).to(torch.long)

                out_class, out_select, out_aux = moie_target(proba_concept_x)

                pi_list = None
                if iteration > 1:
                    pi_list = get_previous_pi_vals(iteration, glt_list, proba_concept_x)

                features, _, _ = bb_mimic(image)
                residual_student_logits = residual(features)
                total_train_loss = kd_Loss(
                    student_preds=residual_student_logits,
                    target=disease_label,
                    selection_weights=out_select,
                    prev_selection_outs=pi_list
                )

                optimizer.zero_grad()
                total_train_loss.backward()
                optimizer.step()

                logger.track_train_loss(total_train_loss.item())
                logger.track_total_train_correct_per_epoch(residual_student_logits, disease_label)

                t.set_postfix(
                    epoch='{0}'.format(epoch + 1),
                    training_loss='{:05.3f}'.format(logger.epoch_train_loss))
                t.update()

        residual.eval()
        positive_selected = 0
        negative_selected = 0
        positive_out_selected = 0
        negative_out_selected = 0

        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for batch_id, data in enumerate(val_loader):
                    image, disease_label, proba_concept_x, attributes_gt, image_names = data
                    if torch.cuda.is_available():
                        image = image.cuda(args.gpu, non_blocking=True)
                        proba_concept_x = proba_concept_x.cuda(args.gpu, non_blocking=True)
                        disease_label = disease_label.cuda(args.gpu, non_blocking=True).to(torch.long)

                    disease_idx = list(map(str.lower, args.chexpert_names)).index(args.disease.lower())
                    disease_label = disease_label[:, disease_idx]
                    out_class, out_select, out_aux = moie_target(proba_concept_x)

                    pi_list = None
                    if iteration > 1:
                        pi_list = get_previous_pi_vals(iteration, glt_list, proba_concept_x)

                    features, _, _ = bb_mimic(image)
                    residual_student_logits = residual(features)

                    total_val_loss = kd_Loss(
                        student_preds=residual_student_logits,
                        target=disease_label,
                        selection_weights=out_select,
                        prev_selection_outs=pi_list
                    )

                    logger.track_val_loss(total_val_loss.item())
                    logger.track_val_outputs(out_select, residual_student_logits, disease_label,
                                             residual_student_logits)
                    logger.track_total_val_correct_per_epoch(residual_student_logits, disease_label)

                    if iteration == 1:
                        idx_selected = (out_select < 0.5).nonzero(as_tuple=True)[0]
                    else:
                        condition = torch.full(pi_list[0].size(), True).to(device)
                        for proba in pi_list:
                            condition = condition & (proba < selection_threshold)
                        idx_selected = (condition & (out_select < selection_threshold)).nonzero(as_tuple=True)[0]

                    if idx_selected.size(0) > 0:
                        positive_selected += disease_label[idx_selected].sum(dim=0).item()
                        negative_selected += disease_label[idx_selected].size(0) - disease_label[idx_selected].sum(
                            dim=0).item()
                        out_class_positive_batch = residual_student_logits[idx_selected].argmax(dim=1).sum(dim=0).item()
                        out_class_negative_batch = residual_student_logits[idx_selected].size(
                            0) - out_class_positive_batch
                        positive_out_selected += out_class_positive_batch
                        negative_out_selected += out_class_negative_batch
                    if iteration > 1:
                        logger.track_val_prev_pi(pi_list)
                    t.set_postfix(
                        epoch='{0}'.format(epoch + 1),
                        validation_loss='{:05.3f}'.format(logger.epoch_val_loss))
                    t.update()

        # evaluate residual for correctly selected samples (pi < 0.5)
        # should be higher
        logger.evaluate_g_correctly(args.selection_threshold, expert="residual")
        #
        # # evaluate residual for correctly rejected samples (pi >= 0.5)
        # # should be lower
        logger.evaluate_g_incorrectly(args.selection_threshold, expert="residual")
        logger.evaluate_coverage_stats(args.selection_threshold, expert="residual")
        logger.end_epoch(residual, optimizer, track_explainer_loss=False, save_model_wrt_g_performance=True)

        print(
            f"Epoch: [{epoch + 1}/{args.epochs_residual}] || "
            f"Train_total_loss: {round(logger.get_final_train_loss(), 4)} || "
            f"Val_total_loss: {round(logger.get_final_val_loss(), 4)} || "
            f"Train_Accuracy: {round(logger.get_final_train_accuracy(), 4)} (%) || "
            f"Val_Accuracy: {round(logger.get_final_val_accuracy(), 4)} (%) || "
            f"Val_Auroc (Entire set): {round(logger.val_auroc, 4)} || "
            f"Val_residual_Accuracy (pi < 0.5): {round(logger.get_final_G_val_accuracy(), 4)} (%) || "
            f"Val_residual_Auroc (pi < 0.5): {round(logger.get_final_G_val_auroc(), 4)} || "
            f"Val_BB_Auroc (pi < 0.5): {round(logger.val_bb_auroc, 4)} || "
            f"Val_residual_Incorrect_Accuracy (pi >= 0.5): {round(logger.get_final_G_val_incorrect_accuracy(), 4)}(%) || "
            f"Val_residual_Incorrect_Auroc (pi >= 0.5): {round(logger.get_final_G_val_incorrect_auroc(), 4)} || "
            f"Val_BB_Incorrect_Auroc (pi >= 0.5): {round(logger.val_bb_incorrect_auroc, 4)} || "
            f"Best_residual_Val_Auroc: {round(logger.get_final_best_G_val_auroc(), 4)} || "
            f"Best_Epoch: {logger.get_best_epoch_id()} || "
            f"n_selected: {logger.get_n_selected()} || "
            f"n_rejected: {logger.get_n_rejected()} || "
            f"coverage: {round(logger.get_coverage(), 4)} || "
            f"n_pos_residual: {positive_selected} || "
            f"n_neg_residual: {negative_selected} || "
            f"n_pos_pred_residual: {positive_out_selected} ||"
            f"n_neg_pred_residual: {negative_out_selected} "
        )
    logger.end_run()


def train_explainer(
        args, moie_configs, moie_target, moie_source, g_chk_pt_path, g_tb_logs_path, g_output_path, train_loader,
        val_loader, glt_list, device
):
    pos_cov_weight = 0.3
    CE = torch.nn.CrossEntropyLoss(reduction="none")
    if args.cov == 0:
        coverage = moie_configs.cov
    else:
        coverage = args.cov

    print(f"=====>>>>> Coverage: {coverage} <<<<<=====")
    selective_CE_loss = Selective_CE_Loss_Domain_Transfer(
        args.iter, CE, selection_threshold=moie_configs.selection_threshold, coverage=coverage,
        lm=moie_configs.lm, cov_weight=pos_cov_weight
    )

    optimizer = None
    if args.optim == "SGD":
        optimizer = torch.optim.SGD(moie_target.parameters(), lr=moie_configs.lr, momentum=0.9, weight_decay=5e-4)
    elif args.optim == "ADAM":
        optimizer = torch.optim.Adam(moie_target.parameters(), lr=moie_configs.lr)

    best_auroc = 0
    n_class = 2
    logger = Logger_MIMIC_CXR(
        args.iter, best_auroc, moie_configs.start_epoch, g_chk_pt_path, g_tb_logs_path, g_output_path, train_loader,
        val_loader,
        n_class, model_type="g", device=device
    )
    if args.profile == "n":
        moie_target = fit_explainer(
            args, logger, train_loader, val_loader, moie_source, moie_target, glt_list, selective_CE_loss,
            optimizer, moie_configs, coverage, pos_cov_weight, device
        )
        pickle.dump(moie_configs, open(os.path.join(g_output_path, "test_explainer_configs.pkl"), "wb"))
        train_loader, val_loader, concepts = get_dataloaders(args, mode="test")
        predict(args, moie_source, moie_target, val_loader, g_output_path, mode="test")
        save_results_selected_by_pi(
            args,
            args.iter,
            val_loader,
            moie_source,
            moie_target,
            args.selection_threshold,
            g_output_path,
            device,
            mode="test",
            higher_iter_params={
                "glt_list": glt_list,
            },
        )
        return 0
    else:
        total_flops = fit_explainer_domain_transfer_cxr(
            args, train_loader, val_loader, moie_source, moie_target, glt_list, selective_CE_loss,
            optimizer, moie_configs, coverage, pos_cov_weight, device
        )
        return total_flops


def fit_explainer(
        args, logger, train_loader, val_loader, moie_source, moie_target, glt_list, selective_CE_loss,
        optimizer, moie_configs, coverage, pos_cov_weight, device
):
    run_id = "MoIE_iters"
    logger.begin_run(run_id)
    for epoch in range(args.epochs):
        logger.begin_epoch()
        moie_target.train()
        epoch_train_emp_coverage_positive = []
        epoch_train_emp_coverage_negative = []
        epoch_train_cov_penalty_positive = 0
        epoch_train_cov_penalty_negative = 0
        with tqdm(total=len(train_loader)) as t:
            for batch_id, data in enumerate(train_loader):
                image, disease_label, proba_concept_x, attributes_gt, image_names = data
                if torch.cuda.is_available():
                    proba_concept_x = proba_concept_x.cuda(args.gpu, non_blocking=True)
                    disease_label = disease_label.cuda(args.gpu, non_blocking=True).view(-1).to(torch.long)

                if args.initialize_w_mimic == "n":
                    out_class, _, out_aux = moie_source(proba_concept_x)
                    _, out_select, _ = moie_target(proba_concept_x)
                    aux_entropy_loss_elens = entropy_loss(moie_source.aux_explainer)
                    entropy_loss_elens = entropy_loss(moie_source.explainer)

                else:
                    out_class, out_select, out_aux = moie_target(proba_concept_x)
                    aux_entropy_loss_elens = entropy_loss(moie_target.aux_explainer)
                    entropy_loss_elens = entropy_loss(moie_target.explainer)

                pi_list = None
                if args.iter > 1:
                    pi_list = get_previous_pi_vals(args.iter, glt_list, proba_concept_x)

                loss_dict = selective_CE_loss(
                    out_class, out_select, disease_label, entropy_loss_elens, moie_configs.lambda_lens,
                    prev_selection_outs=pi_list
                )
                train_selective_loss = loss_dict["selective_loss"]
                emp_coverage_positive = loss_dict["emp_coverage_positive"].item()
                emp_coverage_negative = loss_dict["emp_coverage_negative"].item()
                cov_penalty_positive = loss_dict["cov_penalty_positive"].item()
                cov_penalty_negative = loss_dict["cov_penalty_negative"].item()
                train_selective_loss *= moie_configs.alpha
                aux_KD_loss = torch.nn.CrossEntropyLoss()(out_aux, disease_label)

                train_aux_loss = aux_KD_loss + moie_configs.lambda_lens * aux_entropy_loss_elens
                train_aux_loss *= (1.0 - moie_configs.alpha)

                total_train_loss = train_selective_loss + train_aux_loss
                optimizer.zero_grad()
                total_train_loss.backward()
                optimizer.step()

                epoch_train_emp_coverage_positive.append(emp_coverage_positive)
                epoch_train_emp_coverage_negative.append(emp_coverage_negative)
                epoch_train_cov_penalty_positive += cov_penalty_positive * train_loader.batch_size
                epoch_train_cov_penalty_negative += cov_penalty_negative * train_loader.batch_size

                logger.track_train_loss(total_train_loss.item())
                logger.track_total_train_correct_per_epoch(out_class, disease_label)
                t.set_postfix(epoch='{0}'.format(epoch + 1), training_loss='{:05.3f}'.format(logger.epoch_train_loss))
                t.update()

        moie_target.eval()
        epoch_val_emp_coverage_positive = []
        epoch_val_emp_coverage_negative = []
        epoch_val_cov_penalty_positive = 0
        epoch_val_cov_penalty_negative = 0
        positive_selected = 0
        negative_selected = 0
        positive_out_selected = 0
        negative_out_selected = 0
        tot_pos = 0
        total_item = 0
        with torch.no_grad():
            with tqdm(total=len(val_loader)) as t:
                for batch_id, data in enumerate(val_loader):
                    image, disease_label, proba_concept_x, attributes_gt, image_names = data
                    if torch.cuda.is_available():
                        proba_concept_x = proba_concept_x.cuda(args.gpu, non_blocking=True)
                        disease_label = disease_label.cuda(args.gpu, non_blocking=True).to(torch.long)

                    disease_idx = list(map(str.lower, args.chexpert_names)).index(args.disease.lower())
                    disease_label = disease_label[:, disease_idx]

                    if args.initialize_w_mimic == "n":
                        out_class, _, out_aux = moie_source(proba_concept_x)
                        _, out_select, _ = moie_target(proba_concept_x)
                        aux_entropy_loss_elens = entropy_loss(moie_source.aux_explainer)
                        entropy_loss_elens = entropy_loss(moie_source.explainer)

                    else:
                        out_class, out_select, out_aux = moie_target(proba_concept_x)
                        aux_entropy_loss_elens = entropy_loss(moie_target.aux_explainer)
                        entropy_loss_elens = entropy_loss(moie_target.explainer)

                    pi_list = None
                    if args.iter > 1:
                        pi_list = get_previous_pi_vals(args.iter, glt_list, proba_concept_x)

                    loss_dict = selective_CE_loss(
                        out_class, out_select, disease_label, entropy_loss_elens, moie_configs.lambda_lens,
                        prev_selection_outs=pi_list
                    )

                    val_selective_loss = loss_dict["selective_loss"]
                    emp_coverage_positive = loss_dict["emp_coverage_positive"].item()
                    emp_coverage_negative = loss_dict["emp_coverage_negative"].item()
                    cov_penalty_positive = loss_dict["cov_penalty_positive"].item()
                    cov_penalty_negative = loss_dict["cov_penalty_negative"].item()

                    val_selective_loss *= moie_configs.alpha
                    aux_KD_loss = torch.nn.CrossEntropyLoss()(out_aux, disease_label)

                    val_aux_loss = aux_KD_loss + moie_configs.lambda_lens * aux_entropy_loss_elens
                    val_aux_loss *= (1.0 - moie_configs.alpha)

                    if args.iter == 1:
                        idx_selected = (out_select >= 0.5).nonzero(as_tuple=True)[0]
                    else:
                        condition = torch.full(pi_list[0].size(), True).to(device)
                        for proba in pi_list:
                            condition = condition & (proba < moie_configs.selection_threshold)
                        idx_selected = (
                                condition & (out_select >= moie_configs.selection_threshold)
                        ).nonzero(as_tuple=True)[0]

                    if idx_selected.size(0) > 0:
                        positive_selected += disease_label[idx_selected].sum(dim=0).item()
                        negative_selected += (
                                disease_label[idx_selected].size(0) - disease_label[idx_selected].sum(dim=0).item()
                        )
                        out_class_positive_batch = out_class[idx_selected].argmax(dim=1).sum(dim=0).item()
                        out_class_negative_batch = out_class[idx_selected].size(0) - out_class_positive_batch
                        positive_out_selected += out_class_positive_batch
                        negative_out_selected += out_class_negative_batch

                    total_item += disease_label.size(0)
                    total_val_loss = val_selective_loss + val_aux_loss

                    logger.track_val_loss(total_val_loss.item())
                    logger.track_val_outputs(out_select, out_class, disease_label, out_class_bb=out_select)
                    logger.track_total_val_correct_per_epoch(out_class, disease_label)

                    epoch_val_emp_coverage_positive.append(emp_coverage_positive)
                    epoch_val_emp_coverage_negative.append(emp_coverage_negative)
                    epoch_val_cov_penalty_positive += cov_penalty_positive * val_loader.batch_size
                    epoch_val_cov_penalty_negative += cov_penalty_negative * val_loader.batch_size
                    if args.iter > 1:
                        logger.track_val_prev_pi(pi_list)

                    # ii = (val_y == 1).nonzero(as_tuple=True)[0]
                    # print(out_class[ii, :].argmax(dim=1).sum(dim=0))
                    # print(val_bb_logits[ii, :].argmax(dim=1).sum(dim=0))
                    # print(val_y[ii].sum(dim=0))
                    t.set_postfix(epoch='{0}'.format(epoch + 1), val_loss='{:05.3f}'.format(logger.epoch_val_loss))
                    t.update()

        # evaluate g for correctly selected samples (pi >= 0.5)
        # should be higher
        logger.evaluate_g_correctly(moie_configs.selection_threshold, expert="explainer")

        # evaluate g for correctly rejected samples (pi < 0.5)
        # should be lower
        logger.evaluate_g_incorrectly(moie_configs.selection_threshold, expert="explainer")
        logger.evaluate_coverage_stats(moie_configs.selection_threshold)
        logger.end_epoch(moie_target, optimizer, track_explainer_loss=True, save_model_wrt_g_performance=True)

        print(
            f"Epoch: [{epoch + 1}/{args.epochs}] || "
            f"Train_total_loss: {round(logger.get_final_train_loss(), 4)} || "
            f"epoch_train_emp_coverage_positive: {np.mean(np.array(epoch_train_emp_coverage_positive))} || "
            f"coverage_positive: {coverage * pos_cov_weight} || "
            f"epoch_train_emp_coverage_negative: {np.mean(np.array(epoch_train_emp_coverage_negative))} || "
            f"coverage_negative: {coverage * (1 - pos_cov_weight)} || "
            f"epoch_train_cov_penalty_positive: {round(epoch_train_cov_penalty_positive / len(train_loader), 4)} || "
            f"epoch_train_cov_penalty_negative: {round(epoch_train_cov_penalty_negative / len(train_loader), 4)} || "
            f"Val_total_loss: {round(logger.get_final_val_loss(), 4)} || "
            f"epoch_val_emp_coverage_positive: {np.mean(np.array(epoch_val_emp_coverage_negative))} || "
            f"coverage_positive: {coverage * pos_cov_weight} || "
            f"epoch_val_emp_coverage_negative: {np.mean(np.array(epoch_val_emp_coverage_negative))} || "
            f"coverage_negative: {coverage * (1 - pos_cov_weight)} || "
            f"epoch_val_cov_penalty_positive: {round(epoch_val_cov_penalty_positive / len(val_loader), 4)} || "
            f"epoch_val_cov_penalty_negative: {round(epoch_val_cov_penalty_negative / len(val_loader), 4)} || "
            f"Train_Accuracy: {round(logger.get_final_train_accuracy(), 4)} (%) || "
            f"Val_Accuracy: {round(logger.get_final_val_accuracy(), 4)} (%) || "
            f"Val_Auroc (Entire set by G): {round(logger.val_auroc, 4)} || "
            f"Val_G_Accuracy (pi >= 0.5): {round(logger.get_final_G_val_accuracy(), 4)} (%) || "
            f"Val_G_Auroc (pi >= 0.5): {round(logger.get_final_G_val_auroc(), 4)} || "
            f"Val_BB_Auroc (pi >= 0.5): {round(logger.val_bb_auroc, 4)} || "
            f"Val_G_Incorrect_Accuracy (pi < 0.5): {round(logger.get_final_G_val_incorrect_accuracy(), 4)} (%) || "
            f"Val_G_Incorrect_Auroc (pi < 0.5): {round(logger.get_final_G_val_incorrect_auroc(), 4)} || "
            f"Val_BB_Incorrect_Auroc (pi < 0.5): {round(logger.val_bb_incorrect_auroc, 4)} || "
            f"Best_G_Val_Auroc: {round(logger.get_final_best_G_val_auroc(), 4)} || "
            f"Best_Epoch: {logger.get_best_epoch_id()} || "
            f"n_selected: {logger.get_n_selected()} || "
            f"n_rejected: {logger.get_n_rejected()} || "
            f"coverage: {round(logger.get_coverage(), 4)} || "
            # f"coverage_1: {(positive_selected + negative_selected) / total_item} || "
            f"n_pos_ground_truth_g: {positive_selected} || "
            f"n_neg_ground_truth_g: {negative_selected} ||"
            f"n_pos_pred_g: {positive_out_selected} ||"
            f"n_neg_pred_g: {negative_out_selected} "
        )
        print()
    logger.end_run()
    return moie_target
