import torch
import torch.nn.functional as F

from Explainer.models.entropy_layer import EntropyLinear


def entropy_loss(model: torch.nn.Module):
    """
    Entropy loss function to get simple logic explanations.

    :param model: pytorch model.
    :return: entropy loss.
    """
    loss = 0
    for module in model.model.children():
        if isinstance(module, EntropyLinear):
            loss -= torch.sum(module.alpha * torch.log(module.alpha))
            break
    return loss


def loss_fn_kd(explainer_logits, bb_logits, labels, params, weights):
    """
    reduction="none" as to compute the KD loss for each sample
    """
    alpha = params["alpha"]
    T = params["temperature"]
    distillation_loss = torch.nn.KLDivLoss(reduction="none")(
        F.log_softmax(explainer_logits / T, dim=1), F.softmax(bb_logits / T, dim=1))
    weighted_distillation_loss = weights * torch.sum(distillation_loss, dim=1)
    weighted_prediction_loss = weights * F.cross_entropy(explainer_logits, labels, reduction="none")
    mean_distillation_loss = torch.mean(weighted_distillation_loss)
    mean_prediction_loss = torch.mean(weighted_prediction_loss)
    KD_loss = (alpha * T * T) * mean_distillation_loss + \
              (1. - alpha) * mean_prediction_loss
    return KD_loss


class KD_Residual_Loss(torch.nn.Module):
    def __init__(self, iteration, CE, KLDiv, T_KD, alpha_KD):
        super(KD_Residual_Loss, self).__init__()
        self.CE = CE
        self.KLDiv = KLDiv
        self.T_KD = T_KD
        self.alpha_KD = alpha_KD
        self.iteration = iteration

    def forward(self, student_preds, teacher_preds, target, selection_weights, prev_selection_outs=None):
        if prev_selection_outs is None:
            prev_selection_outs = []

        if self.iteration == 1:
            weights = 1 - selection_weights

        else:
            pi = 1
            for prev_selection_out in prev_selection_outs:
                pi *= (1 - prev_selection_out)
            weights = pi * (1 - selection_weights)

        distillation_loss = torch.sum(
            self.KLDiv(F.log_softmax(student_preds / self.T_KD, dim=1), F.softmax(teacher_preds / self.T_KD, dim=1)),
            dim=1
        )
        distillation_risk = torch.mean(distillation_loss * weights.view(-1))
        CE_risk = torch.mean(self.CE(student_preds, target) * weights.view(-1))
        KD_risk = (self.alpha_KD * self.T_KD * self.T_KD) * distillation_risk + (1.0 - self.alpha_KD) * CE_risk

        return {
            "distillation_risk": distillation_risk,
            "CE_risk": CE_risk,
            "KD_risk": KD_risk
        }


class Distillation_Loss(torch.nn.Module):
    def __init__(self, iteration, CE, KLDiv, T_KD, alpha_KD, selection_threshold, coverage: float, lm: float = 32.0):
        """
        Based on the implementation of SelectiveNet
        Args:
            loss_func: base loss function. the shape of loss_func(x, target) shoud be (B).
                       e.g.) torch.nn.CrossEntropyLoss(reduction=none) : classification
            coverage: target coverage.
            lm: Lagrange multiplier for coverage constraint. original experiment's value is 32.
        """
        super(Distillation_Loss, self).__init__()
        assert 0.0 < coverage <= 1.0
        assert 0.0 < lm

        self.CE = CE
        self.KLDiv = KLDiv
        self.coverage = coverage
        self.lm = lm
        self.T_KD = T_KD
        self.alpha_KD = alpha_KD
        self.iteration = iteration
        self.selection_threshold = selection_threshold

    def forward(
            self, prediction_out, target, bb_out, elens_loss, lambda_lens
    ):
        # compute emp risk (=r^)
        distillation_loss = torch.sum(
            self.KLDiv(F.log_softmax(prediction_out / self.T_KD, dim=1), F.softmax(bb_out / self.T_KD, dim=1)),
            dim=1
        )
        distillation_risk = torch.mean(distillation_loss)
        CE_risk = torch.mean(self.CE(prediction_out, target))
        KD_risk = (self.alpha_KD * self.T_KD * self.T_KD) * distillation_risk + (1.0 - self.alpha_KD) * CE_risk
        entropy_risk = torch.mean(lambda_lens * elens_loss)
        emp_risk = (KD_risk + entropy_risk)

        return {
            "selective_loss": emp_risk,
            "emp_coverage": torch.tensor(1),
            "distillation_risk": distillation_risk,
            "CE_risk": CE_risk,
            "KD_risk": KD_risk,
            "entropy_risk": entropy_risk,
            "emp_risk": emp_risk,
            "cov_penalty": torch.tensor(0)
        }


class Selective_Distillation_Loss(torch.nn.Module):
    def __init__(
            self, iteration, CE, KLDiv, T_KD, alpha_KD, selection_threshold, coverage: float, dataset="cub",
            lm: float = 32.0, arch=None
    ):
        """
        Based on the implementation of SelectiveNet
        Args:
            loss_func: base loss function. the shape of loss_func(x, target) shoud be (B).
                       e.g.) torch.nn.CrossEntropyLoss(reduction=none) : classification
            coverage: target coverage.
            lm: Lagrange multiplier for coverage constraint. original experiment's value is 32.
        """
        super(Selective_Distillation_Loss, self).__init__()
        assert 0.0 < coverage <= 1.0
        assert 0.0 < lm

        self.CE = CE
        self.KLDiv = KLDiv
        self.coverage = coverage
        self.lm = lm
        self.T_KD = T_KD
        self.alpha_KD = alpha_KD
        self.iteration = iteration
        self.selection_threshold = selection_threshold
        self.dataset = dataset
        self.arch = arch

    def forward(
            self, prediction_out, selection_out, target, bb_out, elens_loss, lambda_lens, epoch, device,
            prev_selection_outs=None
    ):
        if prev_selection_outs is None:
            prev_selection_outs = []

        if self.iteration == 1:
            weights = selection_out
        else:
            pi = 1
            for prev_selection_out in prev_selection_outs:
                pi *= (1 - prev_selection_out)
            weights = pi * selection_out

        if self.dataset == "cub" or self.dataset == "CIFAR10":
            if self.iteration > 1 and epoch >= 85:
                condition = torch.full(prev_selection_outs[0].size(), True).to(device)
                for proba in prev_selection_outs:
                    condition = condition & (proba < self.selection_threshold)
                emp_coverage = torch.sum(weights) / (torch.sum(condition) + 1e-12)
            else:
                emp_coverage = torch.mean(weights)
        elif self.dataset == "mimic_cxr":
            emp_coverage = torch.mean(weights)
        elif self.dataset == "HAM10k" or self.dataset == "SIIM-ISIC":
            if self.iteration > 1:
                condition = torch.full(prev_selection_outs[0].size(), True).to(device)
                for proba in prev_selection_outs:
                    condition = condition & (proba < self.selection_threshold)
                emp_coverage = torch.sum(weights) / (torch.sum(condition) + 1e-12)
            else:
                emp_coverage = torch.mean(weights)

        # compute emp risk (=r^)
        distillation_loss = torch.sum(
            self.KLDiv(F.log_softmax(prediction_out / self.T_KD, dim=1), F.softmax(bb_out / self.T_KD, dim=1)),
            dim=1
        )

        distillation_risk = torch.mean(distillation_loss * weights.view(-1))
        CE_risk = torch.mean(self.CE(prediction_out, target) * weights.view(-1))
        KD_risk = (self.alpha_KD * self.T_KD * self.T_KD) * distillation_risk + (1.0 - self.alpha_KD) * CE_risk
        entropy_risk = torch.mean(lambda_lens * elens_loss * weights.view(-1))
        emp_risk = (KD_risk + entropy_risk) / (emp_coverage + 1e-12)

        # compute penalty (=psi)
        coverage = torch.tensor([self.coverage], dtype=torch.float32, requires_grad=True, device="cuda")
        penalty = (torch.max(
            coverage - emp_coverage,
            torch.tensor([0.0], dtype=torch.float32, requires_grad=True, device="cuda"),
        ) ** 2)
        penalty *= self.lm

        selective_loss = emp_risk + penalty
        return {
            "selective_loss": selective_loss,
            "emp_coverage": emp_coverage,
            "distillation_risk": distillation_risk,
            "CE_risk": CE_risk,
            "KD_risk": KD_risk,
            "entropy_risk": entropy_risk,
            "emp_risk": emp_risk,
            "cov_penalty": penalty
        }


class Selective_Distillation_Loss_Mimic_cxr(torch.nn.Module):
    def __init__(
            self, iteration, CE, KLDiv, T_KD, alpha_KD, selection_threshold, coverage: float, dataset="cub",
            lm: float = 32.0, cov_weight=0.2, arch=None
    ):
        super(Selective_Distillation_Loss_Mimic_cxr, self).__init__()
        assert 0.0 < coverage <= 1.0
        assert 0.0 < lm

        self.CE = CE
        self.KLDiv = KLDiv
        self.coverage = coverage
        self.lm = lm
        self.T_KD = T_KD
        self.cov_weight = cov_weight
        self.alpha_KD = alpha_KD
        self.iteration = iteration
        self.selection_threshold = selection_threshold
        self.dataset = dataset
        self.arch = arch

    def compute_emp_risk(self, prediction_out, target, bb_out, weights, lambda_lens, elens_loss, cov):
        distillation_loss = torch.sum(
            self.KLDiv(F.log_softmax(prediction_out / self.T_KD, dim=1), F.softmax(bb_out / self.T_KD, dim=1)),
            dim=1
        )

        distillation_risk = torch.mean(distillation_loss * weights.view(-1))
        CE_risk = torch.mean(self.CE(prediction_out, target) * weights.view(-1))
        # KD_risk = (self.alpha_KD * self.T_KD * self.T_KD) * distillation_risk + (1.0 - self.alpha_KD) * CE_risk
        KD_risk = CE_risk
        entropy_risk = torch.mean(lambda_lens * elens_loss * weights.view(-1))
        emp_risk = (KD_risk + entropy_risk) / (cov + 1e-12)
        return emp_risk

    def forward(
            self, prediction_out, selection_out, target, bb_out, elens_loss, lambda_lens, epoch, device,
            prev_selection_outs=None
    ):
        if prev_selection_outs is None:
            prev_selection_outs = []

        c1_cov = self.coverage * self.cov_weight
        c0_cov = self.coverage * (1 - self.cov_weight)
        idx_positive = (target == 1).nonzero(as_tuple=True)[0]
        idx_negative = (target == 0).nonzero(as_tuple=True)[0]
        selection_out_positive = selection_out[idx_positive]
        selection_out_negative = selection_out[idx_negative]
        prediction_out_positive = prediction_out[idx_positive]
        prediction_out_negative = prediction_out[idx_negative]
        target_positive = target[idx_positive]
        target_negative = target[idx_negative]
        bb_out_positive = bb_out[idx_positive]
        bb_out_negative = bb_out[idx_negative]

        if self.iteration == 1:
            weights_positive = selection_out_positive
            weights_negative = selection_out_negative
        else:
            pi_positive = 1
            pi_negative = 1
            for prev_selection_out in prev_selection_outs:
                pi_positive *= (1 - prev_selection_out[idx_positive])
                pi_negative *= (1 - prev_selection_out[idx_negative])
            weights_positive = pi_positive * selection_out_positive
            weights_negative = pi_negative * selection_out_negative

        # compute emp risk (=r^)
        if idx_positive.size(0) != 0:
            emp_coverage_positive = torch.mean(weights_positive)
            emp_risk_positve = self.compute_emp_risk(
                prediction_out_positive, target_positive, bb_out_positive, weights_positive, lambda_lens, elens_loss,
                emp_coverage_positive
            )
        else:
            emp_coverage_positive = torch.tensor([0], dtype=torch.float32, requires_grad=True, device="cuda")
            emp_risk_positve = torch.tensor([0], dtype=torch.float32, requires_grad=True, device="cuda")

        if idx_negative.size(0) != 0:
            emp_coverage_negative = torch.mean(weights_negative)
            emp_risk_negative = self.compute_emp_risk(
                prediction_out_negative, target_negative, bb_out_negative, weights_negative, lambda_lens, elens_loss,
                emp_coverage_negative
            )
        else:
            emp_coverage_negative = torch.tensor([0], dtype=torch.float32, requires_grad=True, device="cuda")
            emp_risk_negative = torch.tensor([0], dtype=torch.float32, requires_grad=True, device="cuda")

        # compute penalty (=psi)
        coverage_positive = torch.tensor([c1_cov], dtype=torch.float32, requires_grad=True, device="cuda")
        coverage_negative = torch.tensor([c0_cov], dtype=torch.float32, requires_grad=True, device="cuda")
        penalty_positive = (torch.max(
            coverage_positive - emp_coverage_positive,
            torch.tensor([0.0], dtype=torch.float32, requires_grad=True, device="cuda"),
        ) ** 2)
        penalty_negative = (torch.max(
            coverage_negative - emp_coverage_negative,
            torch.tensor([0.0], dtype=torch.float32, requires_grad=True, device="cuda"),
        ) ** 2)

        penalty = self.lm * (penalty_positive + penalty_negative)
        selective_loss = (emp_risk_positve + emp_risk_negative) + penalty

        # print()
        # print(idx_positive.size(), idx_negative.size(), emp_coverage_positive.item(), c1_cov,
        #       emp_coverage_negative.item(), c0_cov,
        #       penalty.item(), selective_loss.item())
        # print()
        return {
            "selective_loss": selective_loss,
            "emp_coverage_positive": emp_coverage_positive,
            "emp_coverage_negative": emp_coverage_negative,
            "cov_penalty_positive": penalty_positive,
            "cov_penalty_negative": penalty_negative
        }


class Selective_CE_Loss_Domain_Transfer(torch.nn.Module):
    def __init__(self, iteration, CE, selection_threshold, coverage: float, lm: float = 32.0, cov_weight=0.2):
        """
        Based on the implementation of SelectiveNet
        Args:
            loss_func: base loss function. the shape of loss_func(x, target) shoud be (B).
                       e.g.) torch.nn.CrossEntropyLoss(reduction=none) : classification
            coverage: target coverage.
            lm: Lagrange multiplier for coverage constraint. original experiment's value is 32.
        """
        super(Selective_CE_Loss_Domain_Transfer, self).__init__()
        assert 0.0 < coverage <= 1.0
        assert 0.0 < lm

        self.CE = CE
        self.coverage = coverage
        self.lm = lm
        self.cov_weight = cov_weight
        self.iteration = iteration
        self.selection_threshold = selection_threshold

    def compute_emp_risk(self, prediction_out, target, weights, lambda_lens, elens_loss, cov):
        CE_risk = torch.mean(self.CE(prediction_out, target) * weights.view(-1))
        entropy_risk = torch.mean(lambda_lens * elens_loss * weights.view(-1))
        emp_risk = (CE_risk + entropy_risk) / (cov + 1e-12)
        return emp_risk

    def forward(
            self, prediction_out, selection_out, target, elens_loss, lambda_lens, prev_selection_outs=None
    ):
        if prev_selection_outs is None:
            prev_selection_outs = []
        c1_cov = self.coverage * self.cov_weight
        c0_cov = self.coverage * (1 - self.cov_weight)
        idx_positive = (target == 1).nonzero(as_tuple=True)[0]
        idx_negative = (target == 0).nonzero(as_tuple=True)[0]
        selection_out_positive = selection_out[idx_positive]
        selection_out_negative = selection_out[idx_negative]

        prediction_out_positive = prediction_out[idx_positive]
        prediction_out_negative = prediction_out[idx_negative]
        target_positive = target[idx_positive]
        target_negative = target[idx_negative]

        if self.iteration == 1:
            weights_positive = selection_out_positive
            weights_negative = selection_out_negative
        else:
            pi_positive = 1
            pi_negative = 1
            for prev_selection_out in prev_selection_outs:
                pi_positive *= (1 - prev_selection_out[idx_positive])
                pi_negative *= (1 - prev_selection_out[idx_negative])
            weights_positive = pi_positive * selection_out_positive
            weights_negative = pi_negative * selection_out_negative

        # compute emp risk (=r^)
        if idx_positive.size(0) != 0:
            emp_coverage_positive = torch.mean(weights_positive)
            emp_risk_positve = self.compute_emp_risk(
                prediction_out_positive, target_positive, weights_positive, lambda_lens, elens_loss,
                emp_coverage_positive
            )
        else:
            emp_coverage_positive = torch.tensor([0], dtype=torch.float32, requires_grad=True, device="cuda")
            emp_risk_positve = torch.tensor([0], dtype=torch.float32, requires_grad=True, device="cuda")

        if idx_negative.size(0) != 0:
            emp_coverage_negative = torch.mean(weights_negative)
            emp_risk_negative = self.compute_emp_risk(
                prediction_out_negative, target_negative, weights_negative, lambda_lens, elens_loss,
                emp_coverage_negative
            )
        else:
            emp_coverage_negative = torch.tensor([0], dtype=torch.float32, requires_grad=True, device="cuda")
            emp_risk_negative = torch.tensor([0], dtype=torch.float32, requires_grad=True, device="cuda")

        # compute penalty (=psi)
        coverage_positive = torch.tensor([c1_cov], dtype=torch.float32, requires_grad=True, device="cuda")
        coverage_negative = torch.tensor([c0_cov], dtype=torch.float32, requires_grad=True, device="cuda")
        penalty_positive = (torch.max(
            coverage_positive - emp_coverage_positive,
            torch.tensor([0.0], dtype=torch.float32, requires_grad=True, device="cuda"),
        ) ** 2)
        penalty_negative = (torch.max(
            coverage_negative - emp_coverage_negative,
            torch.tensor([0.0], dtype=torch.float32, requires_grad=True, device="cuda"),
        ) ** 2)

        penalty = self.lm * (penalty_positive + penalty_negative)
        selective_loss = (emp_risk_positve + emp_risk_negative) + penalty

        # print()
        # print(idx_positive.size(), idx_negative.size(), emp_coverage_positive.item(), c1_cov,
        #       emp_coverage_negative.item(), c0_cov,
        #       penalty.item(), selective_loss.item())
        # print()
        return {
            "selective_loss": selective_loss,
            "emp_coverage_positive": emp_coverage_positive,
            "emp_coverage_negative": emp_coverage_negative,
            "cov_penalty_positive": penalty_positive,
            "cov_penalty_negative": penalty_negative
        }


class KD_Residual_Loss_domain_transfer(torch.nn.Module):
    def __init__(self, iteration, CE):
        super(KD_Residual_Loss_domain_transfer, self).__init__()
        self.CE = CE
        self.iteration = iteration

    def forward(self, student_preds, target, selection_weights, prev_selection_outs=None):
        if prev_selection_outs is None:
            prev_selection_outs = []

        if self.iteration == 1:
            weights = 1 - selection_weights

        else:
            pi = 1
            for prev_selection_out in prev_selection_outs:
                pi *= (1 - prev_selection_out)
            weights = pi * (1 - selection_weights)

        CE_risk = torch.mean(self.CE(student_preds, target) * weights.view(-1))

        return CE_risk
