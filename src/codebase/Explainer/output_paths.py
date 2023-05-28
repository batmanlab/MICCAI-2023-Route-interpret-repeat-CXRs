paths = {
    "cub_ResNet101": {
        "iter1": {
            "base_path": "ResNet101/lr_0.01_epochs_65_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none",
            "prev_path": "",
            "pred_file": "test_tensor_preds.pt",
            "gt_file": "test_tensor_y.pt",
            "output": "g_outputs"
        },
        "iter2": {
            "base_path": "ResNet101/lr_0.01_epochs_200_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none",
            "prev_path": "cov_0.2_lr_0.01",
            "pred_file": "test_tensor_preds.pt",
            "gt_file": "test_tensor_y.pt",
            "output": "g_outputs"
        },
        "iter3": {
            "base_path": "ResNet101/lr_0.01_epochs_300_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none",
            "prev_path": "cov_0.2_lr_0.01",
            "pred_file": "test_tensor_preds.pt",
            "gt_file": "test_tensor_y.pt",
            "output": "g_outputs"
        }
    },
    "cub_ViT-B_16": {
        "iter1": {
            "base_path": "ViT-B_16/lr_0.01_epochs_65_temperature-lens_6.0_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_10.0_hidden-layers_1_layer_VIT_explainer_init_none",
            "prev_path": "",
            "pred_file": "test_tensor_preds.pt",
            "gt_file": "test_tensor_y.pt",
            "output": "g_outputs"
        },
        "iter2": {
            "base_path": "ViT-B_16/lr_0.01_epochs_250_temperature-lens_6.0_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_10.0_hidden-layers_1_layer_VIT_explainer_init_none",
            "prev_path": "cov_0.2_lr_0.01",
            "pred_file": "test_tensor_preds.pt",
            "gt_file": "test_tensor_y.pt",
            "output": "g_outputs"
        },
        "iter3": {
            "base_path": "ViT-B_16/lr_0.01_epochs_300_temperature-lens_6.0_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_10.0_hidden-layers_1_layer_VIT_explainer_init_none",
            "prev_path": "cov_0.2_lr_0.01",
            "pred_file": "test_tensor_preds.pt",
            "gt_file": "test_tensor_y.pt",
            "output": "g_outputs"
        }
    },
    "awa2_ResNet50": {
        "iter1": {
            "base_path": "ResNet50/lr_0.0001_epochs_150_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.4_alpha_0.5_selection-threshold_0.5_lambda-lens_0.001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none",
            "prev_path": "",
            "pred_file": "test_tensor_preds.pt",
            "gt_file": "test_tensor_y.pt",
            "output": "g_outputs"
        },
        "iter2": {
            "base_path": "ResNet50/lr_0.0001_epochs_50_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.4_alpha_0.5_selection-threshold_0.5_lambda-lens_0.001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none",
            "prev_path": "cov_0.4_lr_0.0001",
            "pred_file": "test_tensor_preds.pt",
            "gt_file": "test_tensor_y.pt",
            "output": "g_outputs"
        },
        "iter3": {
            "base_path": "ResNet50/lr_0.0001_epochs_50_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.4_alpha_0.5_selection-threshold_0.5_lambda-lens_0.001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none",
            "prev_path": "cov_0.4_lr_0.0001",
            "pred_file": "test_tensor_preds.pt",
            "gt_file": "test_tensor_y.pt",
            "output": "g_outputs"
        }
    },
    "awa2_ViT-B_16": {
        "iter1": {
            "base_path": "ViT-B_16/lr_0.01_epochs_150_temperature-lens_6.0_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_10.0_hidden-layers_1_layer_VIT_explainer_init_none",
            "prev_path": "",
            "pred_file": "test_tensor_preds.pt",
            "gt_file": "test_tensor_y.pt",
            "output": "g_outputs"
        },
        "iter2": {
            "base_path": "ViT-B_16/lr_0.01_epochs_50_temperature-lens_6.0_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_10.0_hidden-layers_1_layer_VIT_explainer_init_none",
            "prev_path": "cov_0.2_lr_0.01",
            "pred_file": "test_tensor_preds.pt",
            "gt_file": "test_tensor_y.pt",
            "output": "g_outputs"
        },
        "iter3": {
            "base_path": "ViT-B_16/lr_0.01_epochs_50_temperature-lens_6.0_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_10.0_hidden-layers_1_layer_VIT_explainer_init_none",
            "prev_path": "cov_0.2_lr_0.01",
            "pred_file": "test_tensor_preds.pt",
            "gt_file": "test_tensor_y.pt",
            "output": "g_outputs"
        }
    },
    "HAM10k_Inception_V3": {
        "iter1": {
            "base_path": "lr_0.01_epochs_80_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1",
            "prev_path": "",
            "pred_file": "val_tensor_preds.pt",
            "gt_file": "val_tensor_y.pt",
            "output": "accuracy/g_outputs"
        },
        "iter2": {
            "base_path": "lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1",
            "prev_path": "cov_0.2",
            "pred_file": "val_tensor_preds.pt",
            "gt_file": "val_tensor_y.pt",
            "output": "accuracy/g_outputs"
        },
        "iter3": {
            "base_path": "lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1",
            "prev_path": "cov_0.2",
            "pred_file": "val_tensor_preds.pt",
            "gt_file": "val_tensor_y.pt",
            "output": "accuracy/g_outputs"
        }
    },
    "SIIM-ISIC_Inception_V3": {
        "iter1": {
            "base_path": "lr_0.01_epochs_80_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1",
            "prev_path": "",
            "pred_file": "val_tensor_preds.pt",
            "gt_file": "val_tensor_y.pt",
            "output": "accuracy/g_outputs"
        },
        "iter2": {
            "base_path": "lr_0.01_epochs_80_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1",
            "prev_path": "cov_0.2",
            "pred_file": "val_tensor_preds.pt",
            "gt_file": "val_tensor_y.pt",
            "output": "accuracy/g_outputs"
        },
        "iter3": {
            "base_path": "lr_0.01_epochs_80_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1",
            "prev_path": "cov_0.2",
            "pred_file": "val_tensor_preds.pt",
            "gt_file": "val_tensor_y.pt",
            "output": "accuracy/g_outputs"
        },
        "iter4": {
            "base_path": "lr_0.01_epochs_80_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1",
            "prev_path": "cov_0.2",
            "pred_file": "val_tensor_preds.pt",
            "gt_file": "val_tensor_y.pt",
            "output": "accuracy/g_outputs"
        },
        "iter5": {
            "base_path": "lr_0.01_epochs_80_temperature-lens_0.7_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1",
            "prev_path": "cov_0.2",
            "pred_file": "val_tensor_preds.pt",
            "gt_file": "val_tensor_y.pt",
            "output": "accuracy/g_outputs"
        }
    }

}