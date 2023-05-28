#!/bin/sh
#SBATCH --output=Path/Stanford_CXR/moie/effusion/performance/effusion_15000_%j.out
pwd
hostname
date

CURRENT=$(date +"%Y-%m-%d_%T")
echo $CURRENT

slurm_output_bb = Path/Stanford_CXR/moie/cardiomegaly/performance/cardiomegaly_15000_bb_$CURRENT.out
slurm_output_train = Path/Stanford_CXR/moie/cardiomegaly/performance/cardiomegaly_15000_t_train_$CURRENT.out
slurm_output_test= Path/Stanford_CXR/moie/cardiomegaly/performance/cardiomegaly_15000_t_test_$CURRENT.out

slurm_output1=Path/Stanford_CXR/moie/effusion/performance/effusion_15000_iter_1_$CURRENT.out
slurm_output2=Path/Stanford_CXR/moie/effusion/performance/effusion_15000_iter_2_$CURRENT.out
slurm_output3=Path/Stanford_CXR/moie/effusion/performance/effusion_15000_iter_3_$CURRENT.out
slurm_output_residual=Path/Stanford_CXR/moie/effusion/performance/effusion_15000_iter_3_residual_$CURRENT.out
slurm_output_total=Path/Stanford_CXR/moie/effusion/performance/effusion_15000_performance_$CURRENT.out

echo "Stanford_CXR"
source Path/anaconda3/etc/profile.d/conda.sh
# conda activate python_3_7

conda activate python_3_7_rtx_6000

# Train BB of Stanford CXR
python ../../../codebase/train_BB_stanford_cxr.py --arch='densenet121' --selected-obs="effusion" --epochs 5 --labels "0 (No effusion)" "1 (effusion)"  --domain_transfer "y" --tot_samples 15000 --checkpoint-mimic-cxr "mimic_checkpoint" >$slurm_output_bb

# Train SSL
python ../../../codebase/train_t_ssl_main.py \
  --target_dataset "stanford_cxr" \
  --source_dataset "mimic_cxr" \
  --disease "effusion" \
  --source-checkpoint-bb_path "lr_0.01_epochs_60_loss_CE" \
  --source-checkpoint-bb "g_best_model_epoch_8.pth.tar" \
  --source-checkpoint-t-path "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --source-checkpoint-t "g_best_model_epoch_7.pth.tar" \
  --tot_samples 15000 >$slurm_output_train

# Test SSL to get concepts
python ../../../codebase/test_t_ssl_main.py \
    --target_dataset "stanford_cxr" \
  --source_dataset "mimic_cxr" \
  --disease "effusion" \
  --source-checkpoint-bb_path "lr_0.01_epochs_60_loss_CE" \
  --source-checkpoint-bb "g_best_model_epoch_8.pth.tar" \
  --source-checkpoint-t-path "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --source-checkpoint-t "g_best_model_epoch_7.pth.tar" \
  --tot_samples 15000 >$slurm_output_test

# Transfer MOIE-CXR from MIMIC-CXR to Stanford CXR
# if `--initialize_w_mimic` == "y", we finetune both the selectors (pi) and the experts (g).
# if `--initialize_w_mimic` == "n", we finetune only the selectors (pi), not the experts (g).

python ../../../codebase/train_explainer_cxr_domain_transfer.py \
  --target_dataset "stanford_cxr" \
  --disease "effusion" \
  --tot_samples 15000 \
  --batch-size 64 \
  --iter 1 --cov 0.4  --initialize_w_mimic "y" --epochs 5 > $slurm_output1


python ../../../codebase/train_explainer_cxr_domain_transfer.py \
  --target_dataset "stanford_cxr" \
  --disease "effusion" \
  --tot_samples 15000 \
  --batch-size 64 \
  --iter 2 --prev_covs 0.4 --cov 0.3 --initialize_w_mimic "y" --epochs 5 \
  --prev_chk_pt_explainer_folder "{soft_hard_filter}_concepts/seed_{seed}/effusion/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_96.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4_sample_15000" \
  --checkpoint-model "best_model.pth.tar" > $slurm_output2


python ../../../codebase/train_explainer_cxr_domain_transfer.py \
  --target_dataset "stanford_cxr" \
  --disease "effusion" \
  --tot_samples 15000 \
  --batch-size 64 \
  --iter 3 --prev_covs 0.4 0.3 --cov 0.3 --initialize_w_mimic "y" --epochs 5 \
  --prev_chk_pt_explainer_folder "{soft_hard_filter}_concepts/seed_{seed}/effusion/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_96.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4_sample_15000" "{soft_hard_filter}_concepts/seed_{seed}/effusion/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_96.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4_sample_15000" \
  --checkpoint-model "best_model.pth.tar" "best_model.pth.tar" > $slurm_output3

python ../../../codebase/train_explainer_cxr_domain_transfer.py \
  --target_dataset "stanford_cxr" \
  --expert-to-train "residual" \
  --disease "effusion" \
  --source-checkpoint-bb_path "lr_0.01_epochs_60_loss_CE" \
  --source-checkpoint-bb "g_best_model_epoch_8.pth.tar" \
  --tot_samples 15000 \
  --batch-size 4 \
  --iter 3 --prev_covs 0.4 0.3 --cov 0.3 --initialize_w_mimic "y" --epochs 5 \
  --prev_chk_pt_explainer_folder "{soft_hard_filter}_concepts/seed_{seed}/effusion/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_96.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4_sample_15000" "{soft_hard_filter}_concepts/seed_{seed}/effusion/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_96.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4_sample_15000" \
  --checkpoint-model "best_model.pth.tar" "best_model.pth.tar" > $slurm_output_residual

python ../../../codebase/performance_calculation_cxr_domain_transfer_main.py --iterations 3 --disease "effusion" --model "MoIE" --tot_samples 15000 --cov 0.4 0.3 0.3 --initialize_w_mimic "y"  > $slurm_output_total
