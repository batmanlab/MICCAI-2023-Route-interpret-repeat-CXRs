#!/bin/sh
#SBATCH --output=Path/explainer/edema/%j_bash_run_explainer.out
pwd
hostname
date

CURRENT=$(date +"%Y-%m-%d_%T")
echo $CURRENT

slurm_train_BB=Path/explainer/edema/slurm_train_BB$CURRENT.out
slurm_train_t=Path/explainer/edema/slurm_train_t$CURRENT.out
slurm_test_t=Path/explainer/edema/slurm_test_t$CURRENT.out

slurm_train_g_1=Path/explainer/edema/slurm_train_g_1$CURRENT.out
slurm_test_g_1=Path/explainer/edema/slurm_test_g_1$CURRENT.out
slurm_fol_g_1=Path/explainer/edema/slurm_fol_g_1$CURRENT.out
slurm_train_res_1=Path/explainer/edema/slurm_train_res_1$CURRENT.out
slurm_test_res_1=Path/explainer/edema/slurm_test_res_1$CURRENT.out

slurm_train_g_2=Path/explainer/edema/slurm_train_g_2$CURRENT.out
slurm_test_g_2=Path/explainer/edema/slurm_test_g_2$CURRENT.out
slurm_fol_g_2=Path/explainer/edema/slurm_fol_g_2$CURRENT.out
slurm_train_res_2=Path/explainer/edema/slurm_train_res_2$CURRENT.out
slurm_test_res_2=Path/explainer/edema/slurm_test_res_2$CURRENT.out

slurm_train_g_3=Path/explainer/edema/slurm_train_g_3$CURRENT.out
slurm_test_g_3=Path/explainer/edema/slurm_test_g_3$CURRENT.out
slurm_fol_g_3=Path/explainer/edema/slurm_fol_g_3$CURRENT.out
slurm_train_res_3=Path/explainer/edema/slurm_train_res_3$CURRENT.out
slurm_test_res_3=Path/explainer/edema/slurm_test_res_3$CURRENT.out

slurm_train_g_4=Path/explainer/edema/slurm_train_g_4$CURRENT.out
slurm_test_g_4=Path/explainer/edema/slurm_test_g_4$CURRENT.out
slurm_fol_g_4=Path/explainer/edema/slurm_fol_g_4$CURRENT.out
slurm_train_res_4=Path/explainer/edema/slurm_train_res_4$CURRENT.out
slurm_test_res_4=Path/explainer/edema/slurm_test_res_4$CURRENT.out

slurm_train_g_5=Path/explainer/edema/slurm_train_g_5$CURRENT.out
slurm_test_g5=Path/explainer/edema/slurm_test_g5$CURRENT.out
slurm_fol_g_5=Path/explainer/edema/slurm_fol_g_5$CURRENT.out
slurm_train_res_5=Path/explainer/edema/slurm_train_res_5$CURRENT.out
slurm_test_res_5=Path/explainer/edema/slurm_test_res_5$CURRENT.out

slurm_all_perf=Path/explainer/edema/slurm_all_perf$CURRENT.out
slurm_res_v_BB=Path/explainer/edema/slurm_res_v_BB$CURRENT.out

echo "MIMIC_CXR"
source Path_of_conda/anaconda3/etc/profile.d/conda.sh
# conda activate python_3_7

conda activate python_3_7_rtx_6000

##########################################
# BB
##########################################
python ../../../codebase/train_BB_mimic_cxr.py \
  --arch='densenet121' \
  --workers=5 \
  --epochs=60 \
  --start-epoch=0 \
  --batch-size=16 \
  --learning-rate=0.01 \
  --resize=512 \
  --resume='' \
  --loss="CE"\
  --gpu=0 \
  --world-size=1 \
  --rank=0 \
  --ngpus-per-node=2 \
  --selected-obs="edema" \
  --labels "0 (No Edema)" "1 (Edema)" >$slurm_train_BB

##########################################
# t
##########################################
python ../../../codebase/train_t_mimic_cxr.py \
  --arch='densenet121' \
  --workers=5 \
  --epochs=60 \
  --start-epoch=0 \
  --batch-size=2 \
  --learning-rate=0.01 \
  --loss1="BCE_W"\
  --resize=512 \
  --resume='' \
  --gpu=0 \
  --world-size=1 \
  --rank=0 \
  --ngpus-per-node=2 \
  --bb-chkpt-folder="lr_0.01_epochs_60_loss_CE"\
  --checkpoint-bb="g_best_model_epoch_4.pth.tar"\
  --flattening-type="flatten" \
  --layer="features_denseblock4" \
  --selected-obs="edema" \
  --labels "0 (No Edema)" "1 (Edema)">$slurm_train_t

python ../../../codebase/test_t_mimic_cxr.py \
  --arch='densenet121' \
  --workers=5 \
  --epochs=60 \
  --start-epoch=0 \
  --batch-size=16 \
  --learning-rate=0.01 \
  --loss1="BCE_W"\
  --resize=512 \
  --resume='' \
  --gpu=0 \
  --world-size=1 \
  --rank=0 \
  --ngpus-per-node=2 \
  --bb-chkpt-folder="lr_0.01_epochs_60_loss_CE"\
  --checkpoint-bb="g_best_model_epoch_4.pth.tar"\
  --flattening-type="flatten" \
  --layer="features_denseblock4" \
  --checkpoint-t="g_best_model_epoch_16.pth.tar"\
  --selected-obs="edema"\
  --labels "0 (No edema)" "1 (edema)">$slurm_test_t

##############################
# sub-select the concepts whose validation auroc >= 0.7 using the following notebook
# ./src/codebase/jupyter_notebook/MIMIC-CXR/Edema/Sub-Select-Concepts.ipynb
##############################

#######################################
# iter1
#######################################
# Train explainerß
python ../../../codebase/train_explainer_mimic_cxr.py \
  --iter 1 \
  --with_seed "y" \
  --expert-to-train "explainer" \
  --dataset "mimic_cxr" \
  --cov 0.25 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 128.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --arch "densenet121" \
  --selected-obs="edema" \
  --labels "0 (No Edema)" "1 (Edema)" >$slurm_train_g_1

python ../../../codebase/test_explainer_mimic_cxr.py \
  --iter 1 \
  --with_seed "y" \
  --expert-to-train "explainer" \
  --dataset "mimic_cxr" \
  --cov 0.25 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 128.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --arch "densenet121" \
  --selected-obs="edema" \
  --checkpoint-model "model_seq_epoch_16.pth.tar" \
  --labels "0 (No Edema)" "1 (Edema)" >$slurm_test_g_1

python ../../../codebase/FOLs_mimic_cxr_main.py --iteration 1 --disease "edema" --model "MoIE" >$slurm_fol_g_1

# train_residual
python ../../../codebase/train_explainer_mimic_cxr.py \
  --iter 1 \
  --with_seed "y" \
  --expert-to-train "residual" \
  --dataset "mimic_cxr" \
  --cov 0.25 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 128.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --metric "auroc" \
  --bb-chkpt-folder "lr_0.01_epochs_60_loss_CE" \
  --checkpoint-bb "g_best_model_epoch_4.pth.tar" \
  --checkpoint-model "model_seq_epoch_16.pth.tar" \
  --arch "densenet121" \
  --selected-obs="edema" \
  --labels "0 (No Edema)" "1 (Edema)" >$slurm_train_res_1

python ../../../codebase/test_explainer_mimic_cxr.py \
  --iter 1 \
  --with_seed "y" \
  --expert-to-train "residual" \
  --dataset "mimic_cxr" \
  --cov 0.25 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 128.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --metric "auroc" \
  --bb-chkpt-folder "lr_0.01_epochs_60_loss_CE" \
  --checkpoint-bb "g_best_model_epoch_4.pth.tar" \
  --checkpoint-model "model_seq_epoch_16.pth.tar" \
  --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" \
  --arch "densenet121" \
  --selected-obs="edema" \
  --labels "0 (No Edema)" "1 (Edema)" >$slurm_test_res_1

#######################################
# iter2
#######################################
python ../../../codebase/train_explainer_mimic_cxr.py \
  --iter 2 \
  --with_seed "y" \
  --expert-to-train "explainer" \
  --dataset "mimic_cxr" \
  --cov 0.25 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 128.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --arch "densenet121" \
  --selected-obs="edema" \
  --prev_chk_pt_explainer_folder "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
  --checkpoint-model "model_seq_epoch_16.pth.tar" \
  --labels "0 (No Edema)" "1 (Edema)" >$slurm_train_g_2

python ../../../codebase/test_explainer_mimic_cxr.py \
  --iter 2 \
  --with_seed "y" \
  --expert-to-train "explainer" \
  --dataset "mimic_cxr" \
  --cov 0.25 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 128.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --arch "densenet121" \
  --selected-obs="edema" \
  --prev_chk_pt_explainer_folder "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
  --checkpoint-model "model_seq_epoch_16.pth.tar" "model_seq_epoch_24.pth.tar" \
  --labels "0 (No Edema)" "1 (Edema)" >$slurm_test_g_2

python ../../../codebase/FOLs_mimic_cxr_main.py --iteration 2 --disease "edema" --model "MoIE" >$slurm_fol_g_2

python ../../../codebase/train_explainer_mimic_cxr.py \
  --iter 2 \
  --with_seed "y" \
  --expert-to-train "residual" \
  --dataset "mimic_cxr" \
  --cov 0.25 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 128.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --arch "densenet121" \
  --selected-obs="edema" \
  --prev_chk_pt_explainer_folder "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
  --checkpoint-model "model_seq_epoch_16.pth.tar" "model_seq_epoch_24.pth.tar" \
  --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" \
  --checkpoint-bb "g_best_model_epoch_4.pth.tar" \
  --labels "0 (No Edema)" "1 (Edema)" >$slurm_train_res_2

python ../../../codebase/test_explainer_mimic_cxr.py \
  --iter 2 \
  --with_seed "y" \
  --expert-to-train "residual" \
  --dataset "mimic_cxr" \
  --cov 0.25 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 128.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --arch "densenet121" \
  --selected-obs="edema" \
  --prev_chk_pt_explainer_folder "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
  --checkpoint-model "model_seq_epoch_16.pth.tar" "model_seq_epoch_24.pth.tar" \
  --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" \
  --checkpoint-bb "g_best_model_epoch_4.pth.tar" \
  --labels "0 (No Edema)" "1 (Edema)" >$slurm_test_res_2

#######################################
# iter3
#######################################
python ../../../codebase/train_explainer_mimic_cxr.py \
  --iter 3 \
  --with_seed "y" \
  --expert-to-train "explainer" \
  --dataset "mimic_cxr" \
  --cov 0.25 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 128.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --arch "densenet121" \
  --selected-obs="edema" \
  --prev_chk_pt_explainer_folder "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
  --checkpoint-model "model_seq_epoch_16.pth.tar" "model_seq_epoch_24.pth.tar" \
  --labels "0 (No Edema)" "1 (Edema)" >$slurm_train_g_3

python ../../../codebase/test_explainer_mimic_cxr.py \
  --iter 3 \
  --with_seed "y" \
  --expert-to-train "explainer" \
  --dataset "mimic_cxr" \
  --cov 0.25 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 128.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --arch "densenet121" \11_
--selected-obs="edema" \
  --prev_chk_pt_explainer_folder "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
  --checkpoint-model "model_seq_epoch_16.pth.tar" "model_seq_epoch_24.pth.tar" "model_seq_epoch_124.pth.tar" \
  --labels "0 (No Edema)" "1 (Edema)" >$slurm_test_g_3

python ../../../codebase/FOLs_mimic_cxr_main.py --iteration 3 --disease "edema" --model "MoIE" >$slurm_fol_g_3

python ../../../codebase/train_explainer_mimic_cxr.py \
  --iter 3 \
  --with_seed "y" \
  --expert-to-train "residual" \
  --dataset "mimic_cxr" \
  --cov 0.25 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 128.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --arch "densenet121" \
  --selected-obs="edema" \
  --prev_chk_pt_explainer_folder "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
  --checkpoint-model "model_seq_epoch_16.pth.tar" "model_seq_epoch_24.pth.tar" "model_seq_epoch_124.pth.tar" \
  --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" \
  --checkpoint-bb "g_best_model_epoch_4.pth.tar" \
  --labels "0 (No Edema)" "1 (Edema)" >$slurm_train_res_3

python ../../../codebase/test_explainer_mimic_cxr.py \
  --iter 3 \
  --with_seed "y" \
  --expert-to-train "residual" \
  --dataset "mimic_cxr" \
  --cov 0.25 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 128.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --arch "densenet121" \
  --selected-obs="edema" \
  --prev_chk_pt_explainer_folder "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
  --checkpoint-model "model_seq_epoch_16.pth.tar" "model_seq_epoch_24.pth.tar" "model_seq_epoch_124.pth.tar" \
  --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" \
  --checkpoint-bb "g_best_model_epoch_4.pth.tar" \
  --labels "0 (No Edema)" "1 (Edema)" >$slurm_test_res_3


python ../../../codebase/train_explainer_mimic_cxr.py \
  --iter 4 \
  --with_seed "y" \
  --expert-to-train "explainer" \
  --dataset "mimic_cxr" \
  --cov 0.25 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 128.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --arch "densenet121" \
  --selected-obs="edema" \
  --prev_chk_pt_explainer_folder "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
  --checkpoint-model "model_seq_epoch_16.pth.tar" "model_seq_epoch_24.pth.tar" "model_seq_epoch_124.pth.tar" \
  --labels "0 (No Edema)" "1 (Edema)" >$slurm_train_g_4

python ../../../codebase/test_explainer_mimic_cxr.py \
  --iter 4 \
  --with_seed "y" \
  --expert-to-train "explainer" \
  --dataset "mimic_cxr" \
  --cov 0.25 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 128.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --arch "densenet121" \
  --selected-obs="edema" \
  --prev_chk_pt_explainer_folder "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
  --checkpoint-model "model_seq_epoch_16.pth.tar" "model_seq_epoch_24.pth.tar" "model_seq_epoch_124.pth.tar" "model_seq_epoch_149.pth.tar" \
  --labels "0 (No Edema)" "1 (Edema)" >$slurm_test_g_4

python ../../../codebase/FOLs_mimic_cxr_main.py --iteration 4 --disease "edema" --model "MoIE" >$slurm_fol_g_4

python ../../../codebase/train_explainer_mimic_cxr.py \
  --iter 4 \
  --with_seed "y" \
  --expert-to-train "residual" \
  --dataset "mimic_cxr" \
  --cov 0.25 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 128.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --arch "densenet121" \
  --selected-obs="edema" \
  --prev_chk_pt_explainer_folder "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
  --checkpoint-model "model_seq_epoch_16.pth.tar" "model_seq_epoch_24.pth.tar" "model_seq_epoch_124.pth.tar" "model_seq_epoch_149.pth.tar" \
  --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" \
  --checkpoint-bb "g_best_model_epoch_4.pth.tar" \
  --labels "0 (No Edema)" "1 (Edema)" >$slurm_train_res_4

python ../../../codebase/train_explainer_mimic_cxr.py \
  --iter 4 \
  --with_seed "y" \
  --expert-to-train "residual" \
  --dataset "mimic_cxr" \
  --cov 0.25 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 128.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --arch "densenet121" \
  --selected-obs="edema" \
  --prev_chk_pt_explainer_folder "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
  --checkpoint-model "model_seq_epoch_16.pth.tar" "model_seq_epoch_24.pth.tar" "model_seq_epoch_124.pth.tar" "model_seq_epoch_149.pth.tar" \
  --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" \
  --checkpoint-bb "g_best_model_epoch_4.pth.tar" \
  --labels "0 (No Edema)" "1 (Edema)" >$slurm_test_res_4


python ../../../codebase/train_explainer_mimic_cxr.py \
  --iter 5 \
  --with_seed "y" \
  --expert-to-train "explainer" \
  --dataset "mimic_cxr" \
  --cov 0.15 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 128.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --arch "densenet121" \
  --selected-obs="edema" \
  --prev_chk_pt_explainer_folder "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
  --checkpoint-model "model_seq_epoch_16.pth.tar" "model_seq_epoch_24.pth.tar" "model_seq_epoch_124.pth.tar" "model_seq_epoch_149.pth.tar" \
  --labels "0 (No Edema)" "1 (Edema)" >$slurm_train_g_5

python ../../../codebase/test_explainer_mimic_cxr.py \
  --iter 5 \
  --with_seed "y" \
  --expert-to-train "explainer" \
  --dataset "mimic_cxr" \
  --cov 0.15 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 128.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --arch "densenet121" \
  --selected-obs="edema" \
  --prev_chk_pt_explainer_folder "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
  --checkpoint-model "model_seq_epoch_16.pth.tar" "model_seq_epoch_24.pth.tar" "model_seq_epoch_124.pth.tar" "model_seq_epoch_149.pth.tar" "model_seq_epoch_96.pth.tar" \
  --labels "0 (No Edema)" "1 (Edema)" >$slurm_test_g_5

python ../../../codebase/FOLs_mimic_cxr_main.py --iteration 5 --disease "edema" --model "MoIE" >$slurm_fol_g_5

python ../../../codebase/train_explainer_mimic_cxr.py \
  --iter 5 \
  --with_seed "y" \
  --expert-to-train "residual" \
  --dataset "mimic_cxr" \
  --cov 0.15 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 128.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --arch "densenet121" \
  --selected-obs="edema" \
  --prev_chk_pt_explainer_folder "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
  --checkpoint-model "model_seq_epoch_16.pth.tar" "model_seq_epoch_24.pth.tar" "model_seq_epoch_124.pth.tar" "model_seq_epoch_149.pth.tar" "model_seq_epoch_96.pth.tar" \
  --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" \
  --checkpoint-bb "g_best_model_epoch_4.pth.tar" \
  --labels "0 (No Edema)" "1 (Edema)" >$slurm_train_res_5

python ../../../codebase/test_explainer_mimic_cxr.py \
  --iter 5 \
  --with_seed "y" \
  --expert-to-train "residual" \
  --dataset "mimic_cxr" \
  --cov 0.15 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 128.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --arch "densenet121" \
  --selected-obs="edema" \
  --prev_chk_pt_explainer_folder "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "{soft_hard_filter}_concepts/seed_{seed}/explainer/edema/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.25_alpha_0.5_selection-threshold_0.5_lm_128.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
  --checkpoint-model "model_seq_epoch_16.pth.tar" "model_seq_epoch_24.pth.tar" "model_seq_epoch_124.pth.tar" "model_seq_epoch_149.pth.tar" "model_seq_epoch_96.pth.tar" \
  --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" \
  --checkpoint-bb "g_best_model_epoch_4.pth.tar" \
  --labels "0 (No Edema)" "1 (Edema)" >$slurm_test_res_5

# All Performance
python ../../../codebase/performance_calculation_mimic_cxr_main.py --iterations 5 --disease "edema" --model "MoIE" >$slurm_all_perf

# Residual vs BB
python ../../../codebase/performance_calculation_mimic_cxr_residual_main.py --iterations 5 --disease "edema" --model "MoIE" >$slurm_res_v_BB