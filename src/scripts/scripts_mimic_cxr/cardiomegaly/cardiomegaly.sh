#!/bin/sh
#SBATCH --output=Path/explainer/cardiomegaly/%j_bash_run_explainer.out
pwd
hostname
date

CURRENT=$(date +"%Y-%m-%d_%T")
echo $CURRENT

slurm_train_BB=Path/explainer/cardiomegaly/slurm_train_BB$CURRENT.out
slurm_train_t=Path/explainer/cardiomegaly/slurm_train_t$CURRENT.out
slurm_test_t=Path/explainer/cardiomegaly/slurm_test_t$CURRENT.out

slurm_train_g_1=Path/explainer/cardiomegaly/slurm_train_g_1$CURRENT.out
slurm_test_g_1=Path/explainer/cardiomegaly/slurm_test_g_1$CURRENT.out
slurm_fol_g_1=Path/explainer/cardiomegaly/slurm_fol_g_1$CURRENT.out
slurm_train_res_1=Path/explainer/cardiomegaly/slurm_train_res_1$CURRENT.out
slurm_test_res_1=Path/explainer/cardiomegaly/slurm_test_res_1$CURRENT.out

slurm_train_g_2=Path/explainer/cardiomegaly/slurm_train_g_2$CURRENT.out
slurm_test_g_2=Path/explainer/cardiomegaly/slurm_test_g_2$CURRENT.out
slurm_fol_g_2=Path/explainer/cardiomegaly/slurm_fol_g_2$CURRENT.out
slurm_train_res_2=Path/explainer/cardiomegaly/slurm_train_res_2$CURRENT.out
slurm_test_res_2=Path/explainer/cardiomegaly/slurm_test_res_2$CURRENT.out

slurm_train_g_3=Path/explainer/cardiomegaly/slurm_train_g_3$CURRENT.out
slurm_test_g_3=Path/explainer/cardiomegaly/slurm_test_g_3$CURRENT.out
slurm_fol_g_3=Path/explainer/cardiomegaly/slurm_fol_g_3$CURRENT.out
slurm_train_res_3=Path/explainer/cardiomegaly/slurm_train_res_3$CURRENT.out
slurm_test_res_3=Path/explainer/cardiomegaly/slurm_test_res_3$CURRENT.out

slurm_train_g_4=Path/explainer/cardiomegaly/slurm_train_g_4$CURRENT.out
slurm_test_g_4=Path/explainer/cardiomegaly/slurm_test_g_4$CURRENT.out
slurm_fol_g_4=Path/explainer/cardiomegaly/slurm_fol_g_4$CURRENT.out
slurm_train_res_4=Path/explainer/cardiomegaly/slurm_train_res_4$CURRENT.out
slurm_test_res_4=Path/explainer/cardiomegaly/slurm_test_res_4$CURRENT.out

slurm_train_g_5=Path/explainer/cardiomegaly/slurm_train_g_5$CURRENT.out
slurm_test_g5=Path/explainer/cardiomegaly/slurm_test_g5$CURRENT.out
slurm_fol_g_5=Path/explainer/cardiomegaly/slurm_fol_g_5$CURRENT.out
slurm_train_res_5=Path/explainer/cardiomegaly/slurm_train_res_5$CURRENT.out
slurm_test_res_5=Path/explainer/cardiomegaly/slurm_test_res_5$CURRENT.out

slurm_all_perf=Path/explainer/cardiomegaly/slurm_all_perf$CURRENT.out
slurm_res_v_BB=Path/explainer/cardiomegaly/slurm_res_v_BB$CURRENT.out

echo "MIMIC_CXR"
source Path/anaconda3/etc/profile.d/conda.sh
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
  --selected-obs="cardiomegaly" \
  --labels "0 (No Cardiomegaly)" "1 (Cardiomegaly)" >$slurm_train_BB

##########################################
# t
##########################################
python ../../../codebase/train_t_mimic_cxr.py \
  --arch='densenet121' \
  --workers=5 \
  --epochs=60 \
  --start-epoch=0 \
  --batch-size=18 \
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
  --selected-obs="cardiomegaly" \
  --labels "0 (No Cardiomegaly)" "1 (Cardiomegaly)">$slurm_train_t

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
  --selected-obs="cardiomegaly"\
  --labels "0 (No Cardiomegaly)" "1 (Cardiomegaly)">$slurm_test_t

##############################
# sub-select the concepts whose validation auroc >= 0.7 using the following notebook
# ./src/codebase/jupyter_notebook/MIMIC-CXR/Cardiomegaly/Sub-Select-Concepts.ipynb
##############################

##########################################
#### MoIE
##########################################
# # iter 1
#---------------------------------
# Train explainer

python ../../../codebase/train_explainer_mimic_cxr.py \
  --iter 1 \
  --with_seed "y" \
  --expert-to-train "explainer" \
  --dataset "mimic_cxr" \
  --cov 0.2 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 1024.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --metric "auroc" \
  --arch "densenet121" \
  --selected-obs="cardiomegaly" \
  --labels "0 (No Cardiomegaly)" "1 (Cardiomegaly)" >$slurm_train_g_1

python ../../../codebase/test_explainer_mimic_cxr.py \
  --iter 1 \
  --with_seed "y" \
  --expert-to-train "explainer" \
  --dataset "mimic_cxr" \
  --cov 0.2 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 1024.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --metric "auroc" \
  --arch "densenet121" \
  --checkpoint-model "model_seq_epoch_384.pth.tar" \
  --selected-obs="cardiomegaly" \
  --labels "0 (No Cardiomegaly)" "1 (Cardiomegaly)">$slurm_test_g_1

python ../../../codebase/FOLs_mimic_cxr_main.py --iteration 1 --disease "cardiomegaly" --model "MoIE">$slurm_fol_g_1

# # train_residual
python ../../../codebase/train_explainer_mimic_cxr.py \
  --iter 1 \
  --epochs-residual 1 \
  --with_seed "y" \
  --expert-to-train "residual" \
  --dataset "mimic_cxr" \
  --cov 0.2 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 1024.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --metric "auroc" \
  --bb-chkpt-folder "lr_0.01_epochs_60_loss_CE" \
  --checkpoint-bb "g_best_model_epoch_4.pth.tar" \
  --checkpoint-model "model_seq_epoch_384.pth.tar" \
  --arch "densenet121" \
  --selected-obs="cardiomegaly" \
  --labels "0 (No Cardiomegaly)" "1 (Cardiomegaly)" >$slurm_train_res_1

python ../../../codebase/test_explainer_mimic_cxr.py \
  --iter 1 \
  --epochs-residual 1 \
  --with_seed "y" \
  --expert-to-train "residual" \
  --dataset "mimic_cxr" \
  --cov 0.2 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 1024.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --metric "auroc" \
  --arch "densenet121" \
  --bb-chkpt-folder "lr_0.01_epochs_60_loss_CE" \
  --checkpoint-bb "g_best_model_epoch_4.pth.tar" \
  --checkpoint-model "model_seq_epoch_384.pth.tar" \
  --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" \
  --selected-obs="cardiomegaly" \
  --labels "0 (No Cardiomegaly)" "1 (Cardiomegaly)" >$slurm_test_res_1

#######################################
# iter2
#######################################
# train g
python ../../../codebase/train_explainer_mimic_cxr.py \
  --iter 2 \
  --with_seed "y" \
  --expert-to-train "explainer" \
  --dataset "mimic_cxr" \
  --cov 0.2 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 1024.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --metric "auroc" \
  --prev_chk_pt_explainer_folder "{soft_hard_filter}_concepts/seed_{seed}/explainer/cardiomegaly/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.2_alpha_0.5_selection-threshold_0.5_lm_1024.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
  --checkpoint-model "model_seq_epoch_384.pth.tar" \
  --arch "densenet121" \
  --selected-obs="cardiomegaly" \
  --labels "0 (No Cardiomegaly)" "1 (Cardiomegaly)" >$slurm_train_g_2

# # # test_g

python ../../../codebase/test_explainer_mimic_cxr.py \
  --iter 2 \
  --with_seed "y" \
  --expert-to-train "explainer" \
  --dataset "mimic_cxr" \
  --cov 0.2 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 1024.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --arch "densenet121" \
  --metric "auroc" \
  --prev_chk_pt_explainer_folder "{soft_hard_filter}_concepts/seed_{seed}/explainer/cardiomegaly/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.2_alpha_0.5_selection-threshold_0.5_lm_1024.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
  --checkpoint-model "model_seq_epoch_384.pth.tar" "model_seq_epoch_41.pth.tar" \
  --arch "densenet121" \
  --selected-obs="cardiomegaly" \
  --labels "0 (No Cardiomegaly)" "1 (Cardiomegaly)" >$slurm_test_g_2

python ../../../codebase/FOLs_mimic_cxr_main.py --iteration 2 --disease "cardiomegaly" --model "MoIE" >$slurm_fol_g_2

python ../../../codebase/train_explainer_mimic_cxr.py \
  --iter 2 \
  --with_seed "y" \
  --expert-to-train "residual" \
  --dataset "mimic_cxr" \
  --cov 0.2 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 1024.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --metric "auroc" \
  --prev_chk_pt_explainer_folder "{soft_hard_filter}_concepts/seed_{seed}/explainer/cardiomegaly/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.2_alpha_0.5_selection-threshold_0.5_lm_1024.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
  --checkpoint-model "model_seq_epoch_384.pth.tar" "model_seq_epoch_41.pth.tar" \
  --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" \
  --arch "densenet121" \
  --checkpoint-bb "g_best_model_epoch_4.pth.tar" \
  --selected-obs="cardiomegaly" \
  --labels "0 (No Cardiomegaly)" "1 (Cardiomegaly)" >$slurm_train_res_2

python ../../../codebase/test_explainer_mimic_cxr.py \
  --iter 2 \
  --with_seed "y" \
  --expert-to-train "residual" \
  --dataset "mimic_cxr" \
  --cov 0.2 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 1024.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --metric "auroc" \
  --arch "densenet121" \
  --bb-chkpt-folder "lr_0.01_epochs_60_loss_CE" \
  --checkpoint-bb "g_best_model_epoch_4.pth.tar" \
  --prev_chk_pt_explainer_folder "{soft_hard_filter}_concepts/seed_{seed}/explainer/cardiomegaly/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.2_alpha_0.5_selection-threshold_0.5_lm_1024.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
  --checkpoint-model "model_seq_epoch_384.pth.tar" "model_seq_epoch_41.pth.tar" \
  --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" \
  --selected-obs="cardiomegaly" \
  --labels "0 (No Cardiomegaly)" "1 (Cardiomegaly)" >$slurm_test_res_2

# #######################################
# # iter3
# #######################################
# # train g
python ../../../codebase/train_explainer_mimic_cxr.py \
  --iter 3 \
  --expert-to-train "explainer" \
  --with_seed "y" \
  --dataset "mimic_cxr" \
  --cov 0.35 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 1024.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --metric "auroc" \
  --prev_chk_pt_explainer_folder "{soft_hard_filter}_concepts/seed_{seed}/explainer/cardiomegaly/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.2_alpha_0.5_selection-threshold_0.5_lm_1024.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "{soft_hard_filter}_concepts/seed_{seed}/explainer/cardiomegaly/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.2_alpha_0.5_selection-threshold_0.5_lm_1024.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
  --checkpoint-model "model_seq_epoch_384.pth.tar" "model_seq_epoch_41.pth.tar" \
  --arch "densenet121" \
  --selected-obs="cardiomegaly" \
  --labels "0 (No Cardiomegaly)" "1 (Cardiomegaly)" >$slurm_train_g_3

python ../../../codebase/test_explainer_mimic_cxr.py \
  --iter 3 \
  --expert-to-train "explainer" \
  --with_seed "y" \
  --dataset "mimic_cxr" \
  --cov 0.35 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 1024.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --metric "auroc" \
  --prev_chk_pt_explainer_folder "{soft_hard_filter}_concepts/seed_{seed}/explainer/cardiomegaly/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.2_alpha_0.5_selection-threshold_0.5_lm_1024.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "{soft_hard_filter}_concepts/seed_{seed}/explainer/cardiomegaly/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.2_alpha_0.5_selection-threshold_0.5_lm_1024.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
  --checkpoint-model "model_seq_epoch_384.pth.tar" "model_seq_epoch_41.pth.tar" "model_seq_epoch_169.pth.tar" \
  --arch "densenet121" \
  --selected-obs="cardiomegaly" \
  --labels "0 (No Cardiomegaly)" "1 (Cardiomegaly)" >$slurm_test_g_3

python ../../../codebase/FOLs_mimic_cxr_main.py --iteration 3 --disease "cardiomegaly" --model "MoIE" >$slurm_fol_g_3

python ../../../codebase/train_explainer_mimic_cxr.py \
  --iter 3 \
  --expert-to-train "residual" \
  --with_seed "y" \
  --dataset "mimic_cxr" \
  --cov 0.35 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 1024.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --metric "auroc" \
  --bb-chkpt-folder "lr_0.01_epochs_60_loss_CE" \
  --checkpoint-bb "g_best_model_epoch_4.pth.tar" \
  --prev_chk_pt_explainer_folder "{soft_hard_filter}_concepts/seed_{seed}/explainer/cardiomegaly/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.2_alpha_0.5_selection-threshold_0.5_lm_1024.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "{soft_hard_filter}_concepts/seed_{seed}/explainer/cardiomegaly/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.2_alpha_0.5_selection-threshold_0.5_lm_1024.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
  --checkpoint-model "model_seq_epoch_384.pth.tar" "model_seq_epoch_41.pth.tar" "model_seq_epoch_169.pth.tar" \
  --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" \
  --arch "densenet121" \
  --selected-obs="cardiomegaly" \
  --labels "0 (No Cardiomegaly)" "1 (Cardiomegaly)" >$slurm_train_res_3

python ../../../codebase/test_explainer_mimic_cxr.py \
  --iter 3 \
  --expert-to-train "residual" \
  --with_seed "y" \
  --dataset "mimic_cxr" \
  --cov 0.35 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 1024.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --metric "auroc" \
  --bb-chkpt-folder "lr_0.01_epochs_60_loss_CE" \
  --checkpoint-bb "g_best_model_epoch_4.pth.tar" \
  --prev_chk_pt_explainer_folder "{soft_hard_filter}_concepts/seed_{seed}/explainer/cardiomegaly/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.2_alpha_0.5_selection-threshold_0.5_lm_1024.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "{soft_hard_filter}_concepts/seed_{seed}/explainer/cardiomegaly/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.2_alpha_0.5_selection-threshold_0.5_lm_1024.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
  --checkpoint-model "model_seq_epoch_384.pth.tar" "model_seq_epoch_41.pth.tar" "model_seq_epoch_169.pth.tar" \
  --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" \
  --arch "densenet121" \
  --selected-obs="cardiomegaly" \
  --labels "0 (No Cardiomegaly)" "1 (Cardiomegaly)" >$slurm_test_res_3

# #######################################
# # iter4
# #######################################
# # train g

python ../../../codebase/train_explainer_mimic_cxr.py \
  --iter 4 \
  --expert-to-train "explainer" \
  --with_seed "y" \
  --dataset "mimic_cxr" \
  --cov 0.3 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 1024.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --metric "auroc" \
  --prev_chk_pt_explainer_folder "{soft_hard_filter}_concepts/seed_{seed}/explainer/cardiomegaly/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.2_alpha_0.5_selection-threshold_0.5_lm_1024.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "{soft_hard_filter}_concepts/seed_{seed}/explainer/cardiomegaly/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.2_alpha_0.5_selection-threshold_0.5_lm_1024.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "{soft_hard_filter}_concepts/seed_{seed}/explainer/cardiomegaly/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.35_alpha_0.5_selection-threshold_0.5_lm_1024.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
  --checkpoint-model "model_seq_epoch_384.pth.tar" "model_seq_epoch_41.pth.tar" "model_seq_epoch_169.pth.tar" \
  --arch "densenet121" \
  --selected-obs="cardiomegaly" \
  --labels "0 (No Cardiomegaly)" "1 (Cardiomegaly)" >$slurm_train_g_4

python ../../../codebase/test_explainer_mimic_cxr.py \
  --iter 4 \
  --expert-to-train "explainer" \
  --with_seed "y" \
  --dataset "mimic_cxr" \
  --cov 0.3 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 1024.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --metric "auroc" \
  --prev_chk_pt_explainer_folder "{soft_hard_filter}_concepts/seed_{seed}/explainer/cardiomegaly/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.2_alpha_0.5_selection-threshold_0.5_lm_1024.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "{soft_hard_filter}_concepts/seed_{seed}/explainer/cardiomegaly/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.2_alpha_0.5_selection-threshold_0.5_lm_1024.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "{soft_hard_filter}_concepts/seed_{seed}/explainer/cardiomegaly/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.35_alpha_0.5_selection-threshold_0.5_lm_1024.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
  --checkpoint-model "model_seq_epoch_384.pth.tar" "model_seq_epoch_41.pth.tar" "model_seq_epoch_169.pth.tar" "model_seq_epoch_139.pth.tar" \
  --arch "densenet121" \
  --selected-obs="cardiomegaly" \
  --labels "0 (No Cardiomegaly)" "1 (Cardiomegaly)">$slurm_test_g_4

python ../../../codebase/FOLs_mimic_cxr_main.py --iteration 4 --disease "cardiomegaly" --model "MoIE" >$slurm_fol_g_4

python ../../../codebase/train_explainer_mimic_cxr.py \
  --iter 4 \
  --expert-to-train "residual" \
  --with_seed "y" \
  --dataset "mimic_cxr" \
  --cov 0.3 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 1024.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --metric "auroc" \
  --bb-chkpt-folder "lr_0.01_epochs_60_loss_CE" \
  --checkpoint-bb "g_best_model_epoch_4.pth.tar" \
  --prev_chk_pt_explainer_folder "{soft_hard_filter}_concepts/seed_{seed}/explainer/cardiomegaly/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.2_alpha_0.5_selection-threshold_0.5_lm_1024.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "{soft_hard_filter}_concepts/seed_{seed}/explainer/cardiomegaly/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.2_alpha_0.5_selection-threshold_0.5_lm_1024.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "{soft_hard_filter}_concepts/seed_{seed}/explainer/cardiomegaly/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.35_alpha_0.5_selection-threshold_0.5_lm_1024.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
  --checkpoint-model "model_seq_epoch_384.pth.tar" "model_seq_epoch_41.pth.tar" "model_seq_epoch_169.pth.tar" "model_seq_epoch_139.pth.tar" \
  --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" \
  --arch "densenet121" \
  --selected-obs="cardiomegaly" \
  --labels "0 (No Cardiomegaly)" "1 (Cardiomegaly)" >$slurm_train_res_4


python ../../../codebase/test_explainer_mimic_cxr.py \
  --iter 4 \
  --expert-to-train "residual" \
  --with_seed "y" \
  --dataset "mimic_cxr" \
  --cov 0.3 \
  --bs 1028 \
  --dataset-folder-concepts "lr_0.01_epochs_60_loss_BCE_W_flattening_type_flatten_layer_features_denseblock4" \
  --input-size-pi 2048 \
  --optim "SGD" \
  --lr 0.01 \
  --temperature-lens 7.6 \
  --lm 1024.0 \
  --lambda-lens 0.0001 \
  --alpha-KD 0.99 \
  --temperature-KD 20.0 \
  --hidden-nodes 20 20 \
  --layer "layer4" \
  --metric "auroc" \
  --bb-chkpt-folder "lr_0.01_epochs_60_loss_CE" \
  --checkpoint-bb "g_best_model_epoch_4.pth.tar" \
  --prev_chk_pt_explainer_folder "{soft_hard_filter}_concepts/seed_{seed}/explainer/cardiomegaly/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.2_alpha_0.5_selection-threshold_0.5_lm_1024.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "{soft_hard_filter}_concepts/seed_{seed}/explainer/cardiomegaly/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.2_alpha_0.5_selection-threshold_0.5_lm_1024.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" "{soft_hard_filter}_concepts/seed_{seed}/explainer/cardiomegaly/densenet121_1028_lr_0.01_SGD_temperature-lens_7.6_cov_0.35_alpha_0.5_selection-threshold_0.5_lm_1024.0_lambda-lens_0.0001_alpha-KD_0.99_temperature-KD_20.0_hidden-layers_2020_input-size-pi_2048_layer_layer4" \
  --checkpoint-model "model_seq_epoch_384.pth.tar" "model_seq_epoch_41.pth.tar" "model_seq_epoch_169.pth.tar" "model_seq_epoch_139.pth.tar" \
  --checkpoint-residual "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" "model_residual_best_model_epoch_1.pth.tar" \
  --arch "densenet121" \
  --selected-obs="cardiomegaly" \
  --labels "0 (No Cardiomegaly)" "1 (Cardiomegaly)"  >$slurm_test_res_4


# All Performance
python ../../../codebase/performance_calculation_mimic_cxr_main.py --iterations 4 --disease "cardiomegaly" --model "MoIE" >$slurm_all_perf

# Residual vs BB
python ../../../codebase/performance_calculation_mimic_cxr_residual_main.py --iterations 4 --disease "cardiomegaly" --model "MoIE" >$slurm_res_v_BB



