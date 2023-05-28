#---------------------------------
# # iter 1 
#---------------------------------
# lr 0.01
# cov 0.45
# Train explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_ham10k.py --iter 1 --expert-to-train "explainer" --dataset "HAM10k" --cov 0.45 0.35 --bs 32 --lr 0.01 0.001 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3"

# Test explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_ham10k.py --iter 1 --checkpoint-model "model_g_best_model_epoch_138.pth.tar" --expert-to-train "explainer" --dataset "HAM10k" --cov 0.45 0.35 --bs 32 --lr 0.01 0.001 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --arch "Inception_V3"


# Train residual
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_ham10k.py --iter 1 --checkpoint-model "model_g_best_model_epoch_138.pth.tar" --expert-to-train "residual" --dataset "HAM10k" --cov 0.45 0.35 --bs 32 --lr 0.01 0.001 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --arch "Inception_V3"



#---------------------------------
# # iter 2
#---------------------------------
# lr 0.01
# cov 0.45
# Train explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_ham10k.py --iter 2 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.45_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" --checkpoint-model "model_g_best_model_epoch_138.pth.tar" --checkpoint-residual "model_g_best_model_epoch_4.pth.tar" --expert-to-train "explainer" --dataset "HAM10k" --cov 0.45 0.2 --bs 32 --lr 0.01 0.001 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3"

# Test explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_ham10k.py --iter 2 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.45_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" --checkpoint-model "model_g_best_model_epoch_138.pth.tar" "model_g_best_model_epoch_402.pth.tar" --checkpoint-residual "model_g_best_model_epoch_4.pth.tar" --expert-to-train "explainer" --dataset "HAM10k" --cov 0.45 0.5 --bs 32 --lr 0.01 0.001 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --arch "Inception_V3"



# Train residual
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_ham10k.py --iter 2 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.45_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" --checkpoint-model "model_g_best_model_epoch_138.pth.tar" "model_g_best_model_epoch_402.pth.tar" --checkpoint-residual "model_g_best_model_epoch_4.pth.tar" --expert-to-train "residual" --dataset "HAM10k" --cov 0.45 0.5 --bs 32 --lr 0.01 0.001 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --arch "Inception_V3"


#---------------------------------
# # iter 3
#---------------------------------
# lr 0.01
# cov 0.45
# Train explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_ham10k.py --iter 3 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.45_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.45_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.5/iter2" --checkpoint-model "model_g_best_model_epoch_138.pth.tar" "model_g_best_model_epoch_402.pth.tar" --checkpoint-residual "model_g_best_model_epoch_4.pth.tar" "model_g_best_model_epoch_1.pth.tar"  --expert-to-train "explainer" --dataset "HAM10k" --cov 0.45 0.5 0.05 --bs 32 --lr 0.01 0.001 0.001 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3"

# Test explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_ham10k.py --iter 3 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.45_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.45_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.5/iter2" --checkpoint-model "model_g_best_model_epoch_138.pth.tar" "model_g_best_model_epoch_402.pth.tar" "model_g_best_model_epoch_8.pth.tar" --checkpoint-residual "model_g_best_model_epoch_4.pth.tar" "model_g_best_model_epoch_1.pth.tar" --expert-to-train "explainer" --dataset "HAM10k" --cov 0.45 0.5 0.05 --bs 32 --lr 0.01 0.001 0.001 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --arch "Inception_V3"



# Train residual
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_ham10k.py --iter 3 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.45_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.45_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.5/iter2" --checkpoint-model "model_g_best_model_epoch_138.pth.tar" "model_g_best_model_epoch_402.pth.tar" "model_g_best_model_epoch_8.pth.tar" --checkpoint-residual "model_g_best_model_epoch_4.pth.tar" "model_g_best_model_epoch_1.pth.tar" --expert-to-train "residual" --dataset "HAM10k" --cov 0.45 0.5 0.05 --bs 32 --lr 0.01 0.001 0.001 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --arch "Inception_V3"


#---------------------------------
# # iter 4
#---------------------------------
# lr 0.01
# cov 0.45
# Train explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_ham10k.py --iter 4 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.45_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.45_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.5/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.45_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.05/iter3" --checkpoint-model "model_g_best_model_epoch_138.pth.tar" "model_g_best_model_epoch_402.pth.tar" "model_g_best_model_epoch_8.pth.tar" --checkpoint-residual "model_g_best_model_epoch_4.pth.tar" "model_g_best_model_epoch_1.pth.tar"  "model_g_best_model_epoch_3.pth.tar" --expert-to-train "explainer" --dataset "HAM10k" --cov 0.45 0.5 0.05 0.01 --bs 32 --lr 0.01 0.001 0.001 0.001 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3"

# Test explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_ham10k.py --iter 4 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.45_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.45_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.5/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.45_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.05/iter3" --checkpoint-model "model_g_best_model_epoch_138.pth.tar" "model_g_best_model_epoch_402.pth.tar" "model_g_best_model_epoch_8.pth.tar" "model_g_best_model_epoch_13.pth.tar" --checkpoint-residual "model_g_best_model_epoch_4.pth.tar" "model_g_best_model_epoch_1.pth.tar" "model_g_best_model_epoch_3.pth.tar" --expert-to-train "explainer" --dataset "HAM10k" --cov 0.45 0.5 0.05 0.01 --bs 32 --lr 0.01 0.001 0.001 0.001 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --arch "Inception_V3"



# Train residual
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_ham10k.py --iter 4 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.45_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.45_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.5/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.45_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.05/iter3" --checkpoint-model "model_g_best_model_epoch_138.pth.tar" "model_g_best_model_epoch_402.pth.tar" "model_g_best_model_epoch_8.pth.tar" "model_g_best_model_epoch_13.pth.tar" --checkpoint-residual "model_g_best_model_epoch_4.pth.tar" "model_g_best_model_epoch_1.pth.tar" "model_g_best_model_epoch_3.pth.tar" --expert-to-train "residual" --dataset "HAM10k" --cov 0.45 0.5 0.05 0.01 --bs 32 --lr 0.01 0.001 0.001 0.001 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --arch "Inception_V3"


#---------------------------------
# # iter 5
#---------------------------------
# lr 0.01
# cov 0.45
# Train explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_ham10k.py --iter 5 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.45_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.45_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.5/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.45_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.05/iter3" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.45_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.01/iter4" --checkpoint-model "model_g_best_model_epoch_138.pth.tar" "model_g_best_model_epoch_402.pth.tar" "model_g_best_model_epoch_8.pth.tar" "model_g_best_model_epoch_13.pth.tar" --checkpoint-residual "model_g_best_model_epoch_4.pth.tar" "model_g_best_model_epoch_1.pth.tar"  "model_g_best_model_epoch_3.pth.tar" "model_g_best_model_epoch_2.pth.tar" --expert-to-train "explainer" --dataset "HAM10k" --cov 0.45 0.5 0.05 0.01 0.01 --bs 32 --lr 0.01 0.001 0.001 0.001 0.001 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --lm 64 --arch "Inception_V3"

# Test explainer
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/test_explainer_ham10k.py --iter 5 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.45_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.45_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.5/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.45_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.05/iter3" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.45_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.01/iter4" --checkpoint-model "model_g_best_model_epoch_138.pth.tar" "model_g_best_model_epoch_402.pth.tar" "model_g_best_model_epoch_8.pth.tar" "model_g_best_model_epoch_13.pth.tar" "model_g_best_model_epoch_4.pth.tar" --checkpoint-residual "model_g_best_model_epoch_4.pth.tar" "model_g_best_model_epoch_1.pth.tar" "model_g_best_model_epoch_3.pth.tar" "model_g_best_model_epoch_2.pth.tar" --expert-to-train "explainer" --dataset "HAM10k" --cov 0.45 0.5 0.05 0.01 0.01 --bs 32 --lr 0.01 0.001 0.001 0.001 0.001 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --arch "Inception_V3"



# Train residual
python /ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase/train_explainer_ham10k.py --iter 5 --prev_explainer_chk_pt_folder "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.45_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/iter1" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.45_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.5/iter2" "/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/checkpoints/HAM10k/explainer/lr_0.01_epochs_500_temperature-lens_0.7_input-size-pi_2048_cov_0.45_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1/cov_0.05/iter3" --checkpoint-model "model_g_best_model_epoch_138.pth.tar" "model_g_best_model_epoch_402.pth.tar" "model_g_best_model_epoch_8.pth.tar" "model_g_best_model_epoch_13.pth.tar" --checkpoint-residual "model_g_best_model_epoch_4.pth.tar" "model_g_best_model_epoch_1.pth.tar" "model_g_best_model_epoch_3.pth.tar" --expert-to-train "residual" --dataset "HAM10k" --cov 0.45 0.5 0.05 0.01 --bs 32 --lr 0.01 0.001 0.001 0.001 --temperature-lens 0.7 --lambda-lens 0.0001 --alpha-KD 0.9 --temperature-KD 10 --hidden-nodes 10 --arch "Inception_V3"



