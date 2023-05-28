#!/bin/sh
#SBATCH --output=Path/Stanford_CXR/moie/effusion/flops/effusion_15000_%j.out
pwd
hostname
date

CURRENT=$(date +"%Y-%m-%d_%T")
echo $CURRENT

slurm_output_moie=Path/Stanford_CXR/moie/effusion/flops/effusion_15000_moie_$CURRENT.out
slurm_output_BB=Path/Stanford_CXR/moie/effusion/flops/effusion_15000_BB_$CURRENT.out

echo "Stanford_CXR"
source Path/anaconda3/etc/profile.d/conda.sh
# conda activate python_3_7
conda activate python_3_7_rtx_6000


python ../../../codebase/cal_flops_cxr_domain_transfer_main.py \
  --disease "effusion" \
  --tot_samples 15000 \
  --batch-size 64 \
  --epochs 5 --profile "y" >$slurm_output_moie

python ../../../codebase/train_BB_stanford_cxr.py --arch='densenet121' --selected-obs="effusion" --epochs 1 --labels "0 (No effusion)" "1 (effusion)"  --domain_transfer "y" --tot_samples 15000 --checkpoint-mimic-cxr "g_best_model_epoch_3.pth.tar" --profile "y" >$slurm_output_BB

