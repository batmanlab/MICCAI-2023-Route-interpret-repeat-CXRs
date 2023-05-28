#!/bin/sh
#SBATCH --output=/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/psclogs/Stanford_CXR/moie/cardiomegaly/flops/ecardiomegaly_15000_%j.out
pwd
hostname
date

CURRENT=$(date +"%Y-%m-%d_%T")
echo $CURRENT

slurm_output_moie=/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/psclogs/Stanford_CXR/moie/cardiomegaly/flops/cardiomegaly_15000_moie_$CURRENT.out
slurm_output_BB=/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/psclogs/Stanford_CXR/moie/cardiomegaly/flops/cardiomegaly_15000_BB_$CURRENT.out

echo "Stanford_CXR"
source /ocean/projects/asc170022p/shg121/anaconda3/etc/profile.d/conda.sh
# conda activate python_3_7
conda activate python_3_7_rtx_6000


python ../../../codebase/cal_flops_cxr_domain_transfer_main.py \
  --disease "cardiomegaly" \
  --tot_samples 15000 \
  --batch-size 64 \
  --epochs 5 --profile "y" >$slurm_output_moie

python ../../../codebase/train_BB_stanford_cxr.py --arch='densenet121' --selected-obs="cardiomegaly" --epochs 1 --labels "0 (No cardiomegaly)" "1 (cardiomegaly)"  --domain_transfer "y" --tot_samples 15000 --checkpoint-mimic-cxr "g_best_model_epoch_1.pth.tar" --profile "y" >$slurm_output_BB

