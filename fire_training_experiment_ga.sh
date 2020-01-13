#!/bin/sh 

#SBATCH --nodes=1
#SBATCH -A ulhum001c
#SBATCH -p GpuQ
#SBATCH --time=24:00:00
# Mail me on job start & end
#SBATCH --mail-user=18195121@studentmail.ul.ie
#SBATCH --mail-type=BEGIN,END
export CUDA_VISIBLE_DEVICES=0,1

module load cuda/9.2
module load conda/2
source activate retinopathy
python run_training_ga.py
