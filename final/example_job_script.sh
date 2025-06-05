#!/bin/bash
#SBATCH --mail-type=ALL           #Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --qos=gpu-rtx-reserved
#SBATCH -p gpu-rtx6000-2
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH --array=1-5
#SBATCH --mem=140G              # memory
#SBATCH --cpus-per-task=24
#SBATCH --time=7-00:00          # time (DD-HH:MM)
#SBATCH --job-name=test_job     #Job name
#SBATCH -o test-%A-%a.out       #Standard output log
#SBATCH -e test-%A-%a.err       #Standard error log
module add python/anaconda/2020.11/3.8
source activate AV4
module add gcc
python -u gru_with_augmentation.py  $SLURM_ARRAY_TASK_ID

