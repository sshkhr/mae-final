#!/bin/bash

# Parameters
#SBATCH --array=0-5%6
#SBATCH --cpus-per-task=10
#SBATCH --error=/private/home/sshkhr/mae/demo/CKA-logs-msn/%A_%a_0_log.err
#SBATCH --gpus-per-node=1
#SBATCH --job-name=submitit
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/private/home/sshkhr/mae/demo/CKA-logs-msn/%A_%a_0_log.out
#SBATCH --partition=devlab
#SBATCH --signal=USR1@120
#SBATCH --time=600
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /private/home/sshkhr/mae/demo/CKA-logs-msn/%A_%a_%t_log.out --error /private/home/sshkhr/mae/demo/CKA-logs-msn/%A_%a_%t_log.err /private/home/sshkhr/.conda/envs/pytorch_env/bin/python -u -m submitit.core._submit /private/home/sshkhr/mae/demo/CKA-logs-msn
