#!/bin/bash
#SBATCH --job-name=evo_finetune  # Name of the job
#SBATCH --output=job_logs/output_%A_%a.txt  # Standard output file
#SBATCH --error=job_logs/error_%A_%a.txt  # Standard error file
#SBATCH --time=01:00:00  # Time limit in D-HH:MM:SS, set to 1 hour per job
#SBATCH --partition=gpu-64g  # Partition name for the gpu-64g
#SBATCH --mem=32G # Request 63GB of memory
#SBATCH --mail-type=END  # Notifications for job done
#SBATCH --array=1  # Array job, 864 jobs in total, limit to 25 concurrent jobs

#run multiple jobs, allocating diff memories

source ~/miniconda3/etc/profile.d/conda.sh

conda activate myenv

# Set the WANDB_API_KEY environment variable
export WANDB_API_KEY=ff9c98ae599240e81f8e14040bddbfd653c28887

# Load necessary modules (modify based on your environment)
module load cuda/11.8
module load python/3.8  # Example, adjust to your environment

cd /home/ubuntu/evo-finutune

# Run your Python script
srun python3 main.py --array_task_id=$SLURM_ARRAY_TASK_ID
