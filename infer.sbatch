#!/bin/bash

#SBATCH -o slurm-logs/arrayJob_%A_%a.out
#SBATCH -e slurm-logs/arrayJob_%A_%a.err
#SBATCH -a 1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20


LINE=$(sed -n "$SLURM_ARRAY_TASK_ID"p filelist.txt)
echo "$LINE"

cleaned=$(echo "$LINE" | tr -dc '[:alnum:]')

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python infer-SL1.py "$LINE" AlgalTop10000-10holdout_prompts.txt >> eval-results_Algal_"$cleaned"

wait

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python infer-SL1.py "$LINE" BactTop10000-10holdout_prompts.txt >> eval-results_Bact_"$cleaned" 

wait

./metrics-c3o.sh eval-results_Algal_"$cleaned" eval-results_Bact_"$cleaned" >> eval-results_FinalMetrics_"$cleaned"
