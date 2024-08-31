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


python llm_inference_performance.py $LINE \
    --input_file examples.txt \
    --output_dir ./results
    --use_wandb
    --wandb_project "LLM-Inference-Performance" \
    --wandb_entity "algaeai" \
    --wandb_run_name "pythia-70m-inference-test" \
    --wandb_tags "pythia" "inference" "performance"
