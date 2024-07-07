#!/bin/bash

#SBATCH -o slurm-logs/arrayJob_%A_%a.out
#SBATCH -e slurm-logs/arrayJob_%A_%a.err
#SBATCH -a 1-3
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20


LINE=$(sed -n "$SLURM_ARRAY_TASK_ID"p filelist.txt)
echo "$LINE"

cleaned=$(echo "$LINE" | tr -dc '[:alnum:]')

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && for f in generated_prompts_algae*; do python infer-SL1.py "$LINE" $f >> eval-results_Algal_"$cleaned"-"$f"; done

wait

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && for f in generated_prompts_bact*; do python infer-SL1.py "$LINE" $f >> eval-results_Bact_"$cleaned"-"$f"; done

wait

for f in generated_prompts_algae*; do ./metrics-c3o.sh eval-results_Algal_"$cleaned"-"$f" eval-results_Bact_"$cleaned"-"$f" >> eval-results_FinalMetrics_"$cleaned"-"$f"; done
