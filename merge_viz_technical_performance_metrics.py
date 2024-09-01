#!/bin/bash

#SBATCH --mem=90GB
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=28

python merge_and_visualize_llm_metrics.py "*_metrics.csv" results-1
(/scratch/drn2/software/conda-mamba_1) [drn2@login2 results]$ cat merge_and_visualize_llm_metrics.py 
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import argparse

def merge_csv_files(file_pattern):
    all_files = glob.glob(file_pattern)
    df_list = []
    for filename in all_files:
        df = pd.read_csv(filename)
        model_name = filename.split('_')[0]  # Assuming the model name is the first part of the filename
        df['model'] = model_name
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

def create_visualizations(df, output_dir):
    # 1. Box plot of inference time by model
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='model', y='inference_time', data=df)
    plt.title('Inference Time by Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/inference_time_boxplot.svg', format='svg')
    plt.close()

    # 2. Scatter plot of tokens per second vs input length
    plt.figure(figsize=(12, 6))
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        plt.scatter(model_df['input_length'], model_df['tokens_per_second'], label=model, alpha=0.7)
    plt.xlabel('Input Length')
    plt.ylabel('Tokens per Second')
    plt.title('Tokens per Second vs Input Length')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/tokens_per_second_scatter.svg', format='svg')
    plt.close()

    # 3. Bar plot of average CPU, memory, and GPU usage by model
    usage_df = df.groupby('model')[['cpu_usage', 'memory_usage', 'gpu_usage']].mean()
    usage_df.plot(kind='bar', figsize=(12, 6))
    plt.title('Average Resource Usage by Model')
    plt.xlabel('Model')
    plt.ylabel('Usage')
    plt.legend(title='Resource Type')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/resource_usage_barplot.svg', format='svg')
    plt.close()

def main(input_pattern, output_dir):
    merged_df = merge_csv_files(input_pattern)
    merged_df.to_csv(f'{output_dir}/merged_llm_metrics.csv', index=False)
    create_visualizations(merged_df, output_dir)
    print(f"Merged CSV and visualizations created successfully in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and visualize LLM metrics")
    parser.add_argument("input_pattern", help="Glob pattern for input CSV files (e.g., '*_metrics.csv')")
    parser.add_argument("output_dir", help="Directory to save output files")
    args = parser.parse_args()

    main(args.input_pattern, args.output_dir)
