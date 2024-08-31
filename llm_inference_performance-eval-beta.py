#!/usr/bin/env python3

import argparse
import time
from pathlib import Path
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
import matplotlib.pyplot as plt

class PerformanceLogger:
    def __init__(self, use_wandb):
        self.use_wandb = use_wandb
        self.inference_times = []
        self.input_lengths = []
        self.output_lengths = []
        self.cpu_usages = []
        self.memory_usages = []
        if torch.cuda.is_available():
            self.gpu_usages = []

    def log(self, input_text, output, start_time, end_time):
        inference_time = end_time - start_time
        self.inference_times.append(inference_time)
        self.input_lengths.append(len(input_text))
        self.output_lengths.append(len(output))
        
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        self.cpu_usages.append(cpu_percent)
        self.memory_usages.append(memory_percent)
        
        if torch.cuda.is_available():
            gpu_percent = torch.cuda.utilization()
            self.gpu_usages.append(gpu_percent)

        if self.use_wandb:
            wandb.log({
                "inference_time": inference_time,
                "input_length": len(input_text),
                "output_length": len(output),
                "tokens_per_second": len(output) / inference_time,
                "cpu_usage": cpu_percent,
                "memory_usage": memory_percent
            })
            if torch.cuda.is_available():
                wandb.log({"gpu_usage": gpu_percent})

    def plot_statistics(self):
        fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        
        axs[0, 0].hist(self.inference_times)
        axs[0, 0].set_title("Inference Times")
        axs[0, 0].set_xlabel("Time (s)")
        axs[0, 0].set_ylabel("Frequency")
        
        axs[0, 1].scatter(self.input_lengths, self.inference_times)
        axs[0, 1].set_title("Input Length vs Inference Time")
        axs[0, 1].set_xlabel("Input Length")
        axs[0, 1].set_ylabel("Inference Time (s)")
        
        axs[1, 0].plot(self.cpu_usages, label="CPU")
        axs[1, 0].plot(self.memory_usages, label="Memory")
        if torch.cuda.is_available():
            axs[1, 0].plot(self.gpu_usages, label="GPU")
        axs[1, 0].set_title("Resource Usage Over Time")
        axs[1, 0].set_xlabel("Inference Run")
        axs[1, 0].set_ylabel("Usage (%)")
        axs[1, 0].legend()
        
        axs[1, 1].scatter(self.output_lengths, self.inference_times)
        axs[1, 1].set_title("Output Length vs Inference Time")
        axs[1, 1].set_xlabel("Output Length")
        axs[1, 1].set_ylabel("Inference Time (s)")
        
        plt.tight_layout()
        plt.savefig("inference_performance.png")
        print("Performance plot saved as inference_performance.png")

def load_model(model_path):
    if Path(model_path).exists():
        print(f"Loading local model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        print(f"Loading model from Hugging Face: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer

def inference(model, tokenizer, input_text, max_length=100):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            pad_token_id=tokenizer.pad_token_id
        )
    end_time = time.time()
    
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text, start_time, end_time

def main():
    parser = argparse.ArgumentParser(description="LLM Inference Performance Script")
    parser.add_argument("model_path", type=str, help="Path to local model or Hugging Face model name")
    parser.add_argument("--input_file", type=str, required=True, help="Path to file containing input texts")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="llm_inference", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, help="W&B entity (username or team name)")
    parser.add_argument("--wandb_run_name", type=str, help="W&B run name")
    parser.add_argument("--wandb_tags", type=str, nargs="+", help="W&B tags for the run")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of generated text")
    args = parser.parse_args()

    if args.use_wandb:
        wandb_config = {
            "model": args.model_path,
            "max_length": args.max_length,
        }
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            tags=args.wandb_tags,
            config=wandb_config
        )

    model, tokenizer = load_model(args.model_path)
    logger = PerformanceLogger(args.use_wandb)

    with open(args.input_file, 'r') as f:
        input_texts = f.readlines()

    for input_text in input_texts:
        output, start_time, end_time = inference(model, tokenizer, input_text.strip(), args.max_length)
        logger.log(input_text, output, start_time, end_time)
        print(f"Input: {input_text.strip()}")
        print(f"Output: {output}")
        print(f"Inference Time: {end_time - start_time:.2f} seconds\n")

    logger.plot_statistics()

    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
