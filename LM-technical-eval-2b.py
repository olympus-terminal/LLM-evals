import sys
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import os
from typing import List, Dict, Tuple
import time
import numpy as np
from sacrebleu.metrics import BLEU
import psutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PromptDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> List[str]:
        # Assuming the prompts are in a file named 'prompts.txt' in the 'train' subdirectory
        prompt_file = os.path.join(self.data_dir, 'train', 'prompts.txt')
        if not os.path.exists(prompt_file):
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        
        with open(prompt_file, 'r') as f:
            prompts = f.readlines()
        
        # Remove any leading/trailing whitespace and filter out empty lines
        prompts = [prompt.strip() for prompt in prompts if prompt.strip()]
        
        logging.info(f"Loaded {len(prompts)} prompts from {prompt_file}")
        return prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        return {
            'input_text': prompt,
            'reference': 'UNKNOWN'  # Since we don't have reference outputs
        }

def calculate_perplexity(model: AutoModelForCausalLM, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> float:
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        return torch.exp(outputs.loss).item()

def generate_output_batch(model: AutoModelForCausalLM, batch: Dict[str, List[str]], tokenizer: AutoTokenizer) -> Tuple[List[str], float, float, List[str]]:
    logging.info(f"Batch keys: {batch.keys()}")
    
    inputs = batch['input_text']
    references = batch['reference']
    
    encoded_inputs = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True).to('cuda')
    input_ids = encoded_inputs['input_ids']
    attention_mask = encoded_inputs['attention_mask']

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=15,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    end_time = time.time()

    generation_time = end_time - start_time
    tokens_generated = outputs.shape[1] - input_ids.shape[1]
    tokens_per_second = tokens_generated / generation_time

    perplexity = calculate_perplexity(model, input_ids, attention_mask)

    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return decoded_outputs, perplexity, tokens_per_second, references

def calculate_bleu(references: List[str], hypotheses: List[str]) -> float:
    bleu = BLEU()
    return bleu.corpus_score(hypotheses, [references]).score

if __name__ == "__main__":
    if len(sys.argv) != 4:
        logging.error("Usage: python script.py <model_name_or_path> <tokenized_data_dir> <output_dir>")
        sys.exit(1)

    model_name_or_path = sys.argv[1]
    tokenized_data_dir = sys.argv[2]
    output_dir = sys.argv[3]

    logging.info(f"Model path: {model_name_or_path}")
    logging.info(f"Tokenized data directory: {tokenized_data_dir}")
    logging.info(f"Output directory: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained("hmbyt5/byt5-small-english", use_fast=True, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to('cuda')

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    dataset = PromptDataset(tokenized_data_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_perplexities = []
    all_tokens_per_second = []
    all_inputs = []
    all_outputs = []
    all_references = []

    for i, batch in enumerate(dataloader):
        logging.info(f"Processing batch {i+1}")
        outputs, perplexity, tokens_per_second, references = generate_output_batch(model, batch, tokenizer)
        
        all_perplexities.extend([perplexity] * len(outputs))
        all_tokens_per_second.extend([tokens_per_second] * len(outputs))
        all_inputs.extend(batch['input_text'])
        all_outputs.extend(outputs)
        all_references.extend(references)

        for input_text, output_text, reference in zip(batch['input_text'], outputs, references):
            logging.info(f"Input: {input_text}")
            logging.info(f"Output: {output_text}")
            logging.info(f"Reference: {reference}")
            logging.info("-" * 50)

    # Calculate overall metrics
    avg_perplexity = np.mean(all_perplexities)
    avg_tokens_per_second = np.mean(all_tokens_per_second)
    bleu_score = calculate_bleu(all_references, all_outputs)

    # Memory usage
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 ** 2  # in MB

    # Save results
    results = {
        "avg_perplexity": avg_perplexity,
        "avg_tokens_per_second": avg_tokens_per_second,
        "bleu_score": bleu_score,
        "memory_usage_mb": memory_usage,
        "inputs": all_inputs,
        "outputs": all_outputs,
        "references": all_references
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    logging.info(f"Average Perplexity: {avg_perplexity}")
    logging.info(f"Average Tokens per Second: {avg_tokens_per_second}")
    logging.info(f"BLEU Score: {bleu_score}")
    logging.info(f"Memory Usage: {memory_usage} MB")
    logging.info("Inference completed successfully. Results saved in evaluation_results.json")
