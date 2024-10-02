import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import gc
import random

def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained("hmbyt5/byt5-small-english", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return model, tokenizer

def adjust_length(sequences, target_length):
    return [seq[:target_length] + [0] * (target_length - len(seq)) for seq in sequences]

class CustomShapExplainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.tag_tokens = {
            "bacteria": self.tokenizer.encode("[<!!!>]"),
            "algae": self.tokenizer.encode("[<@@@>]")
        }

    def explain(self, input_data, target_tags):
        self.model.zero_grad()
        input_ids = input_data['input_ids']
        attention_mask = input_data['attention_mask']
        
        input_embeds = self.model.transformer.wte(input_ids)
        input_embeds.retain_grad()

        outputs = self.model(inputs_embeds=input_embeds, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]
        
        targets = torch.tensor([self.tag_tokens[tag][0] for tag in target_tags]).to(logits.device)
        
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()

        shap_values = (input_embeds.grad * input_embeds).sum(dim=-1).detach().cpu().numpy()
        
        return shap_values

def shap_explain_batch(model, tokenizer, batch, explainer):
    device = next(model.parameters()).device
    
    input_sequences = []
    target_tags = []
    for seq in batch['input_ids']:
        tag_start = next(i for i, token in enumerate(seq) if token == tokenizer.encode("[")[0])
        input_sequences.append(seq[:tag_start])
        tag = tokenizer.decode(seq[tag_start:])
        target_tags.append('bacteria' if '[<!!!>]' in tag else 'algae')
    
    max_length = max(len(seq) for seq in input_sequences)
    
    samples_inputs = {
        'input_ids': torch.tensor(adjust_length(input_sequences, max_length)).to(device),
        'attention_mask': torch.tensor([[1] * len(seq) + [0] * (max_length - len(seq)) for seq in input_sequences]).to(device)
    }
    
    shap_values = explainer.explain(samples_inputs, target_tags)
    
    return shap_values, input_sequences, target_tags

def analyze_patterns(shap_values, tokens, tags):
    pattern_stats = defaultdict(lambda: defaultdict(lambda: {'sum': 0.0, 'count': 0, 'values': []}))

    for shap, token_seq, tag in zip(shap_values, tokens, tags):
        for value, token in zip(shap, token_seq):
            if token.startswith('▁'):  # New word/amino acid
                amino_acid = token[1:]
                pattern_stats[tag][amino_acid]['sum'] += value
                pattern_stats[tag][amino_acid]['count'] += 1
                pattern_stats[tag][amino_acid]['values'].append(value)

    # Calculate average importance and standard deviation
    for tag in pattern_stats:
        for amino_acid in pattern_stats[tag]:
            stats = pattern_stats[tag][amino_acid]
            stats['mean'] = stats['sum'] / stats['count']
            stats['std'] = np.std(stats['values'])

    return pattern_stats

def save_pattern_stats(pattern_stats, filename):
    df_data = []
    for tag in pattern_stats:
        for amino_acid, stats in pattern_stats[tag].items():
            df_data.append({
                'Tag': tag,
                'Amino_Acid': amino_acid,
                'Mean_Importance': stats['mean'],
                'Std_Importance': stats['std'],
                'Count': stats['count']
            })
    
    df = pd.DataFrame(df_data)
    df.to_csv(filename, index=False)
    print(f"Pattern statistics saved to {filename}")

def main():
    model_path = "./pretrained_byt5_causal_lm"
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    tokenized_dataset = load_from_disk("/scratch/drn2/PROJECTS/AI/Byte5/byte5-tokenized_dataset")
    
    if 'test' not in tokenized_dataset:
        tokenized_dataset = tokenized_dataset['train'].train_test_split(test_size=0.1, seed=42)
    
    test_dataset = tokenized_dataset['test']
    
    # Sample 1% of the test dataset
    sample_size = int(len(test_dataset) * 0.01)
    sampled_indices = random.sample(range(len(test_dataset)), sample_size)
    sampled_dataset = test_dataset.select(sampled_indices)
    
    batch_size = 32  # Adjust this based on your GPU memory
    explainer = CustomShapExplainer(model, tokenizer)

    all_shap_values = []
    all_input_sequences = []
    all_tags = []

    total_batches = (len(sampled_dataset) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(sampled_dataset), batch_size), total=total_batches, desc="Processing batches"):
        batch = sampled_dataset[i:i+batch_size]
        shap_values, input_sequences, tags = shap_explain_batch(model, tokenizer, batch, explainer)
        
        all_shap_values.extend(shap_values)
        all_input_sequences.extend(input_sequences)
        all_tags.extend(tags)

        # Free up memory
        del shap_values, input_sequences, tags
        torch.cuda.empty_cache()
        gc.collect()

        # Print progress
        if (i // batch_size) % 10 == 0:
            print(f"Processed {i+batch_size} / {len(sampled_dataset)} samples")

    tokens = [tokenizer.convert_ids_to_tokens(seq) for seq in all_input_sequences]
    
    print("Analyzing patterns...")
    pattern_stats = analyze_patterns(all_shap_values, tokens, all_tags)
    
    print("Saving results...")
    save_pattern_stats(pattern_stats, "amino_acid_patterns_1percent_sample.csv")
    
    print("Analysis complete. Results saved to amino_acid_patterns_1percent_sample.csv")

if __name__ == "__main__":
    main()
