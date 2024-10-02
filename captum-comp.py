import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPTNeoConfig
from captum.attr import LayerIntegratedGradients
import numpy as np
import sys
import csv
import json

def load_data(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]

# Updated model and tokenizer names
model_name = "ChlorophyllChampion/gpt-neo125-ALMGA-FL"
tokenizer_name = "gpt2"

# Load the ByT5-small English tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer.pad_token = tokenizer.eos_token

# Load and modify the model configuration
config = GPTNeoConfig.from_pretrained(model_name)
if hasattr(config, 'peft_config'):
    delattr(config, 'peft_config')
config.pad_token_id = tokenizer.pad_token_id

# Load the model with the modified configuration
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
model.config.pad_token_id = tokenizer.pad_token_id

dataset = load_data(sys.argv[1])

def forward_func(input_ids):
    return model(input_ids=input_ids).logits[:, 1]

embedding_layer = model.transformer.wte
lig = LayerIntegratedGradients(forward_func, embedding_layer)

def compute_attribution(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    attributions, delta = lig.attribute(
        inputs=inputs['input_ids'],
        baselines=torch.zeros_like(inputs['input_ids']),
        return_convergence_delta=True,
        n_steps=50
    )
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.cpu().detach().numpy()
    return tokens, attributions

def process_batch(batch, batch_size=32):
    all_tokens = []
    all_attributions = []
    all_probs = []

    for i in range(0, len(batch), batch_size):
        sub_batch = batch[i:i+batch_size]
        inputs = tokenizer(sub_batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs).logits
            probs = torch.nn.functional.softmax(outputs, dim=-1)[:, 1].tolist()
        
        all_probs.extend(probs)
        
        for text in sub_batch:
            tokens, attributions = compute_attribution(text)
            all_tokens.append(tokens)
            all_attributions.append(attributions)

    return all_tokens, all_attributions, all_probs

def save_results(dataset, tokens, attributions, probs, output_file):
    results = []
    for i, (text, tok, attr, prob) in enumerate(zip(dataset, tokens, attributions, probs)):
        result = {
            "sample_id": i+1,
            "text": text,
            "positive_probability": prob,
            "tokens": tok,
            "attributions": attr.tolist()
        }
        results.append(result)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

# Main execution
tokens, attributions, probs = process_batch(dataset)
save_results(dataset, tokens, attributions, probs, 'captum_output.json')

print("JSON output saved as 'captum_output.json'")
