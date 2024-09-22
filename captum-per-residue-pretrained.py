import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPTNeoConfig
from captum.attr import LayerIntegratedGradients
import numpy as np
import sys
import csv
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]

# Updated model and tokenizer names
model_name = "ChlorophyllChampion/gpt-neo125-ALMGA-FL"
tokenizer_name = "hmbyt5/byt5-small-english"

# Load the ByT5-small English tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Set the padding token to the end-of-sequence token
tokenizer.pad_token = tokenizer.eos_token

# Load and modify the model configuration
config = GPTNeoConfig.from_pretrained(model_name)

# Remove adapter references from the configuration if present
if hasattr(config, 'peft_config'):
    delattr(config, 'peft_config')

# Set the pad_token_id in the model configuration
config.pad_token_id = tokenizer.pad_token_id

# Load the model with the modified configuration
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

# Set the model's pad_token_id
model.config.pad_token_id = tokenizer.pad_token_id

dataset = load_data(sys.argv[1])

def forward_func(input_ids):
    return model(input_ids=input_ids).logits[:, 1]

# Reference the correct embedding layer for GPT-Neo
embedding_layer = model.transformer.wte
lig = LayerIntegratedGradients(forward_func, embedding_layer)

def compute_attribution(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,  # We can now use padding=True since pad_token is set
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

def create_svg_visualization(tokens, attributions, filename):
    plt.figure(figsize=(15, 5))
    sns.barplot(x=tokens, y=attributions)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(filename, format='svg')
    plt.close()

with open('captum_output.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(
        ['Sample', 'Text', 'Positive_Probability'] +
        [f'Token_{i}' for i in range(512)] +
        [f'Attribution_{i}' for i in range(512)]
    )

    for i, text in enumerate(dataset):
        tokens, attributions = compute_attribution(text)

        with torch.no_grad():
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            output = model(**inputs).logits
            probabilities = torch.nn.functional.softmax(output, dim=-1)
            positive_prob = probabilities[0, 1].item()

        # Pad tokens and attributions to 512 length
        padded_tokens = tokens + [''] * (512 - len(tokens))
        padded_attributions = list(attributions) + [0] * (512 - len(attributions))

        csvwriter.writerow([i+1, text, positive_prob] + padded_tokens + padded_attributions)

        create_svg_visualization(tokens, attributions, f'sample_{i+1}_visualization.svg')

print("CSV output saved as 'captum_output.csv'")
print("SVG visualizations saved as 'sample_X_visualization.svg'")
