import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer, GPTNeoForSequenceClassification
from peft import PeftModel, PeftConfig
from captum.attr import LayerIntegratedGradients
import numpy as np
import sys
import csv
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]

base_model_name = "EleutherAI/gpt-neo-125M"
adapter_path = "ChlorophyllChampion/gpt-neo125-ALMGA-FL"

base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name)
peft_config = PeftConfig.from_pretrained(adapter_path)
model = PeftModel.from_pretrained(base_model, adapter_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

dataset = load_data(sys.argv[1])

def forward_func(input_ids):
    return model(input_ids=input_ids).logits[:, 1]

lig = LayerIntegratedGradients(forward_func, model.base_model.roberta.embeddings)

def compute_attribution(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    attributions, delta = lig.attribute(inputs=inputs['input_ids'],
                                        baselines=inputs['input_ids'] * 0,
                                        return_convergence_delta=True,
                                        n_steps=50)
    
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
    csvwriter.writerow(['Sample', 'Text', 'Positive_Probability'] + [f'Token_{i}' for i in range(512)] + [f'Attribution_{i}' for i in range(512)])

    for i, text in enumerate(dataset):
        tokens, attributions = compute_attribution(text)
        
        with torch.no_grad():
            output = model(**tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)).logits
            probabilities = torch.nn.functional.softmax(output, dim=-1)
            positive_prob = probabilities[0, 1].item()
        
        # Pad tokens and attributions to 512 length
        padded_tokens = tokens + [''] * (512 - len(tokens))
        padded_attributions = list(attributions) + [0] * (512 - len(attributions))
        
        csvwriter.writerow([i+1, text, positive_prob] + padded_tokens + padded_attributions)
        
        create_svg_visualization(tokens, attributions, f'sample_{i+1}_visualization.svg')

print("CSV output saved as 'captum_output.csv'")
print("SVG visualizations saved as 'sample_X_visualization.svg'")
