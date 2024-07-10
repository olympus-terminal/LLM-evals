import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
import numpy as np
import matplotlib.pyplot as plt

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
        logits = outputs.logits[:, -1, :]  # Get logits for the last token
        
        # Prepare targets
        targets = torch.tensor([self.tag_tokens[tag][0] for tag in target_tags]).to(logits.device)
        
        # Compute loss
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()

        # Compute SHAP values
        shap_values = (input_embeds.grad * input_embeds).sum(dim=-1).detach().cpu().numpy()
        
        return shap_values

def shap_explain(model, tokenizer, test_dataset, num_samples=10):
    device = next(model.parameters()).device
    
    # Create a custom explainer
    explainer = CustomShapExplainer(model, tokenizer)
    
    # Select samples to explain
    samples_to_explain = test_dataset.select(range(min(num_samples, len(test_dataset))))
    
    # Separate input sequences and tags
    input_sequences = []
    target_tags = []
    for seq in samples_to_explain['input_ids']:
        # Find the start of the tag
        tag_start = next(i for i, token in enumerate(seq) if token == tokenizer.encode("[")[0])
        input_sequences.append(seq[:tag_start])
        tag = tokenizer.decode(seq[tag_start:])
        target_tags.append('bacteria' if '[<!!!>]' in tag else 'algae')
    
    # Find the maximum sequence length
    max_length = max(len(seq) for seq in input_sequences)
    
    samples_inputs = {
        'input_ids': torch.tensor(adjust_length(input_sequences, max_length)).to(device),
        'attention_mask': torch.tensor([[1] * len(seq) + [0] * (max_length - len(seq)) for seq in input_sequences]).to(device)
    }
    
    # Compute SHAP values
    shap_values = explainer.explain(samples_inputs, target_tags)
    
    return shap_values, samples_to_explain, input_sequences

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_svg import FigureCanvasSVG

def plot_shap_values(shap_values_list, tokens_list, tags, filename_prefix, organism_type):
    num_plots = len(shap_values_list)
    fig, axs = plt.subplots(num_plots, 1, figsize=(30, 4*num_plots), squeeze=False)
    fig.suptitle(f"SHAP Values for {organism_type.capitalize()} Protein Sequences", fontsize=24)

    for i in range(num_plots):
        shap_values = shap_values_list[i]
        tokens = tokens_list[i]
        tag = tags[i]

        # Plot heatmap
        im = axs[i, 0].imshow(shap_values.reshape(1, -1), aspect='auto', cmap='RdBu_r')
        axs[i, 0].set_yticks([])

        # Add colorbar
        if i == 0:  # Add colorbar only for the first plot
            cbar = fig.colorbar(im, ax=axs[:, 0].ravel().tolist(), shrink=0.8, pad=0.01)
            cbar.ax.tick_params(labelsize=14)

        # Set x-axis ticks and labels
        num_ticks = 30  # Increase number of visible tokens
        step = max(len(tokens) // num_ticks, 1)
        tick_positions = range(0, len(tokens), step)
        tick_labels = [tokens[pos] for pos in tick_positions]
        axs[i, 0].set_xticks(tick_positions)
        axs[i, 0].set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=14)

        # Add tag information
        axs[i, 0].set_title(f"Sample {i+1} - Tag: {tag}", fontsize=16)

    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_{organism_type}.svg", format='svg', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    model_path = "./pretrained_byt5_causal_lm"  # Adjust this to your actual model path
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load the tokenized dataset
    tokenized_dataset = load_from_disk("/scratch/drn2/PROJECTS/AI/Byte5/byte5-tokenized_dataset")
    
    # Assuming you have a test split, if not, create one
    if 'test' not in tokenized_dataset:
        tokenized_dataset = tokenized_dataset['train'].train_test_split(test_size=0.1, seed=42)
    
    shap_values, samples, input_sequences = shap_explain(model, tokenizer, tokenized_dataset['test'])
    
    shap_values_bacteria = []
    tokens_bacteria = []
    tags_bacteria = []
    shap_values_algae = []
    tokens_algae = []
    tags_algae = []

    bacteria_count = 0
    algae_count = 0

    for i in range(len(samples)):
        tokens = tokenizer.convert_ids_to_tokens(input_sequences[i])
        text = tokenizer.decode(input_sequences[i])
        full_text = tokenizer.decode(samples['input_ids'][i])
        tag = full_text[len(text):]
        print(f"Sample {i + 1}: {text}")
        print(f"Tag: {tag}")
        
        if '[<!!!>]' in tag and bacteria_count < 20:
            shap_values_bacteria.append(shap_values[i][:len(input_sequences[i])])
            tokens_bacteria.append(tokens)
            tags_bacteria.append(tag)
            bacteria_count += 1
        elif '[<@@@>]' in tag and algae_count < 20:
            shap_values_algae.append(shap_values[i][:len(input_sequences[i])])
            tokens_algae.append(tokens)
            tags_algae.append(tag)
            algae_count += 1
        
        if bacteria_count == 20 and algae_count == 20:
            break

    plot_shap_values(shap_values_bacteria, tokens_bacteria, tags_bacteria, "shap_plots", "bacteria")
    plot_shap_values(shap_values_algae, tokens_algae, tags_algae, "shap_plots", "algae")

if __name__ == "__main__":
    main()
