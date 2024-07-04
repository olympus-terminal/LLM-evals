import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from captum.attr import LayerIntegratedGradients
import numpy as np
import sys

def load_data(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def get_model_specific_components(model):
    if hasattr(model, 'transformer'):
        return model.transformer.wte, 'gpt2'
    elif hasattr(model, 'gpt_neox'):
        return model.gpt_neox.embed_in, 'gpt_neox'
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

# Load command-line arguments
model_name = sys.argv[1]
tokenizer_name = sys.argv[2]
dataset_path = sys.argv[3]
algal_tag = sys.argv[4]
bacterial_tag = sys.argv[5]

# Load the pretrained model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Set padding token for GPT2-like models
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# Load the dataset
dataset = load_data(dataset_path)

# Get model-specific components
embeddings, model_type = get_model_specific_components(model)

# Define a forward function for next-token prediction
def forward_func(input_ids):
    outputs = model(input_ids=input_ids)
    next_token_logits = outputs.logits[:, -1, :]
    return next_token_logits

# Create an instance of LayerIntegratedGradients
lig = LayerIntegratedGradients(forward_func, embeddings)

# Function to compute attributions for a single input
def compute_attribution(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs['input_ids']
    
    # Get the target token (last token in the sequence)
    target = input_ids[0, -1].unsqueeze(0)
    
    attributions, delta = lig.attribute(inputs=input_ids,
                                        target=target,
                                        baselines=input_ids * 0,
                                        return_convergence_delta=True,
                                        n_steps=50)
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.detach().numpy()
    
    return tokens, attributions

# Function to create HTML visualization
def create_html_visualization(tokens, attributions, algal_prob, bacterial_prob):
    def get_color(value):
        if value > 0:
            return f"rgba(0, 255, 0, {min(abs(value), 1)})"
        else:
            return f"rgba(255, 0, 0, {min(abs(value), 1)})"

    html_content = "<div style='white-space: pre-wrap;'>"
    for token, attr in zip(tokens, attributions):
        color = get_color(attr)
        html_content += f'<span style="background-color: {color};">{token}</span>'
    html_content += "</div>"
    html_content += f"<p>Algal probability ({algal_tag}): {algal_prob:.4f}</p>"
    html_content += f"<p>Bacterial probability ({bacterial_tag}): {bacterial_prob:.4f}</p>"
    return html_content

# Process the dataset and create visualizations
html_output = "<html><body>"
for i, text in enumerate(dataset):
    tokens, attributions = compute_attribution(text)
    
    # Print raw attributions
    print(f"Sample {i + 1} raw attributions:", attributions)
    
    # Compute probabilities for algal and bacterial tags
    with torch.no_grad():
        input_ids = tokenizer.encode(text, return_tensors="pt")
        outputs = model(input_ids=input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        
        algal_id = tokenizer.encode(algal_tag, add_special_tokens=False)[0]
        bacterial_id = tokenizer.encode(bacterial_tag, add_special_tokens=False)[0]
        
        algal_prob = next_token_probs[0, algal_id].item()
        bacterial_prob = next_token_probs[0, bacterial_id].item()
    
    print(f"Sample {i + 1} algal probability ({algal_tag}): {algal_prob:.4f}")
    print(f"Sample {i + 1} bacterial probability ({bacterial_tag}): {bacterial_prob:.4f}")
    
    # Add visualization to HTML output
    html_output += f"<h3>Sample {i + 1}</h3>"
    html_output += f"<p>Text: {text}</p>"
    html_output += create_html_visualization(tokens, attributions, algal_prob, bacterial_prob)
    html_output += "<hr>"

html_output += "</body></html>"

# Save the HTML file
with open("captum_dataset_visualization.html", "w") as f:
    f.write(html_output)

print("Visualization saved as 'captum_dataset_visualization.html'")
