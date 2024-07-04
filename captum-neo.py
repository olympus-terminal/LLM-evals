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

if len(sys.argv) != 7:
    print("Usage: python script.py <model_path> <tokenizer_path> <dataset_path> <algal_tag> <bacterial_tag> <output_filename>")
    sys.exit(1)

model_name = sys.argv[1]
tokenizer_name = sys.argv[2]
dataset_path = sys.argv[3]
algal_tag = sys.argv[4]
bacterial_tag = sys.argv[5]
output_filename = sys.argv[6]

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

# Define a forward function for token generation
def forward_func(input_ids, attention_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.logits

# Create an instance of LayerIntegratedGradients
lig = LayerIntegratedGradients(forward_func, embeddings)

# Function to compute attributions for a single input

# Function to compute attributions for a single input
def compute_attribution(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    # Generate 9 new tokens
    with torch.no_grad():
        outputs = model.generate(
            input_ids, 
            max_new_tokens=9, 
            do_sample=False, 
            pad_token_id=tokenizer.pad_token_id,
            attention_mask=attention_mask
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Get the target token (last token in the sequence)
    target = outputs[0, -1].unsqueeze(0)
    
    # Ensure target is within vocabulary bounds
    target = torch.clamp(target, 0, model.config.vocab_size - 1)
    
    # Create attention mask for the generated sequence
    gen_attention_mask = torch.ones_like(outputs)
    
    try:
        attributions, delta = lig.attribute(
            inputs=(outputs, gen_attention_mask),
            target=target,
            baselines=(outputs * 0, gen_attention_mask),
            return_convergence_delta=True,
            n_steps=50
        )
    except Exception as e:
        print(f"Attribution failed for text: {text}")
        print(f"Error: {str(e)}")
        return None, None, generated_text
    
    tokens = tokenizer.convert_ids_to_tokens(outputs[0])
    attributions = attributions[0].sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.detach().numpy()
    
    return tokens, attributions, generated_text

# Function to check for tag presence and calculate probabilities
def calculate_tag_probabilities(generated_text, algal_tag, bacterial_tag):
    # Convert to lowercase for case-insensitive matching
    generated_text_lower = generated_text.lower()
    algal_tag_lower = algal_tag.lower()
    bacterial_tag_lower = bacterial_tag.lower()
    
    # Check for partial matches
    algal_present = algal_tag_lower in generated_text_lower
    bacterial_present = bacterial_tag_lower in generated_text_lower
    
    # Calculate probabilities
    if algal_present and not bacterial_present:
        return 1.0, 0.0
    elif bacterial_present and not algal_present:
        return 0.0, 1.0
    elif algal_present and bacterial_present:
        # If both tags are present, calculate probabilities based on their positions
        algal_index = generated_text_lower.index(algal_tag_lower)
        bacterial_index = generated_text_lower.index(bacterial_tag_lower)
        if algal_index < bacterial_index:
            return 0.6, 0.4
        elif bacterial_index < algal_index:
            return 0.4, 0.6
        else:
            return 0.5, 0.5
    else:
        return 0.0, 0.0

# Function to create HTML visualization
def create_html_visualization(tokens, attributions, algal_prob, bacterial_prob, generated_text, algal_tag, bacterial_tag):
    def get_color(value):
        if value > 0:
            return f"rgba(0, 255, 0, {min(abs(value), 1)})"
        else:
            return f"rgba(255, 0, 0, {min(abs(value), 1)})"

    html_content = "<div style='white-space: pre-wrap;'>"
    if tokens and attributions is not None:
        for token, attr in zip(tokens, attributions):
            color = get_color(attr)
            html_content += f'<span style="background-color: {color};">{token}</span>'
    else:
        html_content += "Attribution failed for this sample."
    html_content += "</div>"
    html_content += f"<p>Generated text: {generated_text}</p>"
    html_content += f"<p>Algal tag ({algal_tag}) present: {'Yes' if algal_tag.lower() in generated_text.lower() else 'No'}</p>"
    html_content += f"<p>Bacterial tag ({bacterial_tag}) present: {'Yes' if bacterial_tag.lower() in generated_text.lower() else 'No'}</p>"
    html_content += f"<p>Algal probability: {algal_prob:.4f}</p>"
    html_content += f"<p>Bacterial probability: {bacterial_prob:.4f}</p>"
    return html_content

# Process the dataset and create visualizations
html_output = "<html><body>"
for i, text in enumerate(dataset):
    tokens, attributions, generated_text = compute_attribution(text)
    
    if attributions is not None:
        print(f"Sample {i + 1} raw attributions:", attributions)
    else:
        print(f"Sample {i + 1} attribution failed")
    
    # Calculate probabilities based on tag presence
    algal_prob, bacterial_prob = calculate_tag_probabilities(generated_text, algal_tag, bacterial_tag)
    
    print(f"Sample {i + 1} generated text: {generated_text}")
    print(f"Sample {i + 1} algal probability ({algal_tag}): {algal_prob:.4f}")
    print(f"Sample {i + 1} bacterial probability ({bacterial_tag}): {bacterial_prob:.4f}")
    
    # Add visualization to HTML output
    html_output += f"<h3>Sample {i + 1}</h3>"
    html_output += f"<p>Original Text: {text}</p>"
    html_output += create_html_visualization(tokens, attributions, algal_prob, bacterial_prob, generated_text, algal_tag, bacterial_tag)
    html_output += "<hr>"

html_output += "</body></html>"

# Save the HTML file with the specified filename
with open(output_filename, "w") as f:
    f.write(html_output)

print(f"Visualization saved as '{output_filename}'")
