import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
from captum.attr import LayerIntegratedGradients
import numpy as np

# Function to load data from a text file
def load_data(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]

# Load the base model and adapter
base_model_name = "sentence-transformers/all-distilroberta-v1"
adapter_path = "ChlorophyllChampion/distilbert-50000"  # your adapter path

# Load the base model
base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name)

# Load the adapter configuration
peft_config = PeftConfig.from_pretrained(adapter_path)

# Load the adapter onto the base model
model = PeftModel.from_pretrained(base_model, adapter_path)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load the dataset
dataset = load_data("sample.txt")

# Define a forward function
def forward_func(input_ids):
    return model(input_ids=input_ids).logits[:, 1]  # Adjust this based on your model's output

# Create an instance of LayerIntegratedGradients
lig = LayerIntegratedGradients(forward_func, model.base_model.roberta.embeddings)

# Function to compute attributions for a single input
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

# Function to create HTML visualization
def create_html_visualization(tokens, attributions):
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
    return html_content

# Process the dataset and create visualizations
html_output = "<html><body>"
for i, text in enumerate(dataset):
    tokens, attributions = compute_attribution(text)
    
    # Print raw attributions
    print(f"Sample {i + 1} raw attributions:", attributions)
    
    # Print model prediction
    with torch.no_grad():
        output = model(**tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)).logits
        probabilities = torch.nn.functional.softmax(output, dim=-1)
        positive_prob = probabilities[0, 1].item()
    print(f"Sample {i + 1} positive probability: {positive_prob:.4f}")
    
    # Add visualization to HTML output
    html_output += f"<h3>Sample {i + 1}</h3>"
    html_output += f"<p>Text: {text}</p>"
    html_output += f"<p>Positive probability: {positive_prob:.4f}</p>"
    html_output += create_html_visualization(tokens, attributions)
    html_output += "<hr>"

html_output += "</body></html>"

# Save the HTML file
with open("captum_dataset_visualization.html", "w") as f:
    f.write(html_output)

print("Visualization saved as 'captum_dataset_visualization.html'")
