from transformers import AutoModel

# Load the model
model = AutoModel.from_pretrained("distilroberta-base")

# Count the parameters
total_params = sum(p.numel() for p in model.parameters())

print(f"The distilroberta-base model has {total_params:,} parameters.")
