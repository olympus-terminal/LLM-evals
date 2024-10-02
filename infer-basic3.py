import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_output(input_text, model, tokenizer):
    # Tokenize input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs.input_ids.to('cuda')
    attention_mask = inputs.attention_mask.to('cuda')
    
    # Generate output
    with torch.no_grad():
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=5)
    
    # Decode the output tokens back to text
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <base_model_name> <checkpoint_path> <input_file>")
        sys.exit(1)

    base_model_name = sys.argv[1]
    checkpoint_path = sys.argv[2]
    input_file = sys.argv[3]

    # Load the tokenizer from the base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True, padding_side='left')
    
    # Ensure pad_token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the model from the checkpoint
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path).to('cuda')
    # Ensure the model's config is aligned with the tokenizer
    model.config.pad_token_id = tokenizer.pad_token_id

    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    print(f"Model vocabulary size: {model.config.vocab_size}")

    with open(input_file, 'r') as file:
        for line in file:
            input_text = line.strip()
            output_text = generate_output(input_text, model, tokenizer)
            print(f"Input: {input_text}")
            print(f"Output: {output_text}")
            print("-" * 50)
