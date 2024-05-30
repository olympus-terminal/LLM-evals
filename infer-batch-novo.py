import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

# Define a simple dataset to handle text data in batches
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

def generate_output(model, batch, tokenizer, max_new_tokens=3):
    input_ids = batch['input_ids'].to(model.device) 
    attention_mask = batch['attention_mask'].to(model.device)

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens
        )

    # Decode the output tokens back to text
    output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return output_texts

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <model_name_or_path> <input_file>")
        sys.exit(1)

    model_name_or_path = sys.argv[1]
    input_file = sys.argv[2]

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    # Use GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load all texts into a list
    with open(input_file, 'r') as file:
        texts = [line.strip() for line in file]

    # Create a dataset and a data loader
    dataset = TextDataset(texts, tokenizer, max_length=128) 
    dataloader = DataLoader(dataset, batch_size=8) 

    # Generate outputs in batches
    for batch in dataloader:
        output_texts = generate_output(model, batch, tokenizer)
        for text in output_texts:
            print(text)
