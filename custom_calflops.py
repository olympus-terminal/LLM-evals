from calflops import calculate_flops
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model and input parameters
batch_size, max_seq_length = 1, 128
model_name = "ChlorophyllChampion/Pythia70m-55000-D100s-newPretrained"
tokenizer_name = "hmbyt5/byt5-small-english"

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Calculate FLOPs
flops, macs, params = calculate_flops(
    model=model,
    input_shape=(batch_size, max_seq_length),
    transformer_tokenizer=tokenizer,
    print_results=True,
    print_detailed=True
)

print(f"{model_name} FLOPs: {flops}  MACs: {macs}  Params: {params}")
