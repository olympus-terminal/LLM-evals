import torch
import gc
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoConfig, GPTNeoXForCausalLM
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# Define the HuggingFace model name
model_name = "ChlorophyllChampion/duality100s-ckpt-30000_pythia70m-arc"

# Load the configuration
config = AutoConfig.from_pretrained(model_name)

# Load the ByteT5 tokenizer
tokenizer = AutoTokenizer.from_pretrained("hmbyt5/byt5-small-english")

# Explicitly download the safetensors file
safetensors_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")

# Manually create the model
model = GPTNeoXForCausalLM(config)

# Load the state dict using safetensors
state_dict = load_file(safetensors_path)
model.load_state_dict(state_dict, strict=False)

# Move model to GPU if available and use half precision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).half()

def cleanup_model(model):
    try:
        if hasattr(model, 'base_model_prefix') and len(model.base_model_prefix) > 0:
            bm = getattr(model, model.base_model_prefix)
            del bm
    except:
        pass
    del model
    gc.collect()
    torch.cuda.empty_cache()

# Example amino acid sequences
aa_sequence1 = "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQTLSEQVQEELLSSQVTQELRALMDETMKELKAYKSELEEQLTPVAEETRARLSKELQAAQARLGADVLASHGRLVQYRGEVQAMLGQSTEELRVRLASHLRKLRKRLLRDADDLQKRLAVYQAGAREGAERGLSAIRERLGPLVEQGRVRAATVGSLAGQPLQERAQAWGERLRARMEEMGSRTRDRLDEVKEQVAEVRAKLEEQAQQRLKQEVNQILAEAEKRREPTPRO"
aa_sequence2 = "MKAILVVLLYTFATANADTLCIGYHANNSTDTVDTVLEKNVTVTHSVNLLEDKHNGKLCKLRGVAPLHLGKCNIAGWILGNPECESLSTASSWSYIVETPSSDNGTCYPGDFIDYEELREQLSSVSSFERFEIFPKTSSWPNHDSNKGVTAACPHAGAKSFYKNLIWLVKKGNSYPKLSKSYINDKGKEVLVLWGIHHPSTSADQQSLYQNADTYVFVGSSRYSKKFKPEIAIRPKVRDQEGRMNYYWTLVEPGDKITFEATGNLVVPRYAFAMERNAGSGIIISDTPVHDCNTTCQTPKGAINTSLPFQNIHPITIGKCPKYVKSTKLRLATGLRNIPSIQSR"

def aa_to_input_ids(sequence):
    toks = tokenizer.encode(sequence, add_special_tokens=False)  # We don't want special tokens for amino acids
    return torch.as_tensor(toks).view(1, -1).to(model.device)

# Custom function to plot logit lens for GPTNeoXForCausalLM
def plot_gptneox_logit_lens(model, tokenizer, input_ids, start_ix=0, end_ix=20, probs=True):
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        
    hidden_states = outputs.hidden_states
    logits = outputs.logits
    
    num_layers = len(hidden_states)
    token_strs = [tokenizer.decode(t) for t in input_ids[0, start_ix:end_ix]]
    
    plt.figure(figsize=(20, 10))
    for layer in range(num_layers):
        layer_logits = model.gpt_neox.final_layer_norm(hidden_states[layer])
        layer_logits = model.embed_out(layer_logits)
        if probs:
            layer_preds = torch.softmax(layer_logits, dim=-1)
            correct_token_preds = layer_preds[0, start_ix:end_ix, input_ids[0, start_ix:end_ix]].detach().cpu().numpy()
        else:
            correct_token_ranks = (layer_logits[0, start_ix:end_ix] >= layer_logits[0, start_ix:end_ix, input_ids[0, start_ix:end_ix].unsqueeze(1)]).sum(dim=-1).detach().cpu().numpy()
            correct_token_preds = 1 / correct_token_ranks
        
        plt.plot(correct_token_preds, label=f'Layer {layer}')
    
    plt.xticks(range(len(token_strs)), token_strs, rotation=90)
    plt.xlabel('Token')
    plt.ylabel('Probability' if probs else 'Reciprocal Rank')
    plt.title(f"{'Probabilities' if probs else 'Reciprocal Ranks'} of correct tokens at each layer")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

# Generate input_ids from the amino acid sequences
input_ids1 = aa_to_input_ids(aa_sequence1)
input_ids2 = aa_to_input_ids(aa_sequence2)

# Truncate input_ids to 160 tokens
input_ids1 = input_ids1[:, :160]
input_ids2 = input_ids2[:, :160]

# Plot logit lens for the first sequence
plot_gptneox_logit_lens(model, tokenizer, input_ids1, start_ix=0, end_ix=20, probs=True)
plt.savefig("plot1_probs.svg", format="svg")
plt.close()

plot_gptneox_logit_lens(model, tokenizer, input_ids1, start_ix=0, end_ix=20, probs=False)
plt.savefig("plot1_ranks.svg", format="svg")
plt.close()

# Plot logit lens for the second sequence
plot_gptneox_logit_lens(model, tokenizer, input_ids2, start_ix=0, end_ix=20, probs=True)
plt.savefig("plot2_probs.svg", format="svg")
plt.close()

plot_gptneox_logit_lens(model, tokenizer, input_ids2, start_ix=0, end_ix=20, probs=False)
plt.savefig("plot2_ranks.svg", format="svg")
plt.close()

# Cleanup
cleanup_model(model)%   
