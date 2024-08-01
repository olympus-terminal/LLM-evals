import torch
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.signal import find_peaks
import argparse
from transformers import AutoTokenizer, AutoConfig, GPTNeoXForCausalLM
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from tqdm import tqdm
from sklearn.decomposition import PCA
from collections import defaultdict
import matplotlib
import matplotlib.markers as mmarkers
import svgwrite
from svgwrite import cm, mm

matplotlib.use('Agg')  # Use the 'Agg' backend which is usually more robust for text rendering


# Set up argument parser
parser = argparse.ArgumentParser(description="Analyze amino acid sequences using a GPTNeoX model.")
parser.add_argument('file_paths', nargs='+', help='Paths to files containing amino acid sequences (one per line)')
args = parser.parse_args()

# Define the HuggingFace model namel

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

def read_sequences_from_file(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def aa_to_input_ids(sequence):
    toks = tokenizer.encode(sequence, add_special_tokens=False)
    return torch.as_tensor(toks).view(1, -1).to(model.device)

def analyze_sequence(model, tokenizer, input_ids):
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    return [h.squeeze(0).cpu().numpy() for h in hidden_states]

try:
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
except:
    print("DejaVu Sans font not available, using default font")

def plot_aa_representation(hidden_states, sequence, file_name):
    num_layers = len(hidden_states)
    all_aa = 'ACDEFGHIKLMNPQRSTVWYX'
    unique_aa = sorted(set(all_aa + ''.join(set(sequence) - set(all_aa))))
    
    colors = ['#' + ''.join([format(int(x*255), '02x') for x in c[:3]]) for c in plt.cm.tab20(np.linspace(0, 1, len(unique_aa)))]
    aa_to_color = dict(zip(unique_aa, colors))
    aa_to_color['C'] = '#FF0000'  # Red
    aa_to_color['M'] = '#0000FF'  # Blue
    
    pca = PCA(n_components=2)
    
    # A4 size in pixels (assuming 300 DPI)
    a4_width, a4_height = 2480, 3508
    
    # Calculate layout
    cols = 4
    rows = (num_layers + cols - 1) // cols
    plot_size = 500
    h_spacing = 100
    v_spacing = 100
    
    total_width = cols * plot_size + (cols - 1) * h_spacing
    total_height = rows * plot_size + (rows - 1) * v_spacing
    
    x_start = (a4_width - total_width) // 2
    y_start = (a4_height - total_height) // 2
    
    # Create SVG drawing
    dwg = svgwrite.Drawing(f"aa_representation_{file_name}.svg", size=(f"{a4_width}px", f"{a4_height}px"))
    
    # Add styles
    dwg.defs.add(dwg.style('.axis_text { font: bold 14px sans-serif; }'))
    dwg.defs.add(dwg.style('.title { font: bold 16px sans-serif; }'))
    
    for i, state in enumerate(hidden_states):
        reduced_state = pca.fit_transform(state)
        
        # Normalize the reduced state to fit in our SVG
        x_min, x_max = reduced_state[:, 0].min(), reduced_state[:, 0].max()
        y_min, y_max = reduced_state[:, 1].min(), reduced_state[:, 1].max()
        normalized_state = (reduced_state - [x_min, y_min]) / [x_max - x_min, y_max - y_min]
        
        # Calculate subplot position
        x_offset = x_start + (i % cols) * (plot_size + h_spacing)
        y_offset = y_start + (i // cols) * (plot_size + v_spacing)
        
        # Draw subplot border
        dwg.add(dwg.rect(insert=(x_offset, y_offset), size=(plot_size, plot_size), fill='none', stroke='black'))
        
        # Add title
        dwg.add(dwg.text(f'Layer {i}', insert=(x_offset + plot_size//2, y_offset - 10), text_anchor='middle', class_='title'))
        
        # Plot points
        for aa, color in aa_to_color.items():
            mask = np.array([s == aa for s in sequence])
            if np.any(mask):
                points = normalized_state[mask]
                for px, py in points:
                    dwg.add(dwg.circle(center=(x_offset + px*plot_size, y_offset + (1-py)*plot_size), r=10, fill=color, opacity=0.7))
        
        # Add axis labels
        dwg.add(dwg.text('PC1', insert=(x_offset + plot_size//2, y_offset + plot_size + 30), text_anchor='middle', class_='axis_text'))
        dwg.add(dwg.text('PC2', insert=(x_offset - 30, y_offset + plot_size//2), text_anchor='middle', transform=f'rotate(-90 {x_offset - 30} {y_offset + plot_size//2})', class_='axis_text'))
    
    # Add legend to the right side
    legend_x = x_start + total_width + 50
    legend_y = y_start
    for i, (aa, color) in enumerate(aa_to_color.items()):
        dwg.add(dwg.circle(center=(legend_x, legend_y + i*30), r=10, fill=color, opacity=0.7))
        dwg.add(dwg.text(aa, insert=(legend_x + 20, legend_y + i*30 + 5), class_='axis_text'))
    
    dwg.save()
    
def analyze_aa_influence(hidden_states, sequence):
    influence_scores = defaultdict(list)
    for layer, state in enumerate(hidden_states):
        for aa in set(sequence):
            mask = np.array([s == aa for s in sequence])
            if mask.sum() > 0:
                aa_state = state[mask].mean(axis=0)
                non_aa_state = state[~mask].mean(axis=0)
                influence = np.linalg.norm(aa_state - non_aa_state)
                influence_scores[aa].append(influence)
    return influence_scores

def plot_aa_influence(influence_scores, file_name):
    plt.figure(figsize=(12, 8))
    for aa, scores in influence_scores.items():
        plt.plot(range(len(scores)), scores, label=aa, linewidth=2 if aa in ['C', 'M'] else 1)
    plt.title("Amino Acid Influence Across Layers")
    plt.xlabel("Layer")
    plt.ylabel("Influence Score")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"aa_influence_{file_name}.svg", format="svg", bbox_inches='tight')
    plt.close()

def analyze_aa_influence(hidden_states, sequence):
    influence_scores = defaultdict(list)
    for layer, state in enumerate(hidden_states):
        for aa in set(sequence):
            mask = np.array([s == aa for s in sequence])
            if mask.sum() > 0:
                aa_state = state[mask].mean(axis=0)
                non_aa_state = state[~mask].mean(axis=0)
                influence = np.linalg.norm(aa_state - non_aa_state)
                influence_scores[aa].append(influence)
    return influence_scores

import numpy as np
from scipy.signal import find_peaks

def identify_influential_motifs(hidden_states, sequence, window_size=3, percentile=95):
    motifs = []
    sequence_length = len(sequence)
    
    # Calculate influence scores for each position
    position_influence = np.zeros((len(hidden_states), sequence_length))
    for layer, state in enumerate(hidden_states):
        for pos in range(sequence_length):
            pos_state = state[pos]
            other_state = np.mean(np.delete(state, pos, axis=0), axis=0)
            position_influence[layer, pos] = np.linalg.norm(pos_state - other_state)
    
    # Normalize influence scores
    position_influence = (position_influence - position_influence.min()) / (position_influence.max() - position_influence.min())
    
    # Calculate average influence across layers
    avg_influence = np.mean(position_influence, axis=0)
    
    # Find peaks in the average influence
    threshold = np.percentile(avg_influence, percentile)
    peaks, _ = find_peaks(avg_influence, height=threshold, distance=window_size)
    
    # Identify influential motifs around peaks
    for peak in peaks:
        start = max(0, peak - window_size // 2)
        end = min(sequence_length, peak + window_size // 2 + 1)
        motif = sequence[start:end]
        avg_score = np.mean(avg_influence[start:end])
        motifs.append((motif, start, avg_score))
    
    # Sort motifs by influence score
    motifs.sort(key=lambda x: x[2], reverse=True)
    
    return motifs, avg_influence, percentile

def print_influential_motifs(motifs, avg_influence, percentile, top_n=10):
    print(f"\nTop {top_n} Influential Motifs:")
    print("Motif | Start Position | Influence Score")
    print("-" * 40)
    for motif, start, score in motifs[:top_n]:
        print(f"{motif:5} | {start:15} | {score:.4f}")
    
    if not motifs:
        print("No motifs found above the threshold.")
        print(f"Average influence scores: min={avg_influence.min():.4f}, max={avg_influence.max():.4f}, mean={avg_influence.mean():.4f}")
        print(f"Try adjusting the percentile threshold (current: {percentile})")

# Modify the main execution part
for file_path in args.file_paths:
    sequences = read_sequences_from_file(file_path)
    file_name = file_path.split('/')[-1].split('.')[0]
    
    for idx, sequence in enumerate(sequences):
        print(f"\nAnalyzing sequence {idx+1} from {file_name}")
        input_ids = aa_to_input_ids(sequence)
        hidden_states = analyze_sequence(model, tokenizer, input_ids)
        
        plot_aa_representation(hidden_states, sequence, f"{file_name}_seq_{idx}")
        
        influence_scores = analyze_aa_influence(hidden_states, sequence)
        plot_aa_influence(influence_scores, f"{file_name}_seq_{idx}")
        
        # Identify and print influential motifs
        percentile = 95  # You can adjust this value
        motifs, avg_influence, used_percentile = identify_influential_motifs(hidden_states, sequence, window_size=3, percentile=percentile)
        print_influential_motifs(motifs, avg_influence, used_percentile)

# Cleanup
