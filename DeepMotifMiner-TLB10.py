##DeepMotifMiner-TLB10.py

import torch
import gc
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import argparse
from transformers import AutoTokenizer, AutoConfig, GPTNeoXForCausalLM
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from sklearn.decomposition import PCA
from collections import defaultdict
import matplotlib
import svgwrite
from sklearn.manifold import TSNE
from umap import UMAP
import pandas as pd

matplotlib.use('Agg')

parser = argparse.ArgumentParser(description="Analyze amino acid sequences using a GPTNeoX model.")
parser.add_argument('file_paths', nargs='+', help='Paths to files containing amino acid sequences (one per line)')
args = parser.parse_args()

model_name = "ChlorophyllChampion/duality100s-ckpt-30000_pythia70m-arc"
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained("hmbyt5/byt5-small-english")
safetensors_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")
model = GPTNeoXForCausalLM(config)
state_dict = load_file(safetensors_path)
model.load_state_dict(state_dict, strict=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).half()

def cleanup_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()

def read_sequences_from_file(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def aa_to_input_ids(sequence):
    return torch.tensor(tokenizer.encode(sequence, add_special_tokens=False)).view(1, -1).to(device)

def analyze_sequence(model, input_ids):
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    return [h.squeeze(0).cpu().numpy() for h in outputs.hidden_states]

def plot_aa_representation(hidden_states, sequence, file_name):
    num_layers = len(hidden_states)
    all_aa = 'ACDEFGHIKLMNPQRSTVWYX'
    unique_aa = sorted(set(all_aa + ''.join(set(sequence) - set(all_aa))))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_aa)))
    aa_to_color = {aa: '#' + ''.join([format(int(x*255), '02x') for x in c[:3]]) for aa, c in zip(unique_aa, colors)}
    aa_to_color['C'] = '#FF0000'  # Red
    aa_to_color['M'] = '#0000FF'  # Blue
    aa_to_color['P'] = '#00FF00'  # Green (highlighting proline)
    
    pca = PCA(n_components=2)
    
    a4_width, a4_height = 2480, 3508
    cols, rows = 4, (num_layers + 3) // 4
    plot_size, h_spacing, v_spacing = 500, 100, 100
    
    total_width = cols * plot_size + (cols - 1) * h_spacing
    total_height = rows * plot_size + (rows - 1) * v_spacing
    
    x_start = (a4_width - total_width) // 2
    y_start = (a4_height - total_height) // 2
    
    dwg = svgwrite.Drawing(f"aa_representation_{file_name}.svg", size=(f"{a4_width}px", f"{a4_height}px"))
    dwg.defs.add(dwg.style('.axis_text { font: bold 14px sans-serif; }'))
    dwg.defs.add(dwg.style('.title { font: bold 16px sans-serif; }'))
    
    for i, state in enumerate(hidden_states):
        reduced_state = pca.fit_transform(state)
        normalized_state = (reduced_state - reduced_state.min(axis=0)) / (reduced_state.max(axis=0) - reduced_state.min(axis=0))
        
        x_offset = x_start + (i % cols) * (plot_size + h_spacing)
        y_offset = y_start + (i // cols) * (plot_size + v_spacing)
        
        dwg.add(dwg.rect(insert=(x_offset, y_offset), size=(plot_size, plot_size), fill='none', stroke='black'))
        dwg.add(dwg.text(f'Layer {i}', insert=(x_offset + plot_size//2, y_offset - 10), text_anchor='middle', class_='title'))
        
        for aa, color in aa_to_color.items():
            mask = np.array([s == aa for s in sequence])
            if np.any(mask):
                points = normalized_state[mask]
                for px, py in points:
                    circle = dwg.circle(center=(x_offset + px*plot_size, y_offset + (1-py)*plot_size), r=10, fill=color, opacity=0.7)
                    if aa == 'P':  # Highlight proline
                        circle.stroke(color='black', width=2)
                    dwg.add(circle)
        
        dwg.add(dwg.text('PC1', insert=(x_offset + plot_size//2, y_offset + plot_size + 30), text_anchor='middle', class_='axis_text'))
        dwg.add(dwg.text('PC2', insert=(x_offset - 30, y_offset + plot_size//2), text_anchor='middle', transform=f'rotate(-90 {x_offset - 30} {y_offset + plot_size//2})', class_='axis_text'))
    
    legend_x, legend_y = x_start + total_width + 50, y_start
    for i, (aa, color) in enumerate(aa_to_color.items()):
        circle = dwg.circle(center=(legend_x, legend_y + i*30), r=10, fill=color, opacity=0.7)
        if aa == 'P':
            circle.stroke(color='black', width=2)
        dwg.add(circle)
        dwg.add(dwg.text(aa, insert=(legend_x + 20, legend_y + i*30 + 5), class_='axis_text'))
    
    dwg.save()

def analyze_aa_influence(hidden_states, sequence):
    influence_scores = defaultdict(list)
    for state in hidden_states:
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
        plt.plot(range(len(scores)), scores, label=aa, linewidth=2 if aa in ['C', 'M', 'P'] else 1)
    plt.title("Amino Acid Influence Across Layers")
    plt.xlabel("Layer")
    plt.ylabel("Influence Score")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"aa_influence_{file_name}.svg", format="svg", bbox_inches='tight')
    plt.close()

def identify_influential_motifs(hidden_states, sequence, window_size=3, percentile=95):
    sequence_length = len(sequence)
    position_influence = np.array([np.linalg.norm(state - np.mean(state, axis=0), axis=1) for state in hidden_states])
    position_influence = (position_influence - position_influence.min()) / (position_influence.max() - position_influence.min())
    avg_influence = np.mean(position_influence, axis=0)
    
    threshold = np.percentile(avg_influence, percentile)
    peaks, _ = find_peaks(avg_influence, height=threshold, distance=window_size)
    
    motifs = [(sequence[max(0, peak-window_size//2):min(sequence_length, peak+window_size//2+1)],
               max(0, peak-window_size//2),
               np.mean(avg_influence[max(0, peak-window_size//2):min(sequence_length, peak+window_size//2+1)]))
              for peak in peaks]
    
    return sorted(motifs, key=lambda x: x[2], reverse=True), avg_influence, percentile

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

def aggregate_hidden_states(all_hidden_states):
    return np.concatenate([states[-1] for states in all_hidden_states])

def perform_dimensionality_reduction(aggregated_states, method='pca', n_components=2):
    reducers = {
        'pca': PCA(n_components=n_components),
        'tsne': TSNE(n_components=n_components, random_state=42),
        'umap': UMAP(n_components=n_components, random_state=42)
    }
    return reducers[method].fit_transform(aggregated_states)

def plot_reduced_states(reduced_states, all_sequences, method, output_file):
    plt.figure(figsize=(12, 10))
    unique_aa = sorted(set(''.join(all_sequences)))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_aa)))
    aa_to_color = dict(zip(unique_aa, colors))
    
    all_aa = [aa for seq in all_sequences for aa in seq]
    
    for aa in unique_aa:
        mask = np.array([s == aa for s in all_aa])
        if np.any(mask):
            scatter = plt.scatter(reduced_states[mask, 0], reduced_states[mask, 1], c=[aa_to_color[aa]], label=aa, alpha=0.6)
            if aa == 'P':  # Highlight proline
                scatter.set_edgecolors('black')
                scatter.set_linewidth(1)
    
    plt.title(f"Amino Acid Representation ({method.upper()})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_file, format='svg', dpi=300, bbox_inches='tight')
    plt.close()

def save_results_to_csv(all_sequences, reduced_states_dict, all_motifs, output_file):
    data = []
    for seq_idx, sequence in enumerate(all_sequences):
        for aa_idx, aa in enumerate(sequence):
            row = {
                'sequence_id': seq_idx,
                'position': aa_idx,
                'amino_acid': aa
            }
            for method, states in reduced_states_dict.items():
                row[f'{method}_component1'] = states[seq_idx * len(sequence) + aa_idx, 0]
                row[f'{method}_component2'] = states[seq_idx * len(sequence) + aa_idx, 1]
            
            motif_info = all_motifs[seq_idx]
            row['motif'] = motif_info['motif'] if aa_idx in range(motif_info['start'], motif_info['start'] + len(motif_info['motif'])) else ''
            row['motif_score'] = motif_info['score'] if aa_idx in range(motif_info['start'], motif_info['start'] + len(motif_info['motif'])) else 0
            
            data.append(row)
    
    pd.DataFrame(data).to_csv(output_file, index=False)

def main():
    all_sequences = []
    all_hidden_states = []
    all_motifs = []

    for file_path in args.file_paths:
        sequences = read_sequences_from_file(file_path)
        all_sequences.extend(sequences)

        for sequence in sequences:
            input_ids = aa_to_input_ids(sequence)
            hidden_states = analyze_sequence(model, input_ids)
            all_hidden_states.append(hidden_states)

            motifs, avg_influence, percentile = identify_influential_motifs(hidden_states, sequence)
            all_motifs.append({'motif': motifs[0][0], 'start': motifs[0][1], 'score': motifs[0][2]} if motifs else {})

            plot_aa_representation(hidden_states, sequence, f"{file_path.split('/')[-1].split('.')[0]}")
            influence_scores = analyze_aa_influence(hidden_states, sequence)
            plot_aa_influence(influence_scores, f"{file_path.split('/')[-1].split('.')[0]}")
            print_influential_motifs(motifs, avg_influence, percentile)

    aggregated_states = aggregate_hidden_states(all_hidden_states)
    reduced_states_dict = {}
    for method in ['pca', 'tsne', 'umap']:
        reduced_states = perform_dimensionality_reduction(aggregated_states, method=method)
        reduced_states_dict[method] = reduced_states
        plot_reduced_states(reduced_states, all_sequences, method, f"amino_acid_representation_{method}.svg")

    save_results_to_csv(all_sequences, reduced_states_dict, all_motifs, "amino_acid_analysis_results.csv")

    cleanup_model(model)

if __name__ == "__main__":
    main()
