import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from wordcloud import WordCloud
import umap

def load_json_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def create_dataframe(data):
    df = pd.DataFrame(data)
    df['attributions'] = df['attributions'].apply(np.array)
    return df

def pad_sequence(seq, max_length):
    return np.pad(seq, (0, max_length - len(seq)), 'constant')

def pad_attributions(df):
    max_length = df['attributions'].apply(len).max()
    df['padded_attributions'] = df['attributions'].apply(lambda x: pad_sequence(x, max_length))
    return df

def plot_attribution_heatmap(df, n_samples=100, figsize=(20, 10), group=''):
    plt.figure(figsize=figsize)
    sample_df = df.sample(n_samples)
    padded_df = pad_attributions(sample_df)
    heatmap_data = np.vstack(padded_df['padded_attributions'])
    sns.heatmap(heatmap_data, cmap='coolwarm', center=0)
    plt.title(f'Attribution Heatmap - {group} (Sample of {n_samples} sequences)')
    plt.xlabel('Token Position')
    plt.ylabel('Sequence')
    plt.savefig(f'attribution_heatmap_{group.lower()}.svg', format='svg')
    plt.close()

def plot_average_attribution(df, top_n=20, figsize=(12, 6), group=''):
    padded_df = pad_attributions(df)
    avg_attributions = np.mean(np.vstack(padded_df['padded_attributions']), axis=0)
    top_indices = np.argsort(np.abs(avg_attributions))[-top_n:]
    
    plt.figure(figsize=figsize)
    sns.barplot(x=avg_attributions[top_indices], y=np.arange(top_n))
    plt.title(f'Top {top_n} Average Attributions - {group}')
    plt.xlabel('Average Attribution')
    plt.ylabel('Token Position')
    plt.savefig(f'average_attribution_{group.lower()}.svg', format='svg')
    plt.close()

def plot_probability_distribution(df, figsize=(10, 6), group=''):
    plt.figure(figsize=figsize)
    sns.histplot(df['positive_probability'], kde=True)
    plt.title(f'Distribution of Positive Probabilities - {group}')
    plt.xlabel('Positive Probability')
    plt.ylabel('Count')
    plt.savefig(f'probability_distribution_{group.lower()}.svg', format='svg')
    plt.close()

def plot_token_importance(df, top_n=20, figsize=(12, 8), group=''):
    token_importance = {}
    for _, row in df.iterrows():
        for token, attr in zip(row['tokens'], row['attributions']):
            if token in token_importance:
                token_importance[token].append(abs(attr))
            else:
                token_importance[token] = [abs(attr)]
    
    avg_importance = {k: np.mean(v) for k, v in token_importance.items()}
    top_tokens = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    plt.figure(figsize=figsize)
    sns.barplot(x=[t[1] for t in top_tokens], y=[t[0] for t in top_tokens])
    plt.title(f'Top {top_n} Important Tokens - {group}')
    plt.xlabel('Average Absolute Attribution')
    plt.ylabel('Token')
    plt.savefig(f'token_importance_{group.lower()}.svg', format='svg')
    plt.close()

def plot_attribution_clusters(df, n_clusters=5, figsize=(12, 8), group=''):
    padded_df = pad_attributions(df)
    X = np.vstack(padded_df['padded_attributions'])
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=figsize)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
    plt.colorbar(scatter)
    plt.title(f'Attribution Clusters - {group}')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.savefig(f'attribution_clusters_{group.lower()}.svg', format='svg')
    plt.close()

def generate_wordcloud(df, figsize=(12, 8), group=''):
    text = ' '.join([' '.join(tokens) for tokens in df['tokens']])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    plt.figure(figsize=figsize)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud of Tokens - {group}')
    plt.savefig(f'token_wordcloud_{group.lower()}.svg', format='svg')
    plt.close()

def plot_sequence_length_distribution(df, figsize=(10, 6), group=''):
    sequence_lengths = df['tokens'].apply(len)
    plt.figure(figsize=figsize)
    sns.histplot(sequence_lengths, kde=True)
    plt.title(f'Distribution of Sequence Lengths - {group}')
    plt.xlabel('Sequence Length')
    plt.ylabel('Count')
    plt.savefig(f'sequence_length_distribution_{group.lower()}.svg', format='svg')
    plt.close()

def generate_summary_stats(df, group=''):
    summary = {
        'total_sequences': len(df),
        'avg_positive_probability': df['positive_probability'].mean(),
        'median_positive_probability': df['positive_probability'].median(),
        'avg_sequence_length': df['tokens'].apply(len).mean(),
        'median_sequence_length': df['tokens'].apply(len).median(),
    }
    
    with open(f'summary_stats_{group.lower()}.json', 'w') as f:
        json.dump(summary, f, indent=2)

def plot_umap(df, n_neighbors=15, min_dist=0.1, figsize=(12, 8), group=''):
    padded_df = pad_attributions(df)
    X = np.vstack(padded_df['padded_attributions'])
    
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    embedding = reducer.fit_transform(X)
    
    plt.figure(figsize=figsize)
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=df['positive_probability'], cmap='viridis')
    plt.colorbar(scatter, label='Positive Probability')
    plt.title(f'UMAP Projection of Attributions - {group}')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.savefig(f'umap_projection_{group.lower()}.svg', format='svg')
    plt.close()

def generate_visualizations(df, group):
    plot_attribution_heatmap(df, group=group)
    plot_average_attribution(df, group=group)
    plot_probability_distribution(df, group=group)
    plot_token_importance(df, group=group)
    plot_attribution_clusters(df, group=group)
    generate_wordcloud(df, group=group)
    plot_sequence_length_distribution(df, group=group)
    generate_summary_stats(df, group=group)
    plot_umap(df, group=group)  # Add this line to include UMAP analysis

def main():
    # Load and process algal data
    algal_data = load_json_data('algal_captum_output.json')
    algal_df = create_dataframe(algal_data)
    
    # Load and process bacteria data
    bacteria_data = load_json_data('bacteria_captum_output.json')
    bacteria_df = create_dataframe(bacteria_data)
    
    # Generate visualizations for algal data
    generate_visualizations(algal_df, 'Algal')
    
    # Generate visualizations for bacteria data
    generate_visualizations(bacteria_df, 'Bacterial')

if __name__ == "__main__":
