import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# Set the backend to SVG
matplotlib.use('svg')

# Enable text as editable text objects in SVG
plt.rcParams['svg.fonttype'] = 'none'

# Read the CSV file
df = pd.read_csv('LORA-QLORA-FT-metrics.csv')

# Calculate combined F1 score
df['Combined_F1'] = df['Algal F1 Score'] + df['Bacterial F1 Score']

# Sort by Combined F1 score and keep the best performing instance for each Model
df_best = df.sort_values('Combined_F1', ascending=False).drop_duplicates('Model')

# Create a unique identifier for each model (now without checkpoint)
df_best['Model_ID'] = df_best['Model']

# Sort by Model for consistent ordering
df_best = df_best.sort_values('Model')

# Melt the DataFrame to long format
df_melted = df_best.melt(id_vars=['Model_ID'], 
                         value_vars=['Algal Precision', 'Algal Recall', 'Algal F1 Score',
                                     'Bacterial Precision', 'Bacterial Recall', 'Bacterial F1 Score'],
                         var_name='Metric', 
                         value_name='Score')

# Create a pivot table
df_pivot = df_melted.pivot(index='Model_ID', columns='Metric', values='Score')

# Create the heatmap
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(df_pivot, annot=True, cmap='YlOrRd', fmt='.2f', cbar_kws={'label': 'Score'}, ax=ax)
ax.set_title('Performance Metrics Heatmap (Best Checkpoint per Model)', fontsize=16, pad=20)
ax.set_ylabel('Model', fontsize=14, labelpad=20)
ax.set_xlabel('Metrics', fontsize=14, labelpad=20)

# Rotate x-axis labels for better readability
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
plt.setp(ax.get_yticklabels(), fontsize=12)

# Adjust layout to prevent clipping of labels
plt.tight_layout()

# Save the figure as PNG
plt.savefig('performance_metrics_heatmap_best_models.png', dpi=300, bbox_inches='tight')
print("Heatmap has been saved as 'performance_metrics_heatmap_best_models.png'")

# Save the figure as SVG with text as objects
plt.savefig('performance_metrics_heatmap_best_models.svg', format='svg', dpi=300, bbox_inches='tight')
print("Heatmap has been saved as 'performance_metrics_heatmap_best_models.svg'")

# If you want to display the plot (e.g., in a Jupyter notebook), uncomment the next line
# plt.show()

# Print out the best models and their combined F1 scores
print("\nBest performing models:")
for _, row in df_best.iterrows():
    print(f"{row['Model']} (Checkpoint {row['Checkpoint']}): Combined F1 = {row['Combined_F1']:.4f}")
