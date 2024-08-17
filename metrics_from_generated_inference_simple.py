import os
import re
from collections import Counter
import argparse

def process_file(file_path):
    tag_pattern = re.compile(r'<\|label\|>(\w+)')
    tags = []
    
    with open(file_path, 'r') as file:
        for line in file:
            match = tag_pattern.search(line)
            if match:
                tags.append(match.group(1))
    
    return tags

def calculate_metrics(tags):
    total = len(tags)
    counter = Counter(tags)
    metrics = {
        'total_samples': total,
        'counts': dict(counter),
        'percentages': {tag: count / total * 100 for tag, count in counter.items()}
    }
    return metrics

def process_directory(directory):
    results = {}
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):  # Adjust this if your files have a different extension
            file_path = os.path.join(directory, filename)
            tags = process_file(file_path)
            metrics = calculate_metrics(tags)
            results[filename] = metrics
    return results

def print_results(results):
    for filename, metrics in results.items():
        print(f"\nMetrics for {filename}:")
        print(f"Total samples: {metrics['total_samples']}")
        print("Counts:")
        for tag, count in metrics['counts'].items():
            print(f"  {tag}: {count}")
        print("Percentages:")
        for tag, percentage in metrics['percentages'].items():
            print(f"  {tag}: {percentage:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Generate metrics from LLM inference output files")
    parser.add_argument("directory", help="Directory containing the inference output files")
    args = parser.parse_args()

    results = process_directory(args.directory)
    print_results(results)

if __name__ == "__main__":
    main()
