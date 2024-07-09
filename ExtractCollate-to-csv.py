import re
import csv
import sys
from pathlib import Path

def remove_ansi_codes(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def safe_float(value, default=0.0):
    try:
        return float(value.strip())
    except (ValueError, AttributeError):
        return default

def extract_performance_metrics(text, filename):
    # Remove ANSI color codes
    clean_text = remove_ansi_codes(text)
    
    metrics = {
        'Algal Precision': 0.0,
        'Algal Recall': 0.0,
        'Algal F1 Score': 0.0,
        'Bacterial Precision': 0.0,
        'Bacterial Recall': 0.0,
        'Bacterial F1 Score': 0.0
    }
    
    # Extract performance metrics
    pattern = r'(Algal|Bacterial) (Precision|Recall|F1 Score): ([\d.]+)'
    matches = re.findall(pattern, clean_text)
    
    for match in matches:
        category, metric, value = match
        key = f"{category} {metric}"
        metrics[key] = safe_float(value)
    
    if not any(metrics.values()):
        print(f"WARNING: No performance metrics found in file: {filename}")
        print("File content:")
        print(clean_text)
    
    return metrics

def process_files(file_paths):
    all_data = []
    for file_path in file_paths:
        try:
            with open(file_path, 'r') as file:
                content = file.read()
                metrics = extract_performance_metrics(content, file_path)
                metrics['File Name'] = Path(file_path).stem
                all_data.append(metrics)
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            print(f"File content for {file_path}:")
            try:
                with open(file_path, 'r') as file:
                    print(file.read())
            except Exception as read_error:
                print(f"Could not read file content: {str(read_error)}")
    return all_data

def create_csv(all_data, filename='performance_metrics.csv'):
    if not all_data:
        print("No data to write to CSV.")
        return

    fieldnames = ['File Name', 'Algal Precision', 'Algal Recall', 'Algal F1 Score',
                  'Bacterial Precision', 'Bacterial Recall', 'Bacterial F1 Score']

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in all_data:
            writer.writerow(data)

    print(f"CSV file '{filename}' has been created.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_file1.txt> <input_file2.txt> ...")
        sys.exit(1)

    input_files = sys.argv[1:]
    all_data = process_files(input_files)
    create_csv(all_data)
