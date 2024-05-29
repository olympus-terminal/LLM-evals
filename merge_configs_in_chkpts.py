##cat merge.py 
import json

# List of JSON files to merge
json_files = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "adapter_config.json",
    "trainer_state.json"
]

# Initialize an empty dictionary to store the merged data
merged_data = {}

# Iterate over the JSON files and merge their contents
for file_path in json_files:
    with open(file_path) as file:
        data = json.load(file)
        merged_data.update(data)

# Write the merged data to the output file
with open("config.json", "w") as output_file:
    json.dump(merged_data, output_file, indent=2)
