import json
import os
file_path = '/Users/tzhang/projects/LAPDOG/ckpt/xl/ckpt/valid-result.jsonl'
output_dir = '/Users/tzhang/projects/LAPDOG/ckpt/xl/ckpt/result_split/'


# Read the jsonl file and save each entry to a separate JSON file
with open(file_path, 'r') as file:
    for index, line in enumerate(file):
        entry = json.loads(line)
        
        # Define the output file path for each entry
        output_file_path = os.path.join(output_dir, f'entry_{index + 1}.json')
        
        # Write the entry to a separate JSON file
        with open(output_file_path, 'w') as output_file:
            json.dump(entry, output_file, indent=4)  # Save JSON with indentation for readability
            
        print(f'Saved entry {index + 1} to {output_file_path}')