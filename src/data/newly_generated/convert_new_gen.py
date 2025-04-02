import json
import os
from tqdm import tqdm
from typing import List, Dict


def combine_jsonl_files_to_huggingface(input_files: List[str], output_file: str, remove_duplicates: bool = True):
    """
    Combine multiple JSONL files and convert them to a single Hugging Face formatted JSON file.
    
    Args:
        input_files (List[str]): List of paths to input JSONL files
        output_file (str): Path to the output JSON file
        remove_duplicates (bool): Whether to remove duplicate entries based on the hash field
    """
    print(f"Combining {len(input_files)} JSONL files into a single Hugging Face format file...")
    
    # Check if all input files exist
    missing_files = [f for f in input_files if not os.path.exists(f)]
    if missing_files:
        print(f"Error: The following input files do not exist: {', '.join(missing_files)}")
        return
    
    # Read all entries from the JSONL files
    all_entries = []
    unique_hashes = set()
    duplicates_found = 0
    
    for input_file in input_files:
        print(f"\nProcessing: {input_file}")
        entries_count = 0
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc=f"Reading {os.path.basename(input_file)}"):
                    if line.strip():
                        try:
                            entry = json.loads(line)
                            
                            # Handle duplicate entries if needed
                            if remove_duplicates:
                                if entry["hash"] in unique_hashes:
                                    duplicates_found += 1
                                    print(f"Duplicate entry found: {entry['hash']}")
                                    continue
                                unique_hashes.add(entry["hash"])
                            
                            all_entries.append(entry)
                            entries_count += 1
                            
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON in {input_file}: {e}")
                            continue
            
            print(f"Added {entries_count} entries from {input_file}")
            
        except Exception as e:
            print(f"Error processing file {input_file}: {e}")
    
    if duplicates_found > 0 and remove_duplicates:
        print(f"\nRemoved {duplicates_found} duplicate entries based on hash values")
    
    print(f"\nTotal entries collected: {len(all_entries)}")
    
    # Format for Hugging Face
    formatted_data = [
        {
            "content": entry["content"],
            "hash": entry["hash"],
            "conversations": entry["conversations"]
        } 
        for entry in all_entries
    ]
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)) or '.', exist_ok=True)
    
    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nSuccessfully combined and converted dataset to Hugging Face format.")
    print(f"Saved to: {output_file}")
    print(f"Total entries in final dataset: {len(formatted_data)}")
    
    # Print a sample entry
    if formatted_data:
        print("\nSample entry from combined dataset:")
        sample = formatted_data[0]
        print(f"Content: {sample['content']}")
        print("Conversations:")
        for idx, msg in enumerate(sample['conversations']):
            print(f"{idx+1}. {msg['role']}: {msg['content'][:100]}...")


if __name__ == "__main__":
    # Define your input files explicitly here
    input_files = [
        r"gemini_generation\1\dutch_ir_pairs_gemini.jsonl",
        r"gemini_generation\2\dutch_ir_pairs_gemini.jsonl",
        r"gemini_generation\3\dutch_ir_pairs_gemini.jsonl",
        r"gemini_generation\4\dutch_ir_pairs_gemini.jsonl",
        r"gemini_generation\5\dutch_ir_pairs_gemini.jsonl",
        r"gemini_generation\6\dutch_ir_pairs_gemini.jsonl",
        r"gemini_generation\7\dutch_ir_pairs_gemini.jsonl",
        r"gemini_generation\8\dutch_ir_pairs_gemini.jsonl",
        r"generating_data\dutch_ir_pairs_1.jsonl",
        r"generating_data\dutch_ir_pairs_2.jsonl",
        r"generating_data\dutch_ir_pairs_3.jsonl",
        r"generating_data\dutch_ir_pairs_4.jsonl",
        r"generating_data\dutch_ir_pairs_5.jsonl",
        r"generating_data_2\dutch_ir_pairs_1.jsonl",
        r"generating_data_2\dutch_ir_pairs_2.jsonl",
        r"generating_data_2\dutch_ir_pairs_3.jsonl",
        r"generating_data_2\dutch_ir_pairs_4.jsonl",
        r"generating_data_2\dutch_ir_pairs_5.jsonl",

        # Add more files as needed
    ]
    
    # Define output file
    output_file = "dutch_ir_pairs_huggingface_combined.json"
    
    # Set to False if you want to keep duplicates
    remove_duplicates = False
    
    # Run the conversion
    combine_jsonl_files_to_huggingface(
        input_files=input_files,
        output_file=output_file,
        remove_duplicates=remove_duplicates
    )
