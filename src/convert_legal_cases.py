import hashlib
import json
import re
from typing import List, Dict
import os

def parse_line_manually(text: str, line_num: int) -> List[Dict[str, str]]:
    """
    Parse the conversation text manually by searching for start and end markers.

    Args:
        text (str): The conversation text to parse.
        line_num (int): The line number for debugging purposes.

    Returns:
        List[Dict[str, str]]: List of dictionaries with 'role' and 'content'.
    """
    # Remove any <bos> tags if present
    text = text.replace("<bos>", "")
    # Replace literal "\\n" with actual newlines
    text = text.replace("\\n", "\n")
    conversations = []
    
    # Find all start and end positions of turns
    start_positions = [m.start() + len('<start_of_turn>') for m in re.finditer(r'<start_of_turn>', text)]
    end_positions = [m.start() for m in re.finditer(r'<end_of_turn>', text)]
    
    # Process each turn
    for i in range(min(len(start_positions), len(end_positions))):
        start_pos = start_positions[i]
        end_pos = end_positions[i]
        segment = text[start_pos:end_pos].strip()
        
        # Try splitting on newline first
        if '\n' in segment:
            role, content = segment.split('\n', 1)
            role = role.strip()
            content = content.strip()
        else:
            # If no newline, split on first whitespace
            parts = segment.split(None, 1)
            if len(parts) == 2:
                role, content = parts
                role = role.strip()
                content = content.strip()
            else:
                # Skip malformed segments
                continue
        
        # Standardize "model" to "assistant"
        if role == "model":
            role = "assistant"
        
        # Add valid role-content pair to conversations
        if role and content:
            conversations.append({"role": role, "content": content})
    
    return conversations

def compute_hash(text: str) -> str:
    """Compute a hash for the text for deduplication purposes."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def convert_to_huggingface_format(input_file: str, output_file: str, jsonl_output_file: str = None, remove_duplicates: bool = False):
    """
    Convert the input .txt file to Hugging Face format and save as JSON and optionally JSONL.

    Args:
        input_file (str): Path to the input .txt file.
        output_file (str): Path to the output JSON file.
        jsonl_output_file (str, optional): Path to the output JSONL file.
        remove_duplicates (bool): Whether to remove duplicate entries.
    """
    dataset = []
    seen_hashes = set() if remove_duplicates else None
    successful = 0
    failed = 0
    total_lines = 0
    
    print(f"Processing file: {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                total_lines += 1
                if line_num % 5000 == 0:
                    print(f"Processed {line_num} lines... (Successful: {successful}, Failed: {failed})")
                    break  # Uncomment this line to process the entire file
                
                # Clean the line
                line = line.strip()
                if not line:
                    continue
                
                # For .txt, the line is the conversation text
                text = line
                
                # Compute hash for deduplication
                hash_value = compute_hash(text)
                if remove_duplicates and hash_value in seen_hashes:
                    continue
                if remove_duplicates:
                    seen_hashes.add(hash_value)
                
                # Parse the conversation
                conversations = parse_line_manually(text, line_num)
                if conversations:
                    # Changed format to match the required structure with 'content', 'hash', and 'conversations'
                    entry = {"content": text, "hash": hash_value, "conversations": conversations}
                    dataset.append(entry)
                    successful += 1
                    if successful == 1:
                        print("\nFirst successful conversation:")
                        for conv in conversations:
                            print(f"  Role: {conv['role']}")
                            print(f"  Content (first 100 chars): {conv['content'][:100]}...")
                else:
                    failed += 1
                    if failed <= 5:
                        print(f"Failed to parse line {line_num}: {text[:100]}...")
    
    except Exception as e:
        print(f"Error processing the file: {e}")
    
    # Print summary
    print(f"Total lines processed: {total_lines}")
    print(f"Successfully parsed conversations: {successful}")
    print(f"Failed to parse: {failed}")
    
    # Save the dataset
    if dataset:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        if jsonl_output_file:
            with open(jsonl_output_file, 'w', encoding='utf-8') as f:
                for entry in dataset:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print("Conversion complete!")
    else:
        print("No valid conversations were found in the file.")

if __name__ == "__main__":
    # Update this path to your .txt file
    input_file = r"legal_case_dataset.txt"  # Example: r"C:\path\to\your\test.txt"
    output_file = "huggingface_legal_dataset.json"
    jsonl_output_file = "huggingface_legal_dataset.jsonl"
    remove_duplicates = False
    
    convert_to_huggingface_format(input_file, output_file, jsonl_output_file, remove_duplicates)