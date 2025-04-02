import json
import os
import hashlib
import re
from tqdm import tqdm
from typing import List, Dict

def parse_text(text: str) -> List[Dict[str, str]]:
    """
    Parseert de 'text'-inhoud van een JSONL-regel om conversaties te extraheren.
    
    Args:
        text (str): De tekststring met conversaties gemarkeerd door speciale tokens.
    
    Returns:
        List[Dict[str, str]]: Een lijst van dictionaries met 'role' en 'content'.
    """
    # Regex-patroon om conversatiedelen te matchen: <start_of_turn>rol\ninhoud<end_of_turn>
    pattern = r"<start_of_turn>(\w+)\n(.*?)<end_of_turn>"
    matches = re.findall(pattern, text, re.DOTALL)
    conversations = []
    
    for match in matches:
        role = match[0].strip()
        # Map 'model' naar 'assistant' voor Hugging Face-compatibiliteit
        if role == "model":
            role = "assistant"
        content = match[1].strip()
        conversations.append({"role": role, "content": content})
    
    return conversations

def combine_jsonl_files_to_huggingface(input_files: List[str], output_file: str, remove_duplicates: bool = True) -> None:
    """
    Combineert meerdere JSONL-bestanden en zet ze om naar een enkel Hugging Face-geformatteerd JSON-bestand.
    
    Args:
        input_files (List[str]): Lijst van paden naar invoer JSONL-bestanden.
        output_file (str): Pad naar het uitvoer JSON-bestand.
        remove_duplicates (bool): Of duplicaten verwijderd moeten worden op basis van de hash.
    """
    print(f"Combining {len(input_files)} JSONL files into a single Hugging Face format file...")

    # Controleer of alle invoerbestanden bestaan
    missing_files = [f for f in input_files if not os.path.exists(f)]
    if missing_files:
        print(f"Error: The following input files do not exist: {', '.join(missing_files)}")
        return

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
                            text = entry["text"]
                            
                            # Parseer de conversaties uit het 'text'-veld
                            conversations = parse_text(text)
                            
                            # Genereer een hash op basis van de originele tekst
                            hash_value = hashlib.md5(text.encode()).hexdigest()
                            
                            # Behandel duplicaten indien nodig
                            if remove_duplicates:
                                if hash_value in unique_hashes:
                                    duplicates_found += 1
                                    print(f"Duplicate entry found: {hash_value}")
                                    continue
                                unique_hashes.add(hash_value)
                            
                            # Maak de geformatteerde entry voor Hugging Face
                            formatted_entry = {
                                "content": text,
                                "hash": hash_value,
                                "conversations": conversations
                            }
                            all_entries.append(formatted_entry)
                            entries_count += 1
                            
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON in {input_file}: {e}")
                            continue
                        except KeyError as e:
                            print(f"Missing 'text' field in {input_file}: {e}")
                            continue
            
            print(f"Added {entries_count} entries from {input_file}")
            
        except Exception as e:
            print(f"Error processing file {input_file}: {e}")

    if duplicates_found > 0 and remove_duplicates:
        print(f"\nRemoved {duplicates_found} duplicate entries based on hash values")
    
    print(f"\nTotal entries collected: {len(all_entries)}")
    
    # Maak de uitvoermap aan als die nog niet bestaat
    os.makedirs(os.path.dirname(os.path.abspath(output_file)) or '.', exist_ok=True)
    
    # Schrijf naar het uitvoerbestand
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=2)
    
    print(f"\nSuccessfully combined and converted dataset to Hugging Face format.")
    print(f"Saved to: {output_file}")
    print(f"Total entries in final dataset: {len(all_entries)}")
    
    # Print een voorbeeldentry
    if all_entries:
        print("\nSample entry from combined dataset:")
        sample = all_entries[0]
        print(f"Content: {sample['content'][:100]}...")
        print("Conversations:")
        for idx, msg in enumerate(sample['conversations']):
            print(f"{idx+1}. {msg['role']}: {msg['content'][:100]}...")

if __name__ == "__main__":
    # Definieer je invoerbestanden expliciet hier
    input_files = [
        r"dutch_formatted_chat_0.jsonl",
        r"dutch_formatted_chat_1.jsonl",
        r"dutch_formatted_chat_2.jsonl",
        r"dutch_formatted_chat_3.jsonl",
        r"dutch_formatted_chat_4.jsonl",
        r"dutch_formatted_chat_5.jsonl",
        r"dutch_formatted_chat_6.jsonl",
        r"dutch_formatted_chat_7.jsonl",
    ]
    
    # Definieer het uitvoerbestand
    output_file = "dutch_ir_pairs_huggingface_nemotron_combined.json"
    
    # Stel in op False als je duplicaten wilt behouden
    remove_duplicates = False
    
    # Voer de conversie uit
    combine_jsonl_files_to_huggingface(
        input_files=input_files,
        output_file=output_file,
        remove_duplicates=remove_duplicates
    )
