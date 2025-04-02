import json
import re
from google import genai
from pathlib import Path
import time
from tqdm import tqdm
import uuid
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import os
import traceback
import argparse

# Initialize client with a proper API key
# Instead of hardcoding API keys or rotating them, use environment variable
# You should set this environment variable before running the script
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# List of models to try
models = [
    "gemini-1.5-pro",  # Using more mainstream models
    "gemini-1.5-flash"  # Using more mainstream models
]

def extract_conversations(text: str) -> List[Dict[str, str]]:
    """
    Extract user and assistant messages from conversation text.
    
    Args:
        text: The conversation text.
        
    Returns:
        List of dictionaries with user and assistant messages.
    """
    pattern = r"<start_of_turn>user\n(.*?)\n<end_of_turn>\n<start_of_turn>model\n(.*?)\n<end_of_turn>"
    matches = re.findall(pattern, text, re.DOTALL)
    
    conversations = []
    for user_msg, assistant_msg in matches:
        conversations.append({
            "user_msg": user_msg.strip(),
            "assistant_msg": assistant_msg.strip()
        })
    
    return conversations

def create_batch(conversations: List[Dict[str, Any]], processed_indices: set, 
                batch_size: int = 5, conversation_attempts: Dict[int, int] = None, 
                max_attempts: int = 3) -> List[Dict[str, Any]]:
    """
    Create a batch of up to batch_size conversations, skipping already processed 
    conversations and those exceeding max_attempts.
    
    Args:
        conversations: List of conversation dictionaries.
        processed_indices: Set of conversation indices already processed.
        batch_size: Maximum conversations per batch (default: 5).
        conversation_attempts: Dictionary tracking attempts per conversation index.
        max_attempts: Maximum attempts before skipping a conversation (default: 3).
    
    Returns:
        List of conversations to process in this batch.
    """
    if conversation_attempts is None:
        conversation_attempts = {}
    
    batch = []
    for conv in conversations:
        index = conv["index"]
        if index not in processed_indices:
            attempts = conversation_attempts.get(index, 0)
            if attempts < max_attempts:
                batch.append(conv)
                # Increment attempt count when including in a batch
                conversation_attempts[index] = attempts + 1
                if len(batch) >= batch_size:
                    break
    
    return batch

def translate_batch_to_dutch(batch: List[Dict[str, Any]], model_name: str) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Translate a batch of conversations to Dutch using Gemini API.
    
    Args:
        batch: List of conversation dictionaries.
        model_name: The model to use for translation.
        
    Returns:
        Tuple containing:
        - List of dictionaries with translated user and assistant messages
        - Boolean indicating success (True) or failure (False)
    """
    # Construct system prompt for batch translation
    num_conversations = len(batch)
    system_prompt = f"""
    Vertaal de volgende {num_conversations} conversaties vanuit elke taal naar het Nederlands.
    Elke conversatie bestaat uit een gebruikersbericht en een assistentbericht.
    
    BELANGRIJKE INSTRUCTIES:
    - Behoud alle originele opmaak, tags en speciale tekens
    - Vertaal GEEN tags, codeblokken, technische termen of variabelenamen
    - Vertaal ALLEEN de natuurlijke taalinhoud
    - Retourneer een geldige JSON-array met exact {num_conversations} vertaalde conversaties
    
    Geef je antwoord in de volgende JSON-structuur:
    {{
      "translations": [
        {{
          "id": 0,
          "user_translated": "vertaalde user bericht",
          "assistant_translated": "vertaalde assistant bericht"
        }},
        {{
          "id": 1,
          "user_translated": "vertaalde user bericht",
          "assistant_translated": "vertaalde assistant bericht"
        }},
        ...
      ]
    }}
"""
    
    # Prepare the content for the API call
    conversation_texts = []
    for i, conv in enumerate(batch):
        conversation_texts.append(f"CONVERSATION {i}:\nUser message:\n{conv['user_msg']}\n\nAssistant message:\n{conv['assistant_msg']}")
    
    contents = f"{system_prompt}\n\n" + "\n\n---\n\n".join(conversation_texts)
    
    # API call with retries
    max_retries = 3
    retry_delay = 10
    batch_id = str(uuid.uuid4())[:8]
    success = False
    
    retries = 0
    while retries <= max_retries and not success:
        try:
            print(f"Sending batch {batch_id} with {len(batch)} conversations using model {model_name}")
            
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config={
                    'response_mime_type': 'application/json',
                }
            )
            
            # Extract JSON from response
            response_text = response.text
            try:
                translation_data = json.loads(response_text)
            except json.JSONDecodeError:
                print(f"Failed to parse JSON response. Response text: {response_text[:500]}...")
                # Try to extract JSON using regex if direct parsing fails
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
                if json_match:
                    json_str = json_match.group(1)
                    translation_data = json.loads(json_str)
                else:
                    raise ValueError("Could not extract valid JSON from response")
            
            # Validate the response structure
            if "translations" not in translation_data or not isinstance(translation_data["translations"], list):
                raise ValueError(f"Invalid response format - missing translations array. Got: {translation_data}")
            
            if len(translation_data["translations"]) != len(batch):
                print(f"Warning: Expected {len(batch)} translations but got {len(translation_data['translations'])}")
            
            # Process translations
            translations_by_id = {item.get("id", i): item for i, item in enumerate(translation_data["translations"])}
            
            # Add translations to batch
            for i, conv in enumerate(batch):
                if i in translations_by_id:
                    conv["user_translated"] = translations_by_id[i].get("user_translated", conv["user_msg"])
                    conv["assistant_translated"] = translations_by_id[i].get("assistant_translated", conv["assistant_msg"])
                else:
                    print(f"Warning: Missing translation for conversation {i} in batch {batch_id}")
                    # If any translation is missing, consider the batch as failed
                    raise ValueError(f"Missing translation for conversation {i}")
            
            # Successful translation
            print(f"Successfully processed batch {batch_id}")
            success = True
            
        except Exception as e:
            error_str = str(e)
            print(f"Error with batch {batch_id}: {error_str[:200]}...")
            
            # Handle rate limiting and other errors
            if '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str:
                # Retry with exponential backoff
                retry_delay = min(retry_delay * 2, 60)  # Cap at 60 seconds
                print(f"Rate limit hit. Retrying in {retry_delay}s [Attempt {retries+1}/{max_retries}]...")
                time.sleep(retry_delay)
            elif 'invalid JSON' in error_str or 'schema' in error_str:
                # Retry for JSON formatting errors
                print(f"JSON formatting error. Retrying with model {model_name} [Attempt {retries+1}/{max_retries}]...")
                time.sleep(retry_delay)
            else:
                # For other errors, add longer delay
                print(f"General error. Retrying in {retry_delay * 2}s [Attempt {retries+1}/{max_retries}]...")
                time.sleep(retry_delay * 2)
                
            retries += 1
    
    return batch, success

def reconstruct_translated_text(original_data: Dict[str, Any], translated_convs: List[Dict[str, Any]]) -> str:
    """
    Reconstruct the full translated text by replacing original messages with translated ones.
    
    Args:
        original_data: The original conversation data from the input file.
        translated_convs: List of translated conversation turns for this index.
        
    Returns:
        The fully translated conversation text.
    """
    original_text = original_data.get("text", "")
    translated_text = original_text
    for conv in translated_convs:
        user_original = f"<start_of_turn>user\n{conv['user_msg']}\n<end_of_turn>"
        user_translated = f"<start_of_turn>user\n{conv['user_translated']}\n<end_of_turn>"
        translated_text = translated_text.replace(user_original, user_translated)
        assistant_original = f"<start_of_turn>model\n{conv['assistant_msg']}\n<end_of_turn>"
        assistant_translated = f"<start_of_turn>model\n{conv['assistant_translated']}\n<end_of_turn>"
        translated_text = translated_text.replace(assistant_original, assistant_translated)
    return translated_text

def save_checkpoint(checkpoint_file: str, processed_indices: set, failed_indices: set, 
                   last_processed_index: int, total_conversations: int, 
                   line_numbers: Dict[int, int] = None, 
                   conversation_attempts: Dict[int, int] = None,
                   current_model_index: int = 0):
    """
    Save detailed processing checkpoint to a file.
    
    Args:
        checkpoint_file: Path to the checkpoint file.
        processed_indices: Set of indices of successfully processed conversations.
        failed_indices: Set of indices of conversations that failed processing.
        last_processed_index: Index of the last processed conversation.
        total_conversations: Total number of conversations in the input file.
        line_numbers: Dictionary mapping indices to line numbers in the original file.
        conversation_attempts: Dictionary tracking attempts per conversation.
        current_model_index: Current index in the models list.
    """
    checkpoint_data = {
        "input_file": os.path.basename(checkpoint_file.rsplit('_checkpoint.json', 1)[0]),  # Just the filename
        "output_file": os.path.basename(checkpoint_file.replace('_checkpoint.json', '.jsonl')),  # Just the filename
        "processed_indices": sorted(list(processed_indices)),
        "failed_indices": sorted(list(failed_indices)),
        "last_processed_index": last_processed_index,
        "processed_line_numbers": line_numbers or {},  # Map of processed indices to line numbers
        "conversation_attempts": conversation_attempts or {},  # Track attempts per conversation
        "current_model_index": current_model_index,
        "total_conversations": total_conversations,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, indent=2)

def process_jsonl_file(input_file: str, output_file: str, batch_size: int = 5, checkpoint_file: str = None, 
                      shard: int = 0, total_shards: int = 1, max_attempts: int = 3,
                      start_model: str = "gemini-1.5-pro"):
    """
    Process a JSONL file containing conversations and translate them to Dutch in batches.
    
    Args:
        input_file: Path to the input JSONL file.
        output_file: Base path to the output JSONL file (will be modified with shard number).
        batch_size: Number of conversations to process at once (default: 5).
        checkpoint_file: Path to checkpoint file (default: generated from output_file).
        shard: The shard number (0-based index) to process (default: 0).
        total_shards: Total number of shards for parallel processing (default: 1).
        max_attempts: Maximum attempts before skipping a conversation permanently (default: 3).
        start_model: Initial model to use for processing (default: "gemini-1.5-pro").
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_file).parent
    output_path.mkdir(exist_ok=True)
    
    # Adjust output and checkpoint files for sharding
    if total_shards > 1:
        output_path = Path(output_file)
        output_file = str(output_path.with_name(f"{output_path.stem}_{shard}{output_path.suffix}"))
        if checkpoint_file is None:
            checkpoint_file = f"{output_file}_checkpoint.json"
        else:
            checkpoint_file = f"{checkpoint_file}_{shard}.json"
    elif checkpoint_file is None:
        checkpoint_file = f"{output_file}_checkpoint.json"
    
    # Initialize checkpoint data
    processed_indices = set()
    failed_indices = set()
    last_processed_index = -1
    processed_line_numbers = {}
    conversation_attempts = {}
    
    # Set initial model index
    model_index = models.index(start_model) if start_model in models else 0
    
    # Load checkpoint if exists
    checkpoint_path = Path(checkpoint_file)
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                loaded_checkpoint = json.load(f)
                if (loaded_checkpoint.get("input_file") == os.path.basename(input_file) and 
                    loaded_checkpoint.get("output_file") == os.path.basename(output_file)):
                    processed_indices = set(loaded_checkpoint.get("processed_indices", []))
                    failed_indices = set(loaded_checkpoint.get("failed_indices", []))
                    last_processed_index = loaded_checkpoint.get("last_processed_index", -1)
                    processed_line_numbers = loaded_checkpoint.get("processed_line_numbers", {})
                    conversation_attempts = loaded_checkpoint.get("conversation_attempts", {})
                    model_index = loaded_checkpoint.get("current_model_index", model_index)
                    
                    # Convert string keys to integers if needed (JSON serialization converts int keys to strings)
                    processed_line_numbers = {int(k): v for k, v in processed_line_numbers.items() if k.isdigit()}
                    conversation_attempts = {int(k): v for k, v in conversation_attempts.items() if k.isdigit()}
                    
                    # Print processed line numbers for better tracking
                    if processed_line_numbers:
                        line_nums = sorted(processed_line_numbers.values())
                        print(f"Loaded checkpoint: {len(processed_indices)} conversations already processed")
                        print(f"Processed line ranges: {line_nums[0]}-{line_nums[-1]}")
                else:
                    print("Warning: Checkpoint file exists but is for different files. Starting fresh.")
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}. Starting fresh.")
    
    # Load all conversations from the input file
    raw_conversations = []
    line_numbers = {}  # Map conversation indices to line numbers in the file
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):  # Start line numbering at 1
            try:
                data = json.loads(line)
                if "text" in data:
                    idx = len(raw_conversations)
                    raw_conversations.append(data)
                    line_numbers[idx] = line_num
            except json.JSONDecodeError:
                print(f"Skipping malformed line {line_num}: {line[:50]}...")
    
    print(f"Loaded {len(raw_conversations)} conversations from {input_file}")
    
    # Extract conversations and prepare data structure 
    all_conversations = []
    total_turns_per_index = defaultdict(int)
    
    for i, data in enumerate(raw_conversations):
        # Skip indices that are not for this shard
        if total_shards > 1 and i % total_shards != shard:
            continue
            
        # Skip already processed indices
        if i in processed_indices or i in failed_indices:
            continue
            
        text = data.get("text", "")
        conversations = extract_conversations(text)
        
        for conv in conversations:
            conv["original_text"] = text
            conv["index"] = i
            all_conversations.append(conv)
            total_turns_per_index[i] += 1
    
    print(f"Extracted {len(all_conversations)} conversation turns to translate")
    
    if not all_conversations:
        print("All assigned conversations have already been processed. Nothing to do.")
        return
    
    # Open output file in append mode
    with open(output_file, 'a', encoding='utf-8') as outfile:
        translated_turns_per_index = defaultdict(list)
        
        # Track progress
        remaining_conversations = len(set(conv["index"] for conv in all_conversations))
        with tqdm(total=remaining_conversations, desc="Translating conversations", unit="conv") as pbar:
            # While there are still conversations to process
            while all_conversations:
                # Get current model
                current_model = models[model_index]
                print(f"Using model: {current_model}")
                
                # Create a batch of conversations to process
                batch = create_batch(all_conversations, processed_indices, batch_size, 
                                   conversation_attempts, max_attempts)
                
                if not batch:
                    print("No more conversations to process or all remaining have reached max attempts.")
                    break
                
                batch_indices = set(conv["index"] for conv in batch)
                print(f"Processing batch with {len(batch)} turns from {len(batch_indices)} conversations")
                
                # Translate the batch
                translated_batch, success = translate_batch_to_dutch(batch, current_model)
                
                if success:
                    # Process successful translations
                    completed_indices = set()
                    
                    for conv in translated_batch:
                        index = conv["index"]
                        translated_turns_per_index[index].append(conv)
                        
                        # Check if all turns for this conversation are now translated
                        if len(translated_turns_per_index[index]) == total_turns_per_index[index]:
                            # Reconstruct and write the complete translated conversation
                            translated_text = reconstruct_translated_text(
                                raw_conversations[index], 
                                translated_turns_per_index[index]
                            )
                            
                            json.dump({"text": translated_text}, outfile, ensure_ascii=False)
                            outfile.write('\n')
                            
                            processed_indices.add(index)
                            completed_indices.add(index)
                            processed_line_numbers[index] = line_numbers[index]
                            
                            # Remove processed turns to free memory
                            del translated_turns_per_index[index]
                    
                    # Update progress bar with completed conversations
                    pbar.update(len(completed_indices))
                    
                    # Remove processed conversations from consideration
                    all_conversations = [
                        conv for conv in all_conversations 
                        if conv["index"] not in processed_indices
                    ]
                else:
                    # If the batch failed even after trying with the current model
                    print(f"Failed to process batch with model {current_model}. Trying next model...")
                    
                    # Move to the next model
                    model_index = (model_index + 1) % len(models)
                    
                    # If we've cycled through all models, mark conversations with too many attempts as failed
                    if model_index == 0:
                        failed_batch_indices = set()
                        for conv in batch:
                            index = conv["index"]
                            if conversation_attempts.get(index, 0) >= max_attempts:
                                failed_indices.add(index)
                                failed_batch_indices.add(index)
                        
                        if failed_batch_indices:
                            print(f"Permanently skipping {len(failed_batch_indices)} conversations after {max_attempts} failed attempts")
                            
                            # Remove failed conversations from all_conversations
                            all_conversations = [
                                conv for conv in all_conversations 
                                if conv["index"] not in failed_indices
                            ]
                
                # Save checkpoint after each batch
                save_checkpoint(
                    checkpoint_file,
                    processed_indices,
                    failed_indices,
                    max(processed_indices) if processed_indices else -1,
                    len(raw_conversations),
                    processed_line_numbers,
                    conversation_attempts,
                    model_index
                )
                
                # Small delay between batches to respect rate limits
                time.sleep(5)
    
    # Final report
    print(f"Translation completed for shard {shard}.")
    print(f"Processed {len(processed_indices)} conversations.")
    print(f"Failed to process {len(failed_indices)} conversations after {max_attempts} attempts.")
    print(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Translate conversations to Dutch using Gemini API')
    parser.add_argument('--input', required=True, help='Input JSONL file path')
    parser.add_argument('--output', required=True, help='Output JSONL file path')
    parser.add_argument('--checkpoint', default=None, help='Checkpoint file path')
    parser.add_argument('--batch-size', type=int, default=3, help='Batch size for API calls (minimum 1)')
    parser.add_argument('--shard', type=int, default=0, help='Shard number (0-based) to process')
    parser.add_argument('--total-shards', type=int, default=1, help='Total number of shards')
    parser.add_argument('--max-attempts', type=int, default=3, help='Maximum attempts before skipping a conversation')
    parser.add_argument('--start-model', default="gemini-1.5-pro", help='Initial model to use for processing')
    
    args = parser.parse_args()
    
    # Ensure API key is set
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable is not set.")
        print("Please set it before running the script:")
        print("  For Windows: set GEMINI_API_KEY=your_api_key")
        print("  For Linux/Mac: export GEMINI_API_KEY=your_api_key")
        return
    
    # Ensure batch_size is at least 1
    if args.batch_size <= 0:
        print(f"Warning: Invalid batch size {args.batch_size}. Setting to default value of 2.")
        args.batch_size = 2
    
    process_jsonl_file(
        input_file=args.input,
        output_file=args.output,
        batch_size=args.batch_size,
        checkpoint_file=args.checkpoint,
        shard=args.shard,
        total_shards=args.total_shards,
        max_attempts=args.max_attempts,
        start_model=args.start_model
    )

if __name__ == "__main__":
    main()
