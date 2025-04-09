import json
import numpy as np
from openai import OpenAI
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
import pandas as pd
import time
import os
import logging
import concurrent.futures
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Download NLTK tokenizer if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Initialize the OpenAI client for Ollama
client = OpenAI(
    base_url='http://localhost:20000/v1',
    api_key='ollama',  # required, but unused
)

# Define models to test
models = [
    "aacudad/llama-3b-DUTCH",
    "aacudad/gemma-3-DUTCH", 
    "aacudad/dutchlegal-32k",
    "llama3.2:3b",
    "gemma3:latest"
]

# Define all datasets to test
datasets = [
    "dutch_ir_pairs_huggingface_combined.json",
    "dutch_ir_pairs_huggingface_nemotron_combined.json",
    "huggingface_legal_dataset.json",
    "dutch_parquet_huggingface_train_combined.json"
]

# Configure parallel processing
MAX_WORKERS = 10  # Increased as requested
# Use a thread-safe lock for logging to prevent output interleaving
log_lock = threading.Lock()

def thread_safe_log(level, message):
    """Thread-safe logging function"""
    with log_lock:
        if level == "info":
            logger.info(message)
        elif level == "error":
            logger.error(message)
        elif level == "warning":
            logger.warning(message)

def load_test_data(file_path, limit=5):
    """
    Load test data from JSON file with the expected format
    Format: [{"content": "...", "hash": "...", "conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}]
    """
    thread_safe_log("info", f"Loading test data from {file_path}")
    
    if not os.path.exists(file_path):
        thread_safe_log("error", f"File not found: {file_path}")
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract user queries and ground truth responses
        test_samples = []
        for item in data[:limit]:
            conversations = item.get('conversations', [])
            
            # Find user and assistant messages
            user_messages = [msg for msg in conversations if msg.get('role') == 'user']
            assistant_messages = [msg for msg in conversations if msg.get('role') == 'assistant']
            
            # Make sure we have both user and assistant messages
            if user_messages and assistant_messages:
                # For simplicity, we'll use the first user message as query and first assistant message as ground truth
                test_samples.append({
                    'query': user_messages[0]['content'],
                    'ground_truth': assistant_messages[0]['content'],
                    'dataset': os.path.basename(file_path)  # Track which dataset this came from
                })
        
        thread_safe_log("info", f"Successfully loaded {len(test_samples)} test samples from {file_path}")
        return test_samples
    
    except json.JSONDecodeError:
        thread_safe_log("error", f"Error decoding JSON from {file_path}")
        return []
    except Exception as e:
        thread_safe_log("error", f"Error loading test data: {e}")
        return []

def get_model_response(model_name, query, max_retries=3, retry_delay=2):
    """Get response from a model with retry logic"""
    for attempt in range(max_retries):
        try:
            thread_safe_log("info", f"Querying {model_name} (attempt {attempt+1}/{max_retries})")
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    # {"role": "system", "content": "Je bent een behulpzame, respectvolle en eerlijke assistent. Antwoord altijd in het Nederlands."},
                    {"role": "user", "content": query}
                ],
                # temperature=0.7,
                # options={
                #     "num_ctx": 4096  # Specify context length here
                # },
                max_tokens=1024
            )
            
            model_response = response.choices[0].message.content
            thread_safe_log("info", f"Successfully got response from {model_name}")
            return model_response
            
        except Exception as e:
            thread_safe_log("error", f"Error querying {model_name}: {e}")
            if attempt < max_retries - 1:
                thread_safe_log("info", f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
    
    thread_safe_log("error", f"Failed to get response from {model_name} after {max_retries} attempts")
    return "Error: Could not get response from model"

def calculate_rouge(prediction, reference):
    """Calculate ROUGE-1, ROUGE-2, and ROUGE-L scores"""
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    except Exception as e:
        thread_safe_log("error", f"Error calculating ROUGE scores: {e}")
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

def calculate_bleu(prediction, reference):
    """Calculate BLEU score with smoothing"""
    try:
        reference_tokens = nltk.word_tokenize(reference.lower())
        prediction_tokens = nltk.word_tokenize(prediction.lower())
        
        # BLEU requires a list of references
        smoothie = SmoothingFunction().method1
        score = sentence_bleu([reference_tokens], prediction_tokens, smoothing_function=smoothie)
        return score
    except Exception as e:
        thread_safe_log("error", f"Error calculating BLEU score: {e}")
        return 0.0

def process_model_response(model, query, ground_truth, dataset_source):
    """Process a single model response for a given query (for parallel execution)"""
    # Get model response
    prediction = get_model_response(model, query)
    
    # Calculate metrics
    rouge_scores = calculate_rouge(prediction, ground_truth)
    bleu_score = calculate_bleu(prediction, ground_truth)
    
    # Generate a preview of the response
    preview = prediction[:150] + "..." if len(prediction) > 150 else prediction
    thread_safe_log("info", f"Model {model} response: {preview}")
    thread_safe_log("info", f"Model {model} - ROUGE-1: {rouge_scores['rouge1']:.4f}, ROUGE-2: {rouge_scores['rouge2']:.4f}, "
                 f"ROUGE-L: {rouge_scores['rougeL']:.4f}, BLEU: {bleu_score:.4f}")
    
    # Return the results
    return {
        'model': model,
        'prediction': prediction,
        'rouge1': rouge_scores['rouge1'],
        'rouge2': rouge_scores['rouge2'],
        'rougeL': rouge_scores['rougeL'],
        'bleu': bleu_score,
        'dataset': dataset_source
    }

def evaluate_models_parallel(test_data, models):
    """Evaluate multiple models on test data in parallel"""
    # Initialize results dictionary to store metrics for each model
    results = {model: {
        'rouge1': [], 'rouge2': [], 'rougeL': [], 'bleu': [],
        'predictions': [], 'ground_truths': [], 'queries': [], 'datasets': []
    } for model in models}
    
    # Store all responses for detailed analysis
    all_responses = []
    
    # Process each test sample
    for i, sample in enumerate(tqdm(test_data, desc="Evaluating samples")):
        query = sample['query']
        ground_truth = sample['ground_truth']
        dataset_source = sample['dataset']
        
        thread_safe_log("info", f"\n--- Sample {i+1}/{len(test_data)} from {dataset_source} ---")
        thread_safe_log("info", f"Query: {query[:150]}..." if len(query) > 150 else f"Query: {query}")
        
        sample_responses = {
            'query': query,
            'ground_truth': ground_truth,
            'dataset': dataset_source,
            'responses': {}
        }
        
        # Process all models in parallel for this sample
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit tasks for each model
            for model in models:
                future = executor.submit(
                    process_model_response,
                    model=model,
                    query=query,
                    ground_truth=ground_truth,
                    dataset_source=dataset_source
                )
                futures.append(future)
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                model = result['model']
                
                # Store metrics and responses
                results[model]['rouge1'].append(result['rouge1'])
                results[model]['rouge2'].append(result['rouge2'])
                results[model]['rougeL'].append(result['rougeL'])
                results[model]['bleu'].append(result['bleu'])
                results[model]['predictions'].append(result['prediction'])
                results[model]['ground_truths'].append(ground_truth)
                results[model]['queries'].append(query)
                results[model]['datasets'].append(dataset_source)
                
                # Add to sample responses
                sample_responses['responses'][model] = {
                    'text': result['prediction'],
                    'rouge1': result['rouge1'],
                    'rouge2': result['rouge2'],
                    'rougeL': result['rougeL'],
                    'bleu': result['bleu']
                }
        
        all_responses.append(sample_responses)
    
    # Save detailed responses to file
    with open('detailed_responses.json', 'w', encoding='utf-8') as f:
        json.dump(all_responses, f, ensure_ascii=False, indent=2)
    
    return results

def summarize_results_with_dataset_columns(results):
    """Calculate average metrics and create summary dataframes with separate columns for each dataset"""
    # Get unique datasets
    all_datasets = set()
    for model in results:
        all_datasets.update(results[model]['datasets'])
    all_datasets = list(all_datasets)
    
    # Initialize the combined summary DataFrame
    combined_summary = {
        'Model': []
    }
    
    # Add columns for overall metrics
    combined_summary['Overall_ROUGE-1'] = []
    combined_summary['Overall_ROUGE-2'] = []
    combined_summary['Overall_ROUGE-L'] = []
    combined_summary['Overall_BLEU'] = []
    
    # Add columns for each dataset's metrics
    for dataset in all_datasets:
        dataset_key = dataset.replace('.json', '')
        combined_summary[f'{dataset_key}_ROUGE-1'] = []
        combined_summary[f'{dataset_key}_ROUGE-2'] = []
        combined_summary[f'{dataset_key}_ROUGE-L'] = []
        combined_summary[f'{dataset_key}_BLEU'] = []
        combined_summary[f'{dataset_key}_Sample_Count'] = []
    
    # Calculate metrics for each model
    for model in results:
        combined_summary['Model'].append(model)
        
        # Calculate overall metrics
        combined_summary['Overall_ROUGE-1'].append(np.mean(results[model]['rouge1']))
        combined_summary['Overall_ROUGE-2'].append(np.mean(results[model]['rouge2']))
        combined_summary['Overall_ROUGE-L'].append(np.mean(results[model]['rougeL']))
        combined_summary['Overall_BLEU'].append(np.mean(results[model]['bleu']))
        
        # Calculate dataset-specific metrics
        for dataset in all_datasets:
            dataset_key = dataset.replace('.json', '')
            
            # Filter results for this dataset
            dataset_indices = [i for i, ds in enumerate(results[model]['datasets']) if ds == dataset]
            
            if dataset_indices:
                combined_summary[f'{dataset_key}_ROUGE-1'].append(np.mean([results[model]['rouge1'][i] for i in dataset_indices]))
                combined_summary[f'{dataset_key}_ROUGE-2'].append(np.mean([results[model]['rouge2'][i] for i in dataset_indices]))
                combined_summary[f'{dataset_key}_ROUGE-L'].append(np.mean([results[model]['rougeL'][i] for i in dataset_indices]))
                combined_summary[f'{dataset_key}_BLEU'].append(np.mean([results[model]['bleu'][i] for i in dataset_indices]))
                combined_summary[f'{dataset_key}_Sample_Count'].append(len(dataset_indices))
            else:
                # No samples for this dataset for this model
                combined_summary[f'{dataset_key}_ROUGE-1'].append(np.nan)
                combined_summary[f'{dataset_key}_ROUGE-2'].append(np.nan)
                combined_summary[f'{dataset_key}_ROUGE-L'].append(np.nan)
                combined_summary[f'{dataset_key}_BLEU'].append(np.nan)
                combined_summary[f'{dataset_key}_Sample_Count'].append(0)
    
    # Create detailed results dataframe for each model
    for model in results:
        model_df = pd.DataFrame({
            'Query': results[model]['queries'],
            'Ground Truth': results[model]['ground_truths'],
            'Prediction': results[model]['predictions'],
            'Dataset': results[model]['datasets'],
            'ROUGE-1': results[model]['rouge1'],
            'ROUGE-2': results[model]['rouge2'],
            'ROUGE-L': results[model]['rougeL'],
            'BLEU': results[model]['bleu']
        })
        
        # Save model-specific results
        model_filename = f"{model.replace('/', '_')}_results.csv"
        model_df.to_csv(model_filename, index=False)
        logger.info(f"Saved detailed results for {model} to {model_filename}")
    
    # Create the combined summary dataframe
    combined_df = pd.DataFrame(combined_summary)
    
    # Also create a simplified overall summary for quick reference
    overall_summary = combined_df[['Model', 'Overall_ROUGE-1', 'Overall_ROUGE-2', 'Overall_ROUGE-L', 'Overall_BLEU']].copy()
    
    return combined_df, overall_summary

def main():
    """Main function to run the evaluation"""
    all_test_data = []
    
    # Process each dataset
    for dataset_file in datasets:
        if os.path.exists(dataset_file):
            # Load 5 samples from each dataset (adjust as needed)
            samples = load_test_data(dataset_file, limit=125)
            all_test_data.extend(samples)
        else:
            thread_safe_log("warning", f"Dataset file not found: {dataset_file}")
    
    if not all_test_data:
        thread_safe_log("error", "No test data loaded from any dataset, exiting")
        return
    
    thread_safe_log("info", f"Total test samples loaded: {len(all_test_data)}")
    
    # Evaluate models on all test data in parallel
    results = evaluate_models_parallel(all_test_data, models)
    
    # Get combined summary with dataset-specific columns
    combined_summary, overall_summary = summarize_results_with_dataset_columns(results)
    
    # Log and save overall summary
    thread_safe_log("info", "\nOverall Evaluation Summary:")
    thread_safe_log("info", "\n" + overall_summary.to_string(index=False))
    overall_summary.to_csv("overall_evaluation_summary.csv", index=False)
    
    # Log and save combined summary with all dataset metrics
    thread_safe_log("info", "\nCombined Evaluation Summary with Dataset-Specific Metrics:")
    thread_safe_log("info", "\n" + combined_summary.to_string(index=False))
    combined_summary.to_csv("combined_evaluation_summary.csv", index=False)
    
    # Print model ranking
    thread_safe_log("info", "\nOverall Model Ranking by ROUGE-L:")
    ranked_df = overall_summary.sort_values(by='Overall_ROUGE-L', ascending=False)
    thread_safe_log("info", "\n" + ranked_df.to_string(index=False))

if __name__ == "__main__":
    logger.info("Starting parallel model evaluation with dataset-specific metrics")
    main()
    logger.info("Evaluation complete")