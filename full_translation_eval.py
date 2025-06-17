from translation_rewards import detect_language, compute_embedding_similarity
from google import genai
from google.genai import types
from transformers import AutoModelForCausalLM, AutoTokenizer
import tqdm
from datasets import load_dataset
import json
import os
from collections import defaultdict
import csv
import datetime
from dotenv import load_dotenv

load_dotenv()

def load_training_translations(file_path="translation_rl_data.json"):
    """
    Load all translations from the training dataset to check for duplicates.
    
    Args:
        file_path (str): Path to the training dataset JSON file
        
    Returns:
        dict: Dictionary mapping translations to their frequency in the training data
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Extract all translations from the training data
        translations = defaultdict(int)
        for item in data:
            for conversation in item["conversations"]:
                if conversation["role"] == "assistant":
                    translation = conversation["content"].strip()
                    translations[translation] += 1
        
        return translations
    except Exception as e:
        print(f"Error loading training translations: {e}")
        return {}

def load_translation_model(model_path):
    """
    Load a translation model from the specified path.
    
    Args:
        model_path (str): Path to the model
        
    Returns:
        tuple: (model, tokenizer) if successful, (None, None) if model not available
    """
    try:
        print(f"Loading translation model from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Move model to CUDA if available
        import torch
        if torch.cuda.is_available():
            model = model.to("cuda")
            print(f"Model moved to CUDA")
        else:
            print(f"CUDA not available, keeping model on CPU")
        
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model {model_path}: {str(e)}")
        print(f"Model {model_path} is not available. Skipping.")
        return None, None

def evaluate_translation_with_gemini(gemini_client, source_text, generated_translation, reference_translation):
    """
    Use Gemini to evaluate translation quality with a simple True/False score.
    
    Args:
        gemini_client: Initialized Gemini client
        source_text (str): Original English text
        generated_translation (str): Generated French translation
        reference_translation (str): Reference French translation
        
    Returns:
        dict: Evaluation result from Gemini
    """
    evaluation_prompt = f"""You are an expert French-English translation evaluator. Please evaluate whether the following translation is acceptable:

Source (English): "{source_text}"
Generated Translation (French): "{generated_translation}"
Reference Translation (French): "{reference_translation}"

Is the generated translation acceptable? Consider:
- Does it convey the original meaning accurately?
- Is it grammatically correct French?
- Is it natural and fluent?

Respond with ONLY "True" or "False".

Format: True/False"""

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-lite-preview-06-17",
            config=types.GenerateContentConfig(
                system_instruction=evaluation_prompt),
            contents="Hello there"
        )
        evaluation_text = response.text.strip()
        
        # Parse True/False from the response
        is_acceptable = False
        if evaluation_text.lower().startswith('true'):
            is_acceptable = True
        elif evaluation_text.lower().startswith('false'):
            is_acceptable = False
        else:
            # Try to find True/False in the text
            if 'true' in evaluation_text.lower() and 'false' not in evaluation_text.lower():
                is_acceptable = True
            elif 'false' in evaluation_text.lower() and 'true' not in evaluation_text.lower():
                is_acceptable = False
        
        return {
            'is_acceptable': is_acceptable,
            'evaluation_text': evaluation_text
        }
        
    except Exception as e:
        print(f"Error in Gemini evaluation: {e}")
        return {
            'is_acceptable': False,
            'evaluation_text': f"Error: {str(e)}"
        }

def eval_translation_model(model_path, prompts, source_texts, reference_translations, 
                          gemini_api_key, gemini_model_name="gemini-2.5-flash-lite-preview-06-17", 
                          print_interval=10, check_duplicates=True):
    """
    Evaluate a translation model using Gemini as the validator.
    
    Args:
        model_path (str): Path to the translation model to evaluate
        prompts (list): List of translation prompts
        source_texts (list): List of source English texts
        reference_translations (list): List of reference French translations
        gemini_api_key (str): Google API key for Gemini
        gemini_model_name (str): Gemini model to use for evaluation
        print_interval (int): Print results every N inferences
        check_duplicates (bool): Whether to check for duplicates in training data
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Load training translations if checking for duplicates
    training_translations = {}
    if check_duplicates:
        training_translations = load_training_translations()
        print(f"Loaded {len(training_translations)} unique translations from training data")
    
    # Load the translation model
    model, tokenizer = load_translation_model(model_path)
    
    # Skip evaluation if model is not available
    if model is None or tokenizer is None:
        return {
            "model_path": model_path,
            "gemini_acceptance_rate": 0.0,
            "language_accuracy": 0.0,
            "embedding_similarity_avg": 0.0,
            "duplicates": 0,
            "duplicate_percentage": 0.0,
            "total_samples": 0,
            "error": "Model not available"
        }
    
    # Initialize Gemini
    
    gemini_client = genai.Client(api_key=gemini_api_key)
        
    n = len(prompts)
    gemini_acceptable = 0
    language_correct = 0
    embedding_similarity_total = 0.0
    duplicates = 0
    
    # Print header for detailed output
    print("\n" + "="*100)
    print(f"Evaluating translation model: {model_path}")
    print(f"Using Gemini validator: {gemini_model_name}")
    print("="*100)
    
    for i, (prompt, source_text, reference_text) in enumerate(tqdm.tqdm(
        zip(prompts, source_texts, reference_translations), total=n)):
        
        # Generate translation using the model
        messages = [prompt]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        try:
            # Get model device and move inputs to the same device
            model_device = next(model.parameters()).device
            inputs = tokenizer([text], return_tensors="pt").to(model_device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
            )
            
            # Extract generated translation
            full_output = tokenizer.batch_decode(outputs)[0]
            if "model\n" in full_output and "<end_of_turn>" in full_output:
                generated_translation = full_output.split("model\n")[1].split("<end_of_turn>")[0].strip()
            else:
                # Fallback parsing
                generated_translation = full_output.split(text)[-1].strip()
                if generated_translation.startswith("assistant\n"):
                    generated_translation = generated_translation[10:].strip()
            
        except Exception as e:
            print(f"Error generating translation for sample {i}: {e}")
            generated_translation = ""
        
        # Language detection
        detected_lang = detect_language(generated_translation)
        is_french = (detected_lang == "fr")
        if is_french:
            language_correct += 1
        
        # Embedding similarity
        embedding_score = compute_embedding_similarity(source_text, generated_translation)
        embedding_similarity_total += embedding_score
        
        # Gemini evaluation
        gemini_eval = evaluate_translation_with_gemini(
            gemini_client, source_text, generated_translation, reference_text
        )
        
        is_acceptable = gemini_eval['is_acceptable']
        if is_acceptable:
            gemini_acceptable += 1
        
        # Check for duplicates
        is_duplicate = False
        if check_duplicates and generated_translation in training_translations:
            duplicates += 1
            is_duplicate = True
        
        # Print detailed results every print_interval inferences
        if (i + 1) % print_interval == 0:
            print(f"\n--- Inference {i+1}/{n} ---")
            print(f"Source (EN): {source_text}")
            print(f"Reference (FR): {reference_text}")
            print(f"Generated (FR): {generated_translation}")
            print(f"Language: {detected_lang} ({'✓' if is_french else '✗'})")
            print(f"Embedding Score: {embedding_score:.3f}")
            print(f"Gemini Acceptable: {'✓' if is_acceptable else '✗'}")
            if is_duplicate:
                print(f"⚠️ DUPLICATE: This translation appears {training_translations[generated_translation]} times in training data")
            print("-" * 100)
    
    # Calculate averages
    gemini_acceptance_rate = gemini_acceptable / n if n > 0 else 0.0
    language_accuracy = language_correct / n if n > 0 else 0.0
    embedding_similarity_avg = embedding_similarity_total / n if n > 0 else 0.0
    
    # Print summary
    print("\n" + "="*100)
    print("TRANSLATION EVALUATION SUMMARY")
    print("="*100)
    print(f"Model: {model_path}")
    print(f"Gemini Validator: {gemini_model_name}")
    print(f"Total samples: {n}")
    print(f"Language accuracy (French): {language_accuracy:.4f} ({language_correct}/{n})")
    print(f"Average embedding similarity: {embedding_similarity_avg:.4f}")
    print(f"Gemini acceptance rate: {gemini_acceptance_rate:.4f} ({gemini_acceptable}/{n})")
    if check_duplicates:
        print(f"Duplicate translations: {duplicates} ({duplicates/n*100:.2f}%)")
    print("="*100)
    
    return {
        "model_path": model_path,
        "gemini_acceptance_rate": gemini_acceptance_rate,
        "language_accuracy": language_accuracy,
        "embedding_similarity_avg": embedding_similarity_avg,
        "duplicates": duplicates,
        "duplicate_percentage": duplicates/n if n > 0 else 0.0,
        "total_samples": n
    }

def save_translation_eval_results_to_csv(results, filename=None):
    """
    Save translation evaluation results to a CSV file.
    
    Args:
        results (list): List of dictionaries containing evaluation results
        filename (str, optional): Name of the CSV file. If None, a timestamped name will be used.
    """
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"translation_evaluation_results_{timestamp}.csv"
    
    fieldnames = ["model_path", "gemini_acceptance_rate", "language_accuracy", 
                 "embedding_similarity_avg", "duplicates", "duplicate_percentage", 
                 "total_samples", "error"]
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            # Ensure all fields are present
            for field in fieldnames:
                if field not in result:
                    result[field] = ""
            writer.writerow(result)
    
    print(f"\nResults saved to {filename}")

def prepare_evaluation_data(num_samples=200):
    """
    Prepare evaluation data from Tatoeba dataset.
    
    Args:
        num_samples (int): Number of samples to use for evaluation
        
    Returns:
        tuple: (prompts, source_texts, reference_translations)
    """
    print(f"Loading Tatoeba EN-FR dataset for evaluation ({num_samples} samples)...")
    dataset = load_dataset("tatoeba", lang1="en", lang2="fr", trust_remote_code=True)
    train_data = dataset["train"]
    
    prompts = []
    source_texts = []
    reference_translations = []
    
    for i in range(min(num_samples, len(train_data))):
        example = train_data[i]
        english_text = example["translation"]["en"]
        french_text = example["translation"]["fr"]
        
        prompt = {
            "role": "user",
            "content": f"Translate the following sentence from English to French: '{english_text}'\nOutput the translation and only the translation."
        }
        
        prompts.append(prompt)
        source_texts.append(english_text)
        reference_translations.append(french_text)
    
    return prompts, source_texts, reference_translations

if __name__ == "__main__":
    # Get Gemini API key
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        print("Please set the GEMINI_API_KEY environment variable")
        exit(1)
    
    # Prepare evaluation data
    prompts, source_texts, reference_translations = prepare_evaluation_data(num_samples=100)  # Smaller sample for demo
    
    # List of translation models to evaluate
    models = [
        "HuggingFaceTB/SmolLM2-360M-Instruct",  # Base model
        "smollm2-360m-translation-rl",          # RL-trained model
        # Add more models as they become available
    ]
    
    # Get models from command line if provided
    import sys
    if len(sys.argv) > 1:
        models = sys.argv[1:]
    
    # Run evaluation for each model
    all_results = []
    for model_path in models:
        print(f"\n{'='*100}")
        print(f"EVALUATING TRANSLATION MODEL: {model_path}")
        print(f"{'='*100}")
        
        results = eval_translation_model(
            model_path=model_path,
            prompts=prompts,
            source_texts=source_texts,
            reference_translations=reference_translations,
            gemini_api_key=gemini_api_key,
            print_interval=5  # Print more frequently for demo
        )
        all_results.append(results)
        
        # Print final results for this model
        print("\nFINAL RESULTS:")
        print(f"Model: {model_path}")
        if "error" in results and results["error"]:
            print(f"Error: {results['error']}")
        else:
            print(f"Language Accuracy: {results['language_accuracy']:.4f}")
            print(f"Embedding Similarity: {results['embedding_similarity_avg']:.4f}")
            print(f"Gemini Acceptance Rate: {results['gemini_acceptance_rate']:.4f}")
            print(f"Duplicate Translations: {results['duplicates']} ({results['duplicate_percentage']*100:.2f}%)")
    
    # Save all results to a CSV file
    save_translation_eval_results_to_csv(all_results)
    
    # Print a summary table of all results
    print("\n" + "="*80)
    print("SUMMARY OF ALL TRANSLATION MODELS")
    print("="*80)
    print(f"{'Model':<35} {'Lang Acc':<10} {'Embed':<8} {'Gemini':<8} {'Dups%':<6} {'Status':<8}")
    print("-"*80)
    for result in all_results:
        status = "Error" if "error" in result and result["error"] else "Success"
        print(f"{result['model_path'][:34]:<35} "
              f"{result['language_accuracy']:.3f}      "
              f"{result['embedding_similarity_avg']:.3f}    "
              f"{result['gemini_acceptance_rate']:.3f}    "
              f"{result['duplicate_percentage']*100:.1f}%   "
              f"{status:<8}")
    print("="*80) 