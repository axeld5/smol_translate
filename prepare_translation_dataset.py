from datasets import load_dataset
import json
import random
import os

def prepare_translation_dataset(num_samples=200, output_file="translation_rl_data.json"):
    """
    Prepare the Tatoeba dataset for RL training.
    
    Args:
        num_samples (int): Number of samples to extract
        output_file (str): Output file path
    """
    print("Loading Tatoeba dataset...")
    dataset = load_dataset("tatoeba", lang1="en", lang2="fr")
    
    # Extract training samples
    train_data = dataset["train"]
    
    # Randomly sample num_samples examples
    random.seed(42)  # For reproducibility
    sample_indices = random.sample(range(len(train_data)), min(num_samples, len(train_data)))
    
    # Prepare data in the format expected by the RL trainer
    rl_data = []
    
    for idx in sample_indices:
        example = train_data[idx]
        english_text = example["translation"]["en"]
        french_text = example["translation"]["fr"]  # This is for reference, not used in training
        
        # Create the conversation format
        conversation = {
            "conversations": [
                {
                    "role": "user", 
                    "content": f"Translate the following sentence from English to French: '{english_text}'\nOutput the translation and only the translation."
                },
                {
                    "role": "assistant",
                    "content": french_text  # This is the reference translation
                }
            ]
        }
        
        rl_data.append(conversation)
    
    # Save the dataset
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(rl_data, f, indent=2, ensure_ascii=False)
    
    print(f"Dataset saved to {output_file}")
    print(f"Number of samples: {len(rl_data)}")
    
    # Show some examples
    print("\nFirst 3 examples:")
    for i, example in enumerate(rl_data[:3]):
        print(f"Example {i+1}:")
        print(f"  English: {example['conversations'][0]['content']}")
        print(f"  French: {example['conversations'][1]['content']}")
        print()

if __name__ == "__main__":
    prepare_translation_dataset(num_samples=200, output_file="translation_rl_data.json") 