from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from translation_rewards import compute_translation_rewards
import argparse
import os

def train_translation_rl_model(
    model_name="HuggingFaceTB/SmolLM2-360M-Instruct", 
    max_steps=500, 
    save_path="smollm2-360m-translation-rl", 
    dataset_file="translation_rl_data.json"
):
    """
    Train a translation model using GRPO (Generative Reinforcement Policy Optimization).
    
    Args:
        model_name (str): Name or path of the model to fine-tune
        max_steps (int): Maximum number of training steps
        save_path (str): Path to save the fine-tuned model
        dataset_file (str): Path to the translation dataset file
        
    Returns:
        dict: Training statistics
    """
    # Check if dataset exists
    if not os.path.exists(dataset_file):
        print(f"Dataset file {dataset_file} not found. Please run prepare_translation_dataset.py first.")
        return None
    
    # Load the dataset
    print(f"Loading dataset from: {dataset_file}")
    with open(dataset_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Prepare dataset for GRPO training
    rows = []
    for example in data:
        for turn in example["conversations"]:
            if turn["role"] == "user":
                question = turn["content"].strip()
                rows.append({"question": question, "prompt": [turn]})
                break

    dataset = Dataset.from_list(rows)
    print(f"Prepared {len(rows)} training examples")
    
    # Load the model
    print(f"Loading model: {model_name}")
    max_seq_length = 512
    
    checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    
    # Configure training parameters
    max_prompt_length = 256
    training_args = GRPOConfig(
        learning_rate=3e-5,  # Slightly higher learning rate for smaller model
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        logging_steps=10,
        per_device_train_batch_size=2,  # Larger batch size for 360M model
        gradient_accumulation_steps=2,
        num_generations=4,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_seq_length - max_prompt_length,
        num_train_epochs=1,
        max_steps=max_steps,
        save_steps=100,
        max_grad_norm=1.0,
        report_to="none",
        output_dir="outputs",
        remove_unused_columns=False,
    )
    
    # Select reward function based on is_reward_sparse parameter
    reward_func = compute_translation_rewards
    
    # Create the trainer
    print("Creating GRPO trainer")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=dataset,
    )
    
    # Train the model
    print(f"Starting translation RL training for {max_steps} steps")
    print("Reward structure:")
    print("- Language detection: -1 if not French, 0 if French")
    print("- Embedding similarity: Qwen3 embedding score between source and target")
    print("- Final reward: -1 if wrong language, else embedding similarity score")
    
    trainer_stats = trainer.train()
    
    # Save the model
    print(f"Saving model to: {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print("Translation RL training completed successfully!")
    return trainer_stats

def test_translation_model(model_path, test_sentences=None):
    """
    Test the trained translation model on some example sentences.
    
    Args:
        model_path (str): Path to the trained model
        test_sentences (list): List of English sentences to translate
    """
    if test_sentences is None:
        test_sentences = [
            "Hello, how are you?",
            "The weather is beautiful today.",
            "I love reading books in the evening.",
        ]
    
    print(f"Loading trained model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    print("\nTesting translations:")
    for sentence in test_sentences:
        prompt = f"Translate the following sentence from English to French: '{sentence}'\nOutput the translation and only the translation."
        messages = [{"role": "user", "content": prompt}]
        
        input_text = tokenizer.apply_chat_template(messages, tokenize=False)
        model_inputs = tokenizer([input_text], return_tensors="pt")
        
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True
        )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        
        print(f"EN: {sentence}")
        print(f"FR: {output}")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a translation model using GRPO")
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM2-360M-Instruct", 
                        help="Name or path of the model to fine-tune")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum number of training steps")
    parser.add_argument("--save_path", type=str, default="smollm2-360m-translation-rl", 
                        help="Path to save the fine-tuned model")
    parser.add_argument("--dataset_file", type=str, default="translation_rl_data.json",
                        help="Path to the translation dataset file")
    parser.add_argument("--test_only", action="store_true", help="Only test a trained model")
    
    args = parser.parse_args()
    
    if args.test_only:
        test_translation_model(args.save_path)
    else:
        # Run the training
        train_translation_rl_model(
            model_name=args.model_name,
            max_steps=args.max_steps,
            save_path=args.save_path,
            dataset_file=args.dataset_file
        ) 