#!/usr/bin/env python3
"""
Translation RL Pipeline for SmolLM2-360M-Instruct

This script provides a complete pipeline for training a translation model using reinforcement learning.
It includes:
1. Dataset preparation from Tatoeba (EN-FR)
2. Reward functions using Qwen3 embeddings and language detection
3. RL training using GRPO
4. Model testing and evaluation

Usage:
    python action.py --prepare_data    # Prepare the dataset
    python action.py --train          # Train the model  
    python action.py --test           # Test the trained model
    python action.py --demo           # Run a quick demo
"""

import argparse
import os
import sys
from prepare_translation_dataset import prepare_translation_dataset
from translation_rl_trainer import train_translation_rl_model, test_translation_model
from translation_rewards import compute_embedding_similarity, detect_language

def demo_translation():
    """Run a quick demo of the base model and reward functions."""
    print("=== Translation RL Pipeline Demo ===\n")
    
    # Load base model for demonstration
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("Loading base SmolLM2-360M-Instruct model...")
    checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    
    # Test sentences
    test_sentences = [
        "When he asked who had broken the window, all the boys put on an air of innocence.",
        "The weather is beautiful today.",
        "I love reading books in the evening."
    ]
    
    print("Testing base model translations and reward functions:\n")
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"Example {i}:")
        print(f"English: {sentence}")
        
        # Generate translation
        prompt = f"Translate the following sentence from English to French: '{sentence}'\nOutput the translation and only the translation."
        messages = [{"role": "user", "content": prompt}]
        
        input_text = tokenizer.apply_chat_template(messages, tokenize=False)
        model_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
        
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True
        )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        translation = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        
        print(f"French: {translation}")
        
        # Test reward functions
        detected_lang = detect_language(translation)
        embedding_score = compute_embedding_similarity(sentence, translation)
        
        print(f"Detected language: {detected_lang}")
        print(f"Embedding similarity: {embedding_score:.4f}")
        
        # Final reward
        final_reward = -1.0 if detected_lang != "fr" else embedding_score
        print(f"Final reward: {final_reward:.4f}")
        print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="Translation RL Pipeline")
    parser.add_argument("--prepare_data", action="store_true", help="Prepare the Tatoeba dataset")
    parser.add_argument("--train", action="store_true", help="Train the RL model")
    parser.add_argument("--test", action="store_true", help="Test the trained model")
    parser.add_argument("--demo", action="store_true", help="Run a quick demo")
    
    # Training parameters
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM2-360M-Instruct",
                        help="Model to fine-tune")
    parser.add_argument("--max_steps", type=int, default=500, help="Training steps")
    parser.add_argument("--save_path", type=str, default="smollm2-360m-translation-rl",
                        help="Model save path")
    parser.add_argument("--sparse_rewards", action="store_true", help="Use sparse rewards")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of training samples")
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any([args.prepare_data, args.train, args.test, args.demo]):
        parser.print_help()
        return
    
    if args.prepare_data:
        print("Preparing Tatoeba dataset...")
        prepare_translation_dataset(num_samples=args.num_samples)
        print("Dataset preparation completed!")
    
    if args.train:
        print("Starting RL training...")
        if not os.path.exists("translation_rl_data.json"):
            print("Dataset not found. Preparing dataset first...")
            prepare_translation_dataset(num_samples=args.num_samples)
        
        train_translation_rl_model(
            model_name=args.model_name,
            max_steps=args.max_steps,
            save_path=args.save_path,
            is_reward_sparse=args.sparse_rewards
        )
        print("Training completed!")
    
    if args.test:
        print("Testing trained model...")
        if not os.path.exists(args.save_path):
            print(f"Trained model not found at {args.save_path}")
            print("Please train the model first using --train")
            return
        
        test_translation_model(args.save_path)
    
    if args.demo:
        demo_translation()

if __name__ == "__main__":
    main()