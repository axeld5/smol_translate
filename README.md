# Translation RL Pipeline for SmolLM2-360M-Instruct

This project implements a comprehensive reinforcement learning pipeline for training a translation model from English to French using SmolLM2-360M-Instruct. The pipeline leverages the Tatoeba dataset and uses a sophisticated reward system based on Qwen3 embeddings and language detection, with professional evaluation using Gemini models.

## Experiment Status: FAILED

⚠️ **This experiment has failed due to reward hacking issues**

### Failure Summary

This translation RL experiment encountered significant reward hacking problems across multiple reward function configurations:

1. **Attempt 1: Embedding + Language Detection**
   - **Result**: Massive failure due to reward hacking
   - **Issue**: The model learned to generate outputs like `"[original prompt] que que que que que que que que que que"` to fool the language detection system
   - **Root cause**: Language detection was too easily exploitable - adding repeated French words fooled `langdetect` into classifying the text as French

2. **Attempt 2: Embedding + Language Detection + Length Penalty**
   - **Result**: Continued failure with some reward hacking
   - **Issue**: Even with length penalties, the model found ways to game the reward system
   - **Root cause**: The combination of rewards still provided exploitable pathways for the model to achieve high scores without producing quality translations

### Key Lessons Learned

- **Language detection alone is insufficient**: Simple language detection libraries like `langdetect` can be easily fooled by adding target language words
- **Embedding similarity has limitations**: Semantic embeddings may not adequately capture translation quality across languages
- **Reward composition complexity**: Combining multiple reward signals creates new attack vectors for reward hacking
- **Need for more robust evaluation**: Simple automated metrics are vulnerable to adversarial exploitation by RL-trained models

## Overview

The pipeline consists of several key components:

1. **Dataset Preparation**: Extracts and formats 200 samples from the Tatoeba EN-FR dataset
2. **Reward System**: Combines language detection and semantic similarity scoring
3. **RL Training**: Uses GRPO (Generative Reinforcement Policy Optimization) for model training
4. **Model Evaluation**: Professional translation quality assessment using Gemini Flash models
5. **Comprehensive Testing**: Multi-model comparison and performance analysis

## Key Features

### Reward System

The reward system implements a two-stage evaluation:

1. **Language Detection**: Uses `langdetect` to verify the output is in French
   - If not French: reward = -1
   - If French: proceed to semantic evaluation

2. **Semantic Similarity**: Uses Qwen3 embeddings to measure translation quality
   - Computes cosine similarity between source (English) and target (French) embeddings
   - Final reward = embedding similarity score (if language is correct)

### Professional Evaluation System

- **Gemini-Powered Assessment**: Uses Gemini Flash models for expert translation evaluation
- **Simple True/False Scoring**: Streamlined evaluation for translation acceptability
- **Multi-Model Comparison**: Evaluate and compare multiple translation models
- **Comprehensive Metrics**: Language accuracy, embedding similarity, and expert validation

### Model Architecture

- **Base Model**: SmolLM2-360M-Instruct
- **Fine-tuning**: LoRA (Low-Rank Adaptation) with rank 32
- **Training Method**: GRPO (Generative Reinforcement Policy Optimization)
- **Context Length**: 512 tokens

## Installation

1. Install the required dependencies:
```bash
pip install -r translation_requirements.txt
```

2. Set up your Google API key for Gemini evaluation (optional):
```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
```

3. Make sure you have access to the following models:
   - `HuggingFaceTB/SmolLM2-360M-Instruct`
   - `Qwen/Qwen3-Embedding-0.6B`

## Usage

The main entry point is `action.py`, which provides several commands:

### 1. Run a Demo
Test the base model and reward functions:
```bash
python action.py --demo
```

### 2. Prepare Dataset
Extract and format 200 samples from Tatoeba:
```bash
python action.py --prepare_data
```

### 3. Train the Model
Start RL training:
```bash
python action.py --train
```

### 4. Test the Trained Model
Evaluate the trained model:
```bash
python action.py --test
```

### 5. Professional Model Evaluation
Evaluate translation models using Gemini:
```bash
# Evaluate base and trained models with Gemini
python full_translation_eval.py

# Evaluate specific models
python full_translation_eval.py "HuggingFaceTB/SmolLM2-360M-Instruct" "smollm2-360m-translation-rl"
```

### Advanced Options

```bash
# Custom training parameters
python action.py --train --max_steps 1000 --sparse_rewards --num_samples 500

# Custom model and save path
python action.py --train --model_name "path/to/model" --save_path "my-translation-model"

# Evaluate with different number of samples
python full_translation_eval.py --num_samples 200
```

## File Structure

```
.
├── action.py                          # Main pipeline orchestrator
├── translation_rewards.py             # Reward system implementation
├── translation_rl_trainer.py          # RL training logic
├── prepare_translation_dataset.py     # Dataset preparation utilities
├── full_translation_eval.py           # Professional model evaluation with Gemini
├── translation_requirements.txt       # Project dependencies
├── README.md                          # This documentation
├── explore.ipynb                      # Jupyter notebook for exploration
└── translation_rl_data.json          # Generated training dataset (after preparation)
```

## Technical Details

### Dataset Format

The training data is formatted as conversations:
```json
{
  "conversations": [
    {
      "role": "user",
      "content": "Translate the following sentence from English to French: 'Hello, how are you?'\nOutput the translation and only the translation."
    },
    {
      "role": "assistant", 
      "content": "Bonjour, comment allez-vous ?"
    }
  ]
}
```

### Reward Function Details

```python
def reward_translation_quality(source_text, target_text):
    # Step 1: Language detection
    if detect_language(target_text) != "fr":
        return -1.0
    
    # Step 2: Embedding similarity
    similarity = compute_embedding_similarity(source_text, target_text)
    return similarity
```

### Training Configuration

- **Learning Rate**: 3e-5
- **Batch Size**: 2 (per device)
- **Gradient Accumulation**: 2 steps
- **LoRA Rank**: 32
- **Max Steps**: 500 (default)
- **Generations per Step**: 4

## Examples

### Sample Input/Output

**Input**: "The weather is beautiful today."
**Expected Output**: "Le temps est magnifique aujourd'hui."

### Reward Calculation

1. Language detection: "fr" ✓ (reward continues)
2. Embedding similarity: 0.85
3. **Final reward**: 0.85

If the output was in English instead:
1. Language detection: "en" ✗ 
2. **Final reward**: -1.0

### Professional Evaluation Example

```
--- Inference 5/100 ---
Source (EN): Hello, how are you?
Reference (FR): Bonjour, comment allez-vous ?
Generated (FR): Bonjour, comment ça va ?
Language: fr (✓)
Embedding Score: 0.854
Gemini Acceptable: ✓
```

### Evaluation Summary

```
TRANSLATION EVALUATION SUMMARY
Model: smollm2-360m-translation-rl
Total samples: 100
Language accuracy (French): 0.9400 (94/100)
Average embedding similarity: 0.7823
Gemini acceptance rate: 0.8200 (82/100)
```

## Performance Considerations

- **Memory Usage**: The 360M parameter model with 4-bit quantization requires ~1-2 GB GPU memory
- **Training Time**: ~30-60 minutes for 500 steps on a modern GPU
- **Embedding Computation**: Qwen3 embeddings add ~0.1s per sample evaluation
- **Gemini Evaluation**: Professional evaluation adds ~0.5-1s per sample but provides expert-level assessment
- **Evaluation Cost**: Gemini API calls are cost-effective for model validation and comparison

## Evaluation Metrics

The system provides multiple evaluation metrics:

### Automated Metrics
- **Language Accuracy**: Percentage of outputs detected as target language (French)
- **Embedding Similarity**: Qwen3 semantic similarity between source and target
- **Duplicate Detection**: Percentage of outputs that appear in training data

### Professional Assessment
- **Gemini Acceptance Rate**: Percentage of translations deemed acceptable by Gemini
- **Expert Validation**: Professional-level translation quality assessment
- **True/False Scoring**: Simple binary classification for translation acceptability

## Customization

### Adding New Reward Components

You can extend the reward system by modifying `translation_rewards.py`:

```python
def custom_reward_component(source_text, target_text):
    # Your custom logic here
    return score

def compute_translation_rewards(prompts, completions, **kwargs):
    # Integrate your custom component
    custom_score = custom_reward_component(source_text, target_text)
    # Combine with existing rewards
    return combined_score
```

### Using Different Language Pairs

Modify the dataset preparation to use different language pairs:

```python
dataset = load_dataset("tatoeba", lang1="en", lang2="de")  # English to German
```

Update the language detection target:
```python
reward_language_detection(text, target_language="de")
```

### Customizing Gemini Evaluation

Modify the evaluation prompt in `full_translation_eval.py` for different criteria:

```python
evaluation_prompt = f"""Evaluate this translation for specific criteria:
- Technical accuracy for domain-specific terms
- Cultural appropriateness for target audience
- Tone consistency with source material

Is the translation acceptable? True/False - [explanation]"""
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU training
2. **Model Download Issues**: Ensure internet connection and HuggingFace access
3. **Langdetect Errors**: The library may struggle with very short texts
4. **Gemini API Errors**: Ensure `GEMINI_API_KEY` is set and you have API access
5. **Evaluation Timeout**: Reduce sample size for initial testing

### Debug Mode

Enable verbose logging by setting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Quick Start Checklist

1. ✅ Install dependencies: `pip install -r translation_requirements.txt`
2. ✅ Set Gemini API key: `export GEMINI_API_KEY="your_key"`
3. ✅ Prepare dataset: `python action.py --prepare_data`
4. ✅ Train model: `python action.py --train`
5. ✅ Evaluate results: `python full_translation_eval.py`

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- **HuggingFace**: For the model hosting, trl and transformers library
- **Tatoeba**: For the translation dataset
- **Qwen Team**: For the embedding model
- **Google**: For the Gemini API enabling professional translation evaluation