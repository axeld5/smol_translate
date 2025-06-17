from sentence_transformers import SentenceTransformer
from langdetect import detect
import torch

# Load the Qwen3 embedding model
embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

def detect_language(text):
    """
    Detect the language of the given text using langdetect.
    Returns the detected language code.
    """
    try:
        return detect(text)
    except:
        return "unknown"

def compute_embedding_similarity(source_text, target_text):
    """
    Compute similarity between source and target text using Qwen3 embeddings.
    
    Args:
        source_text (str): Original English text
        target_text (str): Translated French text
    
    Returns:
        float: Similarity score between embeddings
    """
    try:
        # Encode both texts
        source_embedding = embedding_model.encode([source_text], prompt_name="query")
        target_embedding = embedding_model.encode([target_text])
        
        # Compute similarity
        similarity = embedding_model.similarity(source_embedding, target_embedding)
        return float(similarity[0][0]) - 0.5  # Extract scalar value
    except Exception as e:
        print(f"Error computing embedding similarity: {e}")
        return 0.0

def reward_language_detection(text, target_language="fr"):
    """
    Check if the text is in the target language.
    
    Args:
        text (str): Text to check
        target_language (str): Expected language code (default: "fr" for French)
    
    Returns:
        int: -1 if not target language, 0 if target language
    """
    detected_lang = detect_language(text)
    return 0 if detected_lang == target_language else -1

def reward_translation_quality(source_text, target_text, target_language="fr"):
    """
    Combined reward for translation quality.
    
    Args:
        source_text (str): Original English text
        target_text (str): Translated text
        target_language (str): Expected language code
    
    Returns:
        float: Combined reward score
    """
    # Check language first
    lang_reward = reward_language_detection(target_text, target_language)
    
    # If language is incorrect, return -1
    if lang_reward == -1:
        return -1.0
    
    # If language is correct, return embedding similarity score
    embedding_score = compute_embedding_similarity(source_text, target_text)
    return embedding_score

def compute_translation_rewards(prompts, completions, **kwargs):
    """
    Sparse version of translation rewards - only gives positive reward for good translations.
    
    Args:
        prompts: List of prompts from GRPO trainer  
        completions: List of completions from GRPO trainer
    
    Returns:
        List[float]: Reward scores for each completion
    """
    scores = []
    
    for i, completion_batch in enumerate(completions):
        # Extract the source English text from the prompt
        prompt_content = prompts[i][0]["content"]
        
        # Extract English sentence from prompt
        try:
            start_quote = prompt_content.find("'") + 1
            end_quote = prompt_content.rfind("'")
            source_text = prompt_content[start_quote:end_quote]
        except:
            source_text = ""
        
        # Extract the translated text from completion
        target_text = completion_batch[0]["content"].strip()
        
        # Compute reward
        if source_text:
            lang_reward = reward_language_detection(target_text, "fr")
            if lang_reward == -1:
                reward = -1.0
            else:
                embedding_score = compute_embedding_similarity(source_text, target_text)
                # Only give positive reward if embedding score is above threshold
                reward = embedding_score
        else:
            reward = -1.0
        
        scores.append(reward)
    
    return scores 