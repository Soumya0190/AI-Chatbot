"""
This module provides functions for loading chat-based language models and text summarization models.

💡 Purpose of the Module:
This module is responsible for:
    * Loading a chat model (like gpt2) for generating text in dialogue/chat format.
    * Loading a text summarization model (like Falconsai/text_summarization).
    * Caching models globally to avoid reloading them multiple times.

✅ Key Benefits of This Implementation
Efficient:      Caches models to avoid reloading on each function call.
Safe:           Uses .eval() mode for inference and runs on CPU to reduce GPU dependency.
Flexible:       You can easily switch out model names in one place.
"""

# Typing annotations to clarify input/output types for functions
from typing import Optional, Tuple

# Accelerate is often used to optimize model performance (not directly used here, but kept for compatibility)
import accelerate

# Import required components from Hugging Face Transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Global variables to cache loaded models and tokenizers to avoid redundant loading
chat_model = None  # Stores the loaded chat model
chat_tokenizer = None  # Stores the tokenizer for the chat model
summarize_model = None  # Stores the summarization model pipeline
summarize_tokenizer = None  # (Unused here, placeholder for future use)


def chat_model_tokenizer(
    model_name: str = "gpt2",  # Default model is GPT-2
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a chat-based language model and tokenizer.

    Parameters:
        model_name (str): The name of the pre-trained language model to load.

    Returns:
        Tuple containing the loaded language model and tokenizer.
    """
    global chat_model, chat_tokenizer  # Use global variables to reuse model/tokenizer across calls

    # Only load the model and tokenizer if not already loaded
    if chat_model is None or chat_tokenizer is None:
        # Load the tokenizer from Hugging Face model hub
        chat_tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True  # Allow models with custom code
        )

        # If tokenizer has no padding token, assign end-of-sequence token as a fallback
        if chat_tokenizer.pad_token is None:
            chat_tokenizer.pad_token = chat_tokenizer.eos_token

        # Load the causal language model (chat model) with optimized CPU memory usage
        chat_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",           # Force model to load on CPU
            trust_remote_code=True,     # Allow remote model code (use with caution)
            low_cpu_mem_usage=True,     # Optimizes memory usage
        ).eval().to("cpu")              # Set model to eval mode and move to CPU

    return chat_model, chat_tokenizer   # Return the model and tokenizer


def text_summarization(
    model_name: str = "Falconsai/text_summarization",  # Default summarization model
) -> Tuple[pipeline, None]:
    """
    Load a text summarization model using Hugging Face's pipeline.

    Parameters:
        model_name (str): The name of the pre-trained summarization model.

    Returns:
        Tuple containing the summarization pipeline and None (for tokenizer placeholder).
    """
    global summarize_model, summarize_tokenizer  # Use global variables to cache model

    
    if summarize_model is None:     # Load summarization model if not already loaded
        summarize_model = pipeline(
            "summarization",        # Type of pipeline
            model=model_name,       # Model name from Hugging Face
            device_map="cpu"        # Force execution on CPU
        )
        summarize_tokenizer = None  # Tokenizer is not used explicitly with pipeline

    return summarize_model, summarize_tokenizer  # Return the pipeline and a placeholder


