# Importing necessary modules
import logging              # For logging the execution of the program
import re                   # Regular expression library to handle text parsing
from typing import List     # For type hinting, indicating that a function will return a list

import torch                                            # PyTorch library for tensor operations
from sentence_transformers import SentenceTransformer   # For working with pre-trained sentence models
from torch.nn.functional import cosine_similarity       # For calculating cosine similarity between tensors

# Importing text summarization utility from another module
from .model_tokenizer import text_summarization  # This will be used later for summarizing text

def remove_extra_newlines(string: str) -> str:
    """
    Removes extra newlines from a string.

    Args:
        string (str): The input string.

    Returns:
        str: The string with extra newlines removed.
    """
    return "\n".join(line for line in string.splitlines() if line.strip())
    # Split the string into lines
    # filter out the empty or spaces-only lines
    # and join them back together with newlines

def separate_paragraphs(text: str) -> List[str]:
    """
    Separates a text into unique paragraphs.

    Args:
        text (str): The input text.

    Returns:
        List[str]: A list of unique paragraphs from the input text.
    """
    paragraphs = text.split("\n")               # Split the input text by newlines into a list of paragraphs
    paragraphs = list(set(paragraphs))          # Remove duplicates by converting the list to a set and back
    return [s for s in paragraphs if s != ""]   # Return non-empty paragraphs

def filter_strings_by_word_count(strings: List[str]) -> List[str]:
    """
    Filter a list of strings to remove elements with 22 words or less.
    It's a function to remove short paragraphs.

    Parameters:
    - strings (List[str]): The input list of strings.

    Returns:
    - List[str]: The filtered list of strings.
    """
    min_number_of_words = 22  # Defining the threshold for paragraph length
    filtered_strings = [s for s in strings if len(s.split()) > min_number_of_words]  # Filter out short paragraphs
    return filtered_strings

def get_unique_sentences(text: str) -> List[str]:
    """
    Split the input text into sentences, remove leading empty lines,
    and return a list of unique sentences.

    Args:
        text (str): The input text containing sentences.

    Returns:
        List[str]: A list of unique sentences.
    """
    text = text.lstrip("\n")                         # Remove leading newlines from the text
    sentences = re.split(r"(?<=[.!?])\s+|\n", text)  # Use regex to split the text into sentences at punctuation marks or newline
    unique_sentences = list(set(sentences))          # Remove duplicates by converting to a set
    return unique_sentences                          # Return the unique sentences

def encode_text(text: List[str], model: SentenceTransformer) -> torch.Tensor:
    """
    Encode the given list of texts using the specified model.

    Args:
        text (List[str]): The input texts to be encoded.
        model (SentenceTransformer): The model used for encoding.

    Returns:
        torch.Tensor: The encoded texts as a tensor.
    """
    with torch.no_grad():                   # Disable gradient computation to save memory during inference
        outputs = model.encode(text)        # Encode the list of texts using the model
        outputs = torch.tensor(outputs)     # Convert the output into a PyTorch tensor for further computations
    return outputs                          # Return the encoded tensor

def get_top_n_cosine_similarity_rows(
    tensor1: torch.Tensor, tensor2: torch.Tensor, topk: int
) -> torch.Tensor:
    """
    Get the indices of the top N rows with highest cosine similarity scores
    between two tensors.

    Args:
        tensor1 (torch.Tensor): The first tensor.
        tensor2 (torch.Tensor): The second tensor.
        n (int): The number of top rows to retrieve.

    Returns:
        torch.Tensor: The indices of the top N rows.
    """
    similarity_scores = cosine_similarity(tensor1, tensor2)             # Compute the cosine similarity between the tensors
    top_n_indices = torch.topk(similarity_scores.squeeze(), topk)[1]    # Get the indices of the top 'topk' similarities
    return top_n_indices                                                # Return the indices of the top matches

def semantic_search(
    model_name: str, mode: str, searching_for: str, text: str, n_similar_texts: int = 5
) -> str:
    """
    Perform semantic search by measuring the similarity between the
    query and the text corpus.
    Also summarize each paragraph independently to prevent information loss.

    Args:
        model_name (str): The name of the SentenceTransformer model.
                          "all-mpnet-base-v2" is recommended.
        mode (str): Search for most similar sentences or paragraphs?
        searching_for (str): The text to search for (query).
        text (str): The corpus of text to search in.
        n_similar_texts (int) = number of top search results

    Returns:
        str: The search results as a formatted string, and also summarized.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise use CPU
    model = SentenceTransformer(model_name).to(device)  # Load the SentenceTransformer model onto the specified device

    text = remove_extra_newlines(text)              # Clean the input text by removing extra newlines
    if mode == "Paragraph":                         # If searching by paragraphs
        text = separate_paragraphs(text)            # Split text into unique paragraphs
        text = filter_strings_by_word_count(text)   # Filter out short paragraphs
    elif mode == "Sentence":                        # If searching by sentences
        text = get_unique_sentences(text)           # Split text into unique sentences

    search_and_text = [searching_for]   # Add the query to the list of texts
    search_and_text.extend(text)        # Add the corpus texts to the list

    encoded = encode_text(search_and_text, model)  # Encode the query and corpus text

    encoded_search = encoded[0]  # The encoded query
    encoded_text = encoded[1:]  # The encoded corpus text

    n_similar_texts = min(n_similar_texts, len(text))  # Ensure we don't request more results than available

    index_of_similar = get_top_n_cosine_similarity_rows(
        encoded_search, encoded_text, n_similar_texts
    )  # Get the indices of the most similar texts

    output = ""                         # Initialize the output for the final search results
    old_output = ""                     # Initialize the old output (before summarization)
    for i in range(n_similar_texts):    # Loop through the top N similar texts
        try:
            old_output += text[index_of_similar[i]] + "\n\n"  # Add the original text to the old output
            output += summarize_text(text[index_of_similar[i]]) + "\n\n"  # Summarize the text and add it to the output
        except:
            pass  # If there was an error (e.g., index out of range), continue without adding anything

    # Configuring logging to show debug information
    logging.basicConfig(level=logging.INFO)  # Set up logging to show information level messages

    # Log the search results before and after summarization
    logging.info("=" * 10)
    logging.info("Top paragraphs search results before summarization:\n" + old_output)
    logging.info("=" * 10)
    logging.info("Top paragraphs search results after summarization:\n" + output)
    logging.info("=" * 10)

    return output  # Return the summarized search results

def summarize_text(
    text: str, max_length: int = 230, min_length: int = 15, do_sample: bool = False
) -> str:
    """
    Summarizes the input text using the Hugging Face summarization pipeline.

    Args:
        text (str): The input text to be summarized.
        max_length (int, optional): The maximum length of the summary. Defaults to 230.
        min_length (int, optional): The minimum length of the summary. Defaults to 15.
        do_sample (bool, optional): If True, uses sampling to generate the summary.

    Returns:
        str: The summarized text.
    """
    model, _ = text_summarization()     # Call the external summarization function to get the model
    with torch.no_grad():               # Disable gradient computation
        summary = model(
            text, max_length=max_length, min_length=min_length, do_sample=do_sample
        )                               # Summarize the text using the model
    return summary[0]["summary_text"]   # Return the summarized text
