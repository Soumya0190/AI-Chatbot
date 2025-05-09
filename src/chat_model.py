'''
Line-by-Line Explanation:

-----------------------------------------------------------------------------------------------------------------------------------

Imports:
from typing import ...:                             Provides type hinting for better readability and IDE support.
import torch:                                       Imports PyTorch, which is used to load and run the deep learning models.
from .model_tokenizer import chat_model_tokenizer:  Imports the custom chat_model_tokenizer function to load the chat model and tokenizer.
from .prompt_generator import prompt_with_rag:      Imports the function prompt_with_rag from the prompt_generator module to enrich the user query with external data.

-----------------------------------------------------------------------------------------------------------------------------------

Function Definition chatbot:

def chatbot(user_query: str, history: List, use_google: bool, search_time: str) -> str:: 
    * Defines a function that simulates a chatbot interaction. 
    * It accepts a user's query, conversation history, a flag to enable Google search, and a search time range.

-----------------------------------------------------------------------------------------------------------------------------------

Parameter Conversions:

user_query = str(user_query):   Ensures that the user_query is a string type.
search_time = str(search_time): Ensures the search_time is a string, useful for handling different time-based search queries.

-----------------------------------------------------------------------------------------------------------------------------------

Generating Prompt with External Context:

new_user_query, all_urls = prompt_with_rag(user_query, use_google, search_time): 
* Calls the prompt_with_rag function to generate an enriched query based on: 
    * whether Google search is enabled 
    * and the specified search time.

-----------------------------------------------------------------------------------------------------------------------------------

Error Handling:

except Exception as e: 
    * Catches any error from the prompt_with_rag function 
    * and returns a descriptive error message.

-----------------------------------------------------------------------------------------------------------------------------------

Prompt Engineering:

new_user_query = "You are an expert financial ChatBot...": 
    * Adds a custom instruction to the user query, 
    * ensuring the model understands its role as a financial expert.

-----------------------------------------------------------------------------------------------------------------------------------

Loading Model and Tokenizer:

model, tokenizer = chat_model_tokenizer(): 
    * Loads the pre-trained chatbot model and tokenizer 
    * by calling the chat_model_tokenizer function.

-----------------------------------------------------------------------------------------------------------------------------------

Model Inference (no gradients):

with torch.no_grad(): 
    Ensures that the model doesn't track gradients during inference, saving memory and computation.

-----------------------------------------------------------------------------------------------------------------------------------

Tokenization:

inputs = tokenizer(...): 
    Tokenizes the concatenated user query with appropriate padding, truncation, and EOS token handling.

-----------------------------------------------------------------------------------------------------------------------------------

Model Response Generation:

output = model.generate(...): 
    Generates the chatbot's response using the tokenized input, with a specified max length and attention mask.

-----------------------------------------------------------------------------------------------------------------------------------

Decoding the Output:

response = tokenizer.decode(output[0], skip_special_tokens=True): 
    Decodes the generated token IDs back into a string response, skipping special tokens like padding or EOS.

-----------------------------------------------------------------------------------------------------------------------------------

Return Response with Google References (Optional):

if use_google::                                                         If Google search is enabled, appends the external references to the response.
return "{}\n\nReferences:\n{}".format(response, "\n".join(all_urls)):   Formats and returns the response with the references.
return response:                                                        If no Google search is needed, it just returns the chatbot's response.

-----------------------------------------------------------------------------------------------------------------------------------

Optional Improvements:
GPU Support: If you plan to run the model on GPUs, you might want to add device_map="cuda" or check for available GPUs before loading the model.
Caching: For efficiency, caching the model/tokenizer could be considered (similar to the previous module) to avoid repeated loading.
'''


from typing import Dict, List, Optional, Tuple, Union   # Import type annotations for better clarity on the expected input and output types
import torch                                            # Importing torch for handling the PyTorch-based model (used in deep learning models)
from .model_tokenizer import chat_model_tokenizer       # Import custom function from model_tokenizer.py that loads the chat model and tokenizer
from .prompt_generator import prompt_with_rag           # Import the prompt generator for adding external data (like from Google search) to the user's query


def chatbot(user_query: str, history: List, use_google: bool, search_time: str) -> str:
    """
    Function to simulate a chatbot conversation.

    Parameters:
    - user_query (str): The user's input query to the chatbot.
    - history (List[str]): The conversation history (for context).
    - use_google (bool): Flag to determine if Google search results should be included in the response.
    - search_time (str): The time frame for Google search (e.g., 'past week', 'past month').

    Returns:
    - A string response generated by the chatbot using user query and conversation history.
    """
    
    # Convert user_query and search_time to strings (ensures correct type handling)
    user_query = str(user_query)
    search_time = str(search_time)

    try:
        # Generate the new user query and retrieve any URLs for external context using the RAG method
        new_user_query, all_urls = prompt_with_rag(user_query, use_google, search_time)
    except Exception as e:
        # If any error occurs while generating the prompt, return the error message
        return f"Error generating prompt: {str(e)}"

    # Enhance the user query by adding instructions for the chatbot (financial expert context)
    new_user_query = (
        "You are an expert financial ChatBot, respond to the user message and feel"
        " free to use the extra given online source information during the"
        " conversation, if necessary.\n\n"
        + new_user_query
    )

    model, tokenizer = chat_model_tokenizer()           # Load the chat model and tokenizer from the custom `model_tokenizer` module

    with torch.no_grad():                               # Disable gradient tracking for inference to save memory and computation  
        inputs = tokenizer(                             # Tokenize the user query and conversation history  
            new_user_query + tokenizer.eos_token,       # Append EOS token to signify end of input
            return_tensors="pt",                        # Return tensor format for PyTorch
            padding=True,                               # Pad to the maximum length
            truncation=True                             # Truncate if input exceeds model's maximum length
        )
        
        output = model.generate(                        # Generate a response from the model using the tokenized input
            inputs["input_ids"],                        # Input tokens to the model
            attention_mask=inputs["attention_mask"],    # Mask to tell the model which tokens to attend to
            max_length=1000,                            # Max length of generated response
            pad_token_id=tokenizer.pad_token_id,        # Pad token id for padding
        )
        
        # Decode the model's output back into human-readable text
        response = tokenizer.decode(output[0], skip_special_tokens=True)

    
    if use_google:                                                              # If the flag for using Google search is set, append the references (URLs) to the response
        return "{}\n\nReferences:\n{}".format(response, "\n".join(all_urls))    # Format the response to include URLs as references
    else:
        return response                                                         # If no Google search is needed, return just the response                  
