'''
Line-by-Line Explanation:
-----------------------------------------------------------------------------------------------------------------------------------

Imports:

from typing import List, Tuple:                                 Imports the List and Tuple classes from the typing module to provide type hints for function signatures.
from .online_search import get_website_text, google_search:     Imports the functions get_website_text and google_search from the online_search module. These functions are used to perform online searches and extract text from webpages.
from .semantic_search import semantic_search:                   Imports the semantic_search function from the semantic_search module, which is used to find the most relevant information in the webpage content based on semantic similarity.

Function Definition prompt_with_rag:

def prompt_with_rag(query: str, use_google: bool, search_time: str, num_results: int = 3) -> Tuple[str, List]:: 
    * Defines the function prompt_with_rag that accepts 
        * the user query (query), 
        * a flag to enable Google search (use_google), 
        * a time range for Google search (search_time), 
        * and the maximum number of search results (num_results). 
    * The function returns 
        * a tuple containing the generated prompt 
        * and the list of URLs.

-----------------------------------------------------------------------------------------------------------------------------------

Maximum Prompt Length:

max_prompt_length = 500: 
    * Sets the maximum length for the generated prompt to avoid exceeding the model's input token limit. 
    * This helps ensure the model can handle the entire input.

-----------------------------------------------------------------------------------------------------------------------------------

Initialize Variables:

all_urls = "":      Initializes an empty string to store the URLs of the search results.
answer = "":        Initializes an empty string to store the collected information that will be included in the final answer.
loop_count = 0:     Initializes a counter (loop_count) to track the current iteration over the search results.

-----------------------------------------------------------------------------------------------------------------------------------

Check if Google Search is Enabled:

if use_google:: 
    * Checks if the Google search option is enabled. 
    * If true, the function will perform an online search using Google.

-----------------------------------------------------------------------------------------------------------------------------------

Perform Google Search:

all_urls = google_search(query, search_time): 
* Calls the google_search function 
* with the provided query 
* and search time 
* to retrieve a list of URLs from Google.

-----------------------------------------------------------------------------------------------------------------------------------

Ensure Valid Number of Results:

num_results = min(num_results, len(all_urls)): 
* Ensures that the number of search results considered 
* doesn't exceed the total number of URLs retrieved by Google. 
* This prevents errors when there are fewer results than the specified num_results.

-----------------------------------------------------------------------------------------------------------------------------------

Loop Through Search Results:

while len(answer.split()) < max_prompt_length and loop_count < num_results:: 
    * Starts a loop that runs while 
    * the answer's length is below the maximum prompt length 
    * and there are still search results left to process.

-----------------------------------------------------------------------------------------------------------------------------------

Extract Text from Webpage:

url_in_use = all_urls[loop_count]: 
* Retrieves the URL for the current search result 
* based on the loop count.

top_text = get_website_text(url_in_use): 
* Uses the get_website_text function 
* to extract text content from the webpage 
* at the specified URL.

-----------------------------------------------------------------------------------------------------------------------------------

Perform Semantic Search:

top_answers = semantic_search(...): 
* Uses the semantic_search function 
* to find the most relevant sections of the extracted text 
* based on the user's query. 
* It uses a pre-trained model (all-mpnet-base-v2) 
* to perform the search in "Paragraph" mode, 
* looking for the top 5 most similar answers.

-----------------------------------------------------------------------------------------------------------------------------------

Combine Answers:

answer += f"Information from online source {loop_count + 1}: \n\n" + top_answers + "\n\n": 
    * Appends the semantic search results to the answer string, 
    * including information about the source (i.e., the search result number).

-----------------------------------------------------------------------------------------------------------------------------------

Increment Loop Counter:

loop_count += 1: 
* Increments the loop counter 
* to process the next search result in the next iteration.

-----------------------------------------------------------------------------------------------------------------------------------

Construct the Final Prompt:

prompt = f"User: {query}\n\n" + answer + "\n\n": 
    * Combines the original user query with the gathered information (the answer) 
    * to create the final prompt.

If Google Search is Not Enabled:

else:: 
* If the use_google flag is not set, 
* only the user's query is included in the prompt 
* without any additional information from online sources.

prompt = f"User: {query}\n\n": 
* Constructs the prompt with only the user's query.

-----------------------------------------------------------------------------------------------------------------------------------

Return the Prompt and URLs:

return prompt, all_urls: 
    * Returns a tuple 
        * containing the final prompt 
        * and the list of URLs used for gathering online information.

-----------------------------------------------------------------------------------------------------------------------------------

Optional Improvements:
Error Handling:             You could add exception handling in case any of the external functions (google_search, get_website_text, or semantic_search) fail.
Google Search Limiting:     If Google search is disabled (use_google=False), consider adding an option to still perform semantic search without any external data.
'''


# Import type annotations to provide clarity on function inputs and outputs
from typing import List, Tuple

# Import custom functions for online search and semantic search
from .online_search import get_website_text, google_search
from .semantic_search import semantic_search


"""
    Function to get information from online sources related to the given query using
    either Google search or semantic search.
"""
def prompt_with_rag(
    query: str,             # The user's query to be processed
    use_google: bool,       # Flag to determine if Google search should be used 
    search_time: str,       # The time frame for Google search (e.g., 'past week', 'past month').
    num_results: int = 3    # The maximum number of online search results to consider (default is 3).
) -> Tuple[str, List]:      # Returns a tuple containing the generated prompt and a list of URLs.
    
    # Define a maximum length for the prompt to avoid exceeding the token limit of the model
    max_prompt_length = 500

    # Initialize variables to store URLs and answer content
    all_urls = ""
    answer = ""

    # Initialize loop count for iterating over Google search results
    loop_count = 0

    # Check if Google search is enabled
    if use_google:
        # Perform a Google search using the query and the time frame
        all_urls = google_search(query, search_time)

        # Ensure that the number of search results does not exceed the maximum number of results specified
        num_results = min(num_results, len(all_urls))

        # Loop through the search results, gather information, and build the answer
        while len(answer.split()) < max_prompt_length and loop_count < num_results:
            # Get the URL for the current search result
            url_in_use = all_urls[loop_count]

            # Extract text from the webpage at the current URL
            top_text = get_website_text(url_in_use)

            # Perform semantic search to retrieve the most relevant answers from the webpage content
            top_answers = semantic_search(
                model_name="all-mpnet-base-v2",  # Use a pre-trained model for semantic search
                mode="Paragraph",                # Search mode indicating we want to search by paragraphs
                searching_for=query,             # The original user query to search for
                text=top_text,                   # The extracted text from the webpage to search within
                n_similar_texts=5,               # Number of similar texts to retrieve
            )

            # Append the semantic search results to the final answer
            answer += (
                f"Information from online source {loop_count + 1}: \n\n"
                + top_answers
                + "\n\n"
            )

            # Increment the loop counter to move to the next search result
            loop_count += 1

        # Construct the prompt by combining the user's query and the gathered answer
        prompt = f"User: {query}\n\n" + answer + "\n\n"
    else:
        # If Google search is not enabled, just include the user's query in the prompt
        prompt = f"User: {query}\n\n"

    # Return the generated prompt along with the list of URLs used for information
    return prompt, all_urls
