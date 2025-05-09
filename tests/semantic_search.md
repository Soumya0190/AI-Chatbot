### Overview of the `semantic_search.py` Script:

This script performs **semantic search** using the **SentenceTransformer** model to compute semantic similarity between a query and a corpus of text. It provides a method for finding and summarizing the most relevant text snippets (either sentences or paragraphs) based on their similarity to the query.

### Detailed Explanation of Each Function and Code:

1. **Imports and Dependencies:**

   * `logging`: Used for logging information and errors.
   * `re`: Regular expressions for string manipulation.
   * `torch`: A deep learning framework used for tensor operations, especially important for the SentenceTransformer.
   * `sentence_transformers`: The SentenceTransformer library to encode texts into embeddings (vector representations).
   * `text_summarization`: Custom function to summarize text (imported from another module).

2. **remove\_extra\_newlines**:

   ```python
   def remove_extra_newlines(string: str) -> str:
   ```

   * **Purpose**: Removes extra newlines from the input string, ensuring there are no unnecessary empty lines.
   * **Returns**: The cleaned string without redundant newlines.

3. **separate\_paragraphs**:

   ```python
   def separate_paragraphs(text: str) -> List[str]:
   ```

   * **Purpose**: Splits the input text into individual paragraphs by separating at newlines and ensures that the paragraphs are unique.
   * **Returns**: A list of unique paragraphs from the input text.

4. **filter\_strings\_by\_word\_count**:

   ```python
   def filter_strings_by_word_count(strings: List[str]) -> List[str]:
   ```

   * **Purpose**: Filters out short strings or paragraphs, keeping only those with more than 22 words.
   * **Returns**: A filtered list of strings/paragraphs that contain more than 22 words.

5. **get\_unique\_sentences**:

   ```python
   def get_unique_sentences(text: str) -> List[str]:
   ```

   * **Purpose**: Splits the text into sentences and removes any duplicate sentences. It uses regular expressions to separate sentences based on punctuation marks.
   * **Returns**: A list of unique sentences.

6. **encode\_text**:

   ```python
   def encode_text(text: List[str], model: SentenceTransformer) -> torch.Tensor:
   ```

   * **Purpose**: Encodes the input text (a list of strings) using the `SentenceTransformer` model. It converts the text into embeddings (vector representations).
   * **Returns**: A tensor of embeddings for each input text.

7. **get\_top\_n\_cosine\_similarity\_rows**:

   ```python
   def get_top_n_cosine_similarity_rows(tensor1: torch.Tensor, tensor2: torch.Tensor, topk: int) -> torch.Tensor:
   ```

   * **Purpose**: Computes the cosine similarity between two sets of encoded text tensors (`tensor1` and `tensor2`) and retrieves the indices of the top `k` most similar rows.
   * **Returns**: The indices of the top `k` rows that have the highest cosine similarity.

8. **semantic\_search**:

   ```python
   def semantic_search(model_name: str, mode: str, searching_for: str, text: str, n_similar_texts: int = 5) -> str:
   ```

   * **Purpose**: This is the main function for performing semantic search. It takes a query (`searching_for`) and compares it against a given corpus of text (`text`) to find the most similar sentences or paragraphs.

     * **Model selection**: The model used for encoding is specified via `model_name` (e.g., "all-mpnet-base-v2").
     * **Mode selection**: The `mode` argument determines whether to search by "Sentence" or "Paragraph". For paragraphs, it further filters by word count.
     * **Summarization**: After finding the most relevant paragraphs/sentences, each one is summarized independently to prevent information loss.
     * **Cosine similarity**: The function uses cosine similarity to find the most similar text segments.
     * **Logging**: Logs the results before and after summarization.
   * **Returns**: A string containing the search results, with each relevant text snippet summarized.

9. **summarize\_text**:

   ```python
   def summarize_text(text: str, max_length: int = 230, min_length: int = 15, do_sample: bool = False) -> str:
   ```

   * **Purpose**: Summarizes the input text using a pre-trained summarization model (likely from Hugging Face's transformers).
   * **Parameters**:

     * `max_length`: Maximum length of the summary.
     * `min_length`: Minimum length of the summary.
     * `do_sample`: If `True`, sampling will be used when generating the summary.
   * **Returns**: A summarized version of the input text.

### How Semantic Search Works:

* **Text Preprocessing**:

  * The input text is preprocessed to remove unnecessary newlines and then split into either sentences or paragraphs.

* **Encoding**:

  * Both the query and the text corpus are encoded into vector representations (embeddings) using a `SentenceTransformer` model.

* **Cosine Similarity**:

  * The cosine similarity between the query embedding and each text corpus embedding is calculated to find the most similar sentences/paragraphs.

* **Summarization**:

  * Each of the top similar sentences or paragraphs is summarized to extract the most important information.

* **Results**:

  * The final output is a summary of the most relevant sentences or paragraphs, based on their semantic similarity to the query.

### Logging:

Throughout the process, logging is used to track the search results before and after summarization, helping to monitor the process and debug if necessary.

### Conclusion:

This script enables semantic search by comparing the similarity between a query and a given text. It uses embeddings generated by the `SentenceTransformer` model, calculates cosine similarities, and selects the most relevant sections. Additionally, it summarizes these sections to present concise information to the user.

### Overview of `semantic_search.py`

This script implements a **semantic search** functionality, leveraging deep learning to measure the similarity between a query and a given text corpus. It uses models from the **SentenceTransformer** library to encode and compare text data efficiently. The script also includes functions for text pre-processing, sentence/paragraph extraction, and summarization to enhance the search process.

Let's go through the key parts of the script and explain their functionality.

---

### Imports and Setup

```python
import logging
import re
from typing import List

import torch
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity

from .model_tokenizer import text_summarization
```

* **logging**: Logs information to track the execution of the script.
* **re**: Regular expressions are used to split the text into sentences.
* **torch**: PyTorch for working with tensors and machine learning models.
* **SentenceTransformer**: Used to create sentence embeddings, a necessary step for semantic search.
* **cosine\_similarity**: Measures the similarity between two vectors.
* **text\_summarization**: A custom module for text summarization (not detailed in the provided script).

---

### Helper Functions

1. **`remove_extra_newlines`**:

   ```python
   def remove_extra_newlines(string: str) -> str:
       return "\n".join(line for line in string.splitlines() if line.strip())
   ```

   This function removes any redundant newlines in a string, leaving only the necessary ones.

2. **`separate_paragraphs`**:

   ```python
   def separate_paragraphs(text: str) -> List[str]:
       paragraphs = text.split("\n")
       paragraphs = list(set(paragraphs))
       return [s for s in paragraphs if s != ""]
   ```

   Splits the input text into paragraphs and removes duplicates. It returns a list of unique non-empty paragraphs.

3. **`filter_strings_by_word_count`**:

   ```python
   def filter_strings_by_word_count(strings: List[str]) -> List[str]:
       min_number_of_words = 22
       filtered_strings = [s for s in strings if len(s.split()) > min_number_of_words]
       return filtered_strings
   ```

   Filters out paragraphs or strings with fewer than 22 words, ensuring that only meaningful content is processed.

4. **`get_unique_sentences`**:

   ```python
   def get_unique_sentences(text: str) -> List[str]:
       text = text.lstrip("\n")  # Remove leading empty lines
       sentences = re.split(r"(?<=[.!?])\s+|\n", text)
       unique_sentences = list(set(sentences))
       return unique_sentences
   ```

   Splits the input text into sentences and returns a list of unique sentences, removing leading empty lines.

---

### Core Functions

1. **`encode_text`**:

   ```python
   def encode_text(text: List[str], model: SentenceTransformer) -> torch.Tensor:
       with torch.no_grad():
           outputs = model.encode(text)
           outputs = torch.tensor(outputs)
       return outputs
   ```

   This function encodes the input list of texts (either sentences or paragraphs) into vectors using a pre-trained model from `SentenceTransformer`. It converts the encoded outputs into a tensor.

2. **`get_top_n_cosine_similarity_rows`**:

   ```python
   def get_top_n_cosine_similarity_rows(tensor1: torch.Tensor, tensor2: torch.Tensor, topk: int) -> torch.Tensor:
       similarity_scores = cosine_similarity(tensor1, tensor2)
       top_n_indices = torch.topk(similarity_scores.squeeze(), topk)[1]
       return top_n_indices
   ```

   Computes the cosine similarity between two sets of vectors (the query and the text corpus). It returns the indices of the top `n` most similar rows based on cosine similarity.

---

### The `semantic_search` Function

This is the core function that performs the semantic search:

```python
def semantic_search(
    model_name: str, mode: str, searching_for: str, text: str, n_similar_texts: int = 5
) -> str:
```

#### Parameters:

* `model_name`: The name of the pre-trained SentenceTransformer model (e.g., "all-mpnet-base-v2").
* `mode`: Determines whether the search is for most similar sentences or paragraphs.
* `searching_for`: The query or text to search for.
* `text`: The corpus of text to search in.
* `n_similar_texts`: The number of similar results to return.

#### Steps:

1. **Select Device**: Choose CUDA (GPU) or CPU for processing.

   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = SentenceTransformer(model_name).to(device)
   ```

2. **Text Pre-processing**: The input text is cleaned and split into sentences or paragraphs based on the `mode`.

   ```python
   text = remove_extra_newlines(text)
   if mode == "Paragraph":
       text = separate_paragraphs(text)
       text = filter_strings_by_word_count(text)
   elif mode == "Sentence":
       text = get_unique_sentences(text)
   ```

3. **Encoding**: The query and text corpus are encoded into vectors using the pre-trained model.

   ```python
   search_and_text = [searching_for]
   search_and_text.extend(text)
   encoded = encode_text(search_and_text, model)
   ```

4. **Cosine Similarity**: The function calculates the cosine similarity between the query and the corpus text, retrieving the top `n` most similar sentences/paragraphs.

   ```python
   index_of_similar = get_top_n_cosine_similarity_rows(encoded_search, encoded_text, n_similar_texts)
   ```

5. **Summarization**: Each top result is summarized to prevent information overload and retain the most relevant content.

   ```python
   output = ""
   for i in range(n_similar_texts):
       output += summarize_text(text[index_of_similar[i]]) + "\n\n"
   ```

6. **Logging**: The script logs the results before and after summarization for debugging and information purposes.

   ```python
   logging.info("=" * 10)
   logging.info("Top paragraphs search results before summarization:\n" + old_output)
   logging.info("=" * 10)
   logging.info("Top paragraphs search results after summarization:\n" + output)
   logging.info("=" * 10)
   ```

---

### **Summarizing Function**

1. **`summarize_text`**:

   ```python
   def summarize_text(
       text: str, max_length: int = 230, min_length: int = 15, do_sample: bool = False
   ) -> str:
       model, _ = text_summarization()
       with torch.no_grad():
           summary = model(
               text, max_length=max_length, min_length=min_length, do_sample=do_sample
           )
       return summary[0]["summary_text"]
   ```

   Summarizes the text using a Hugging Face model for text summarization. It allows configuration of the summary length and whether sampling is used.

---

### Conclusion

This script provides a **semantic search** pipeline that:

* Encodes a query and a text corpus.
* Measures their similarity using **cosine similarity**.
* Retrieves the most similar text segments.
* Summarizes the results for better readability.

It uses **SentenceTransformer** for text encoding, **PyTorch** for tensor operations, and logging for debugging purposes. The script's primary function is to improve the relevance of search results by leveraging semantic understanding of the query and corpus.
