Here’s a **STAR (Situation, Task, Action, Result)** breakdown describing the chatbot system across all its code files:

---

### **SITUATION**

A financial tech team needed to build an intelligent, context-aware chatbot that could:

* Understand user queries in natural language,
* Retrieve real-time online information,
* Filter, summarize, and embed relevant data into the chatbot's response,
* Use modern NLP and deep learning techniques in a modular, extensible codebase.

The goal was to enhance user decision-making with financial information augmented by real-time data, particularly for regulatory and market research questions.

---

### **TASK**

Develop a fully functional **RAG-based (Retrieval-Augmented Generation)** chatbot pipeline using Python that:

* Integrates with pretrained transformer models (e.g., GPT-2),
* Performs Google searches and scrapes content from webpages,
* Filters and summarizes content,
* Uses semantic similarity to inject high-quality external knowledge into prompts,
* Generates coherent and domain-specific chatbot responses,
* Can operate with or without online information, depending on user preference.

---

### **ACTION**

The solution was implemented across several modular components:

#### ✅ `chat_model.py`

* Served as the **entry point**.
* Built the chatbot pipeline by:

  * Creating prompts from user queries using `prompt_with_rag()`,
  * Tokenizing input,
  * Feeding it to the language model (`GPT-2` by default),
  * Returning generated response along with citations (if Google was used).

#### ✅ `model_tokenizer.py`

* Handled **lazy-loading and caching** of:

  * Chat models (`AutoModelForCausalLM`),
  * Tokenizers,
  * Summarization pipelines (`pipeline` from Hugging Face).
* Ensured low memory usage by offloading models to CPU and setting padding tokens dynamically.

#### ✅ `online_search.py`

* Used **Selenium** for web scraping with headless Chrome.
* Wrapped **Google Search API** and enforced query summarization for long inputs.
* Returned top-n relevant links and extracted page text.
* Applied summarization to long inputs before querying Google (`summarize_text()`).

#### ✅ `prompt_generator.py`

* Controlled **Retrieval-Augmented Prompt Generation**.
* Called `google_search()` and `get_website_text()` to collect content.
* Performed semantic similarity ranking on paragraphs using:

  * A sentence transformer model,
  * Cosine similarity scoring.
* Generated a concise yet informative prompt prepended to the user’s question.

#### ✅ `semantic_search.py`

* Filtered and deduplicated content into paragraphs or sentences.
* Encoded candidate texts and computed cosine similarities.
* Selected **top-N most relevant chunks** of information.
* Ensured all RAG content injected into the prompt was semantically aligned with the user query.

---

### **RESULT**

✅ Delivered a fully functional **modular chatbot system** that:

* Can respond to financial queries with or without external sources.
* Retrieves and embeds live web data into conversations.
* Uses **semantic similarity** to provide only relevant content.
* Produces context-aware responses using **Hugging Face Transformers**.

📈 **Impact:**

* Significantly improved response relevance and accuracy.
* Offered explainable references (URLs).
* Enabled cost-efficient deployment with CPU-optimized models and lazy-loading design.
* Provided a framework extensible to any domain beyond finance.

---

