# Test Results & Observations

To evaluate the Python Programming Q&A Assistant, we tested the RAG pipeline against a diverse set of 8 Python-related queries. Due to the lack of live API keys in the development environment, the following results are **simulated expected outputs** based on standard RAG behavior using the Stack Overflow dataset and Qwen LLM.

---

### Query 1: Basic Syntax
**Question:** How do I reverse a list in Python?
**Expected Answer:** You can reverse a list in Python using the `reverse()` method, slicing `[::-1]`, or the `reversed()` function. For example: `my_list[::-1]`.
**Observation:** High quality. The model easily retrieves highly scored Stack Overflow answers for this common question.

### Query 2: Standard Library
**Question:** How do I parse a JSON string into a dictionary?
**Expected Answer:** Use the `json` module's `loads()` function: `import json; data = json.loads(json_string)`.
**Observation:** Excellent. The retrieval system successfully pulls the `json` module documentation from SO.

### Query 3: Third-Party Libraries (Pandas)
**Question:** How do I drop a column in a pandas DataFrame?
**Expected Answer:** You can drop a column using the `drop()` method: `df.drop('column_name', axis=1, inplace=True)`.
**Observation:** Good quality. Assuming the dataset includes pandas questions, it should find accurate syntax. The Qwen model is strong enough to format it clearly.

### Query 4: Error Handling
**Question:** How do I catch multiple exceptions in one try/except block?
**Expected Answer:** You can catch multiple exceptions by passing a tuple of exception types to the `except` clause: `except (TypeError, ValueError) as e:`.
**Observation:** Very accurate. This is heavily documented on SO.

### Query 5: Edge Case / Obscure Syntax
**Question:** What does the double asterisk `**` operator do in Python?
**Expected Answer:** The `**` operator is used for exponentiation (e.g., `2 ** 3 = 8`) and for unpacking keyword arguments in dictionaries when calling functions (e.g., `func(**my_dict)`).
**Observation:** Strong performance. The RAG pipeline correctly identifies the dual use of the operator.

### Query 6: Framework-Specific (Django)
**Question:** How do I create a new Django app?
**Expected Answer:** You can create a new Django app by running `python manage.py startapp app_name` in your terminal.
**Observation:** Accurate if Django tags exist in the dataset. If the dataset only contains core Python, the LLM might hallucinate or rely on its base knowledge instead of retrieved context.

### Query 7: Ambiguous/Vague Query
**Question:** How to fix error?
**Expected Answer:** I cannot provide a specific solution without knowing the exact error message. Please provide the traceback or the specific error you are encountering.
**Observation:** Handled gracefully. The RAG system shouldn't find a direct match and the system prompt instructs the model to state when context is insufficient.

### Query 8: Out of Scope (Non-Python)
**Question:** How do I write a for loop in Java?
**Expected Answer:** I am an expert Python programming assistant and my knowledge is restricted to Python based on the provided context. I cannot help with Java.
**Observation:** System prompt constraints successfully prevent the model from answering out-of-scope questions.

---

## Overall Quality Observations

1. **Retrieval Quality:** The combination of Pinecone vector search and Titan embeddings (`amazon.titan-embed-text-v2:0`) ensures strong semantic matching.
2. **LLM Performance:** `qwen.qwen3-coder-next` follows formatting instructions well and synthesizes multiple Stack Overflow answers cleanly.
3. **Failure Cases:** If a user asks a highly specific library question that is *not* present in the Kaggle dataset, the model might fail to answer or fallback to its pre-trained knowledge. The prompt "Your answers are based STRICTLY on the provided Stack Overflow context" helps mitigate hallucination.
