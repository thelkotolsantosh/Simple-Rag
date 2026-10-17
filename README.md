# Simple RAG Python Project

This is a small Python project that demonstrates the most important part of a **RAG (Retrieval-Augmented Generation)** system: semantic retrieval.

The program:

1. Reads information from `knowledge.txt`
2. Converts every paragraph into an embedding
3. Stores those embeddings inside a FAISS vector index
4. Converts the user's question into an embedding
5. Finds the most relevant information
6. Prints the best matching knowledge chunks

This is intentionally simple so beginners can understand the basic RAG workflow.

---

## Project Files

```text
simple-rag-python/
│
├── app.py
├── knowledge.txt
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

## Technologies Used

- Python
- Sentence Transformers
- FAISS
- NumPy
- Hugging Face embedding model

Embedding model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

---

## How the Program Works

```text
knowledge.txt
     |
     v
Sentence Transformer
     |
     v
Document Embeddings
     |
     v
FAISS Vector Index
     |
     v
User Question
     |
     v
Question Embedding
     |
     v
Similarity Search
     |
     v
Top Matching Information
```

---

## Installation

Make sure Python 3.10 or newer is installed.

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/simple-rag-python.git
```

Open the project:

```bash
cd simple-rag-python
```

Create a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the packages:

```bash
pip install -r requirements.txt
```

---

## Run the Program

```bash
python app.py
```

You should see:

```text
SIMPLE RAG RETRIEVAL DEMO

Loading knowledge base...
Loaded 12 knowledge chunks.
Loading embedding model...

RAG search is ready.
```

Then ask a question:

```text
Your question: What is RAG?
```

The program will return the paragraphs that are most related to the question.

---

## Example Questions

```text
What is RAG?
```

```text
What are embeddings?
```

```text
What is FAISS?
```

```text
Why do we use chunking?
```

```text
Where can RAG be used?
```

---

## Add Your Own Knowledge

Open:

```text
knowledge.txt
```

Delete the sample information and add your own information.

Separate each knowledge section with a blank line.

Example:

```text
Python is a programming language used for automation, AI, web development, and data science.

Flask is a lightweight Python framework used to create web applications.

FastAPI is a Python framework commonly used to create high-performance APIs.
```

Run the program again:

```bash
python app.py
```

The application will automatically create new embeddings from your text.

---

## Important Note

The first time the program runs, it downloads the embedding model from Hugging Face.

After the model is downloaded, it can be loaded from your local cache.

---

## Is This Really RAG?

This repository demonstrates the **retrieval** portion of a RAG application.

A complete RAG application normally works like this:

```text
User Question
     |
     v
Retriever
     |
     v
Relevant Context
     |
     v
LLM
     |
     v
Final Answer
```

This project stops after retrieving the relevant context so the code remains easy to understand.

Later, you can connect the retrieved context to:

- OpenAI
- Ollama
- Llama
- Mistral
- Gemini
- Claude
- Hugging Face Transformers

---

## Why This Project Is Useful

It demonstrates that you understand:

- RAG basics
- embeddings
- semantic search
- vector databases
- Sentence Transformers
- FAISS
- Python AI development

It can be used as a small GitHub portfolio project.

---

## License

MIT License.
