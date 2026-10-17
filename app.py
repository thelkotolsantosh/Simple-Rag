from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DATA_FILE = "knowledge.txt"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_knowledge(file_path: str) -> list[str]:
    """
    Load simple paragraphs from a text file.
    Each non-empty paragraph becomes one searchable document.
    """
    text = Path(file_path).read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    if not paragraphs:
        raise ValueError("knowledge.txt is empty.")

    return paragraphs


def build_index(documents: list[str]):
    """
    Convert documents into embeddings and store them in FAISS.
    """
    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        documents,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    dimension = embeddings.shape[1]

    # Inner Product works like cosine similarity
    # because the embeddings are normalized.
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return model, index


def search(question: str, model, index, documents: list[str], top_k: int = 3):
    """
    Search for the most relevant documents.
    """
    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        results.append(
            {
                "text": documents[idx],
                "score": float(score),
            }
        )

    return results


def main():
    print("=" * 60)
    print("SIMPLE RAG RETRIEVAL DEMO")
    print("=" * 60)

    print("\nLoading knowledge base...")

    documents = load_knowledge(DATA_FILE)

    print(f"Loaded {len(documents)} knowledge chunks.")

    print("Loading embedding model...")
    model, index = build_index(documents)

    print("\nRAG search is ready.")
    print("Ask a question about AI, LLMs, embeddings, or RAG.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Your question: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        results = search(
            question=question,
            model=model,
            index=index,
            documents=documents,
            top_k=3,
        )

        print("\nMost relevant information:\n")

        for number, result in enumerate(results, start=1):
            print(f"[{number}] Similarity score: {result['score']:.4f}")
            print(result["text"])
            print()

        print("-" * 60)


if __name__ == "__main__":
    main()
