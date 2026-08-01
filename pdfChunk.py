from loader import load_pdf
from rag import chunk_text, create_embeddings, create_faiss_index, search
from app import ask_llm

text = load_pdf("sample.pdf")

chunks = chunk_text(text)

embeddings = create_embeddings(chunks)

index = create_faiss_index(embeddings)

while True:
    query = input("\nAsk:")

    results = search(query, model, index, chunks)

    context = "\n".join(results)

    answer = ask_llm(query, context)

    print("\nAnswer:", answer)
