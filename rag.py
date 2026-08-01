import faiss
import numpy as np



def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

def create_embeddings(chunks, embedding_model):
    return embedding_model.encode(chunks)


def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    return index 

def search(query, embedding_model, index, chunks, k=3):
    query_embedding = embedding_model.encode([query])

    distances, indices = index.search(
        np.array(query_embedding).astype("float32"),
        k
    )

    return [chunks[i] for i in indices[0]]



def chunk_documents(documents, chunk_size=500, overlap=100):
    chunks = []

    for doc in documents:
        text = doc["text"]
        start = 0

        while start < len(text):
            end = start + chunk_size

            chunks_text = text[start:end]

            chunks.append({
                "text": chunks_text,
                "source": doc["source"],
                "page": doc["page"]
            })
            start += chunk_size - overlap

    return chunks

def create_documents_embeddings(chunks, embedding_model):
    texts = [c["text"] for c in chunks]
    return embedding_model.encode(texts)


def create_faiss_index_doc(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    return index 

def search_doc(query, embedding_model, index, chunks, k=3):
    query_embedding = embedding_model.encode([query])

    distances, indices = index.search(
        np.array(query_embedding).astype("float32"),
        k
    )

    return [chunks[i] for i in indices[0]]