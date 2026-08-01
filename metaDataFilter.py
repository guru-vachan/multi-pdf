from rag import create_faiss_index

"""
Design
In simple search :

Query -> Search All Chunks -> Top - k -> LLM 
result => LLM confused  -> hallucination

In Smart Search:

Query -> Query Classification -> MetaData Filter ->  Filtered Retrival  -> Top-k -> LLM 

        OR

Query -> Intent Detection(LLM) -> if multi intent then split  -> MetaData Filter for each intent ->  

    Filtered Retrival -> Merge  -> Top-k -> LLM 


result => Target search  -> Accurate Result
"""

'''
In Real world we can use llm for query classification
'''

def classify_query(query):
    query = query.lower()

    if "leave" in query:
        return "HR"
    elif "api" in query:
        return "engineering"
    else:
        return "general"
    

def filter_chunks(chunks, department):
    return [c for c in chunks if c["department"] == department]


def smart_search(query, embedding_model, index, chunks):

    dept = classify_query(query)

    filtered_chunks = filter_chunks(chunks, dept)

    texts = [c["text"] for c in filtered_chunks]

    embedding = embedding_model.encode(texts)

    temp_index = create_faiss_index(embedding) 

    query_embedding = embedding_model.encode([query])

    distances, indicies = temp_index.search(
        np.array(query_embedding).astype("float32"),
        3
    )

    return [filtered_chunks[i] for i in indicies[0]]