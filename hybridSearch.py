"""
simple/smart search problem : Top 10 chunks but Best chunk may be at position 5

Solution
Rerank using deepar scoring 

"""

def keyword_score(query, text):
    query_words = query.lower().split()
    text = text.lower()

    score = 0

    for word in query_words:
        if word in text:
            score += 1

    return score

def hybrid_search(query, embedding_model, index, chunks, k=5):

    query_embedding = embedding_model.encode([query])

    distances, indicies = index.search(
        query_embedding.astype("float32"),
        k * 3
    )

    candidates = [chunks[i] for i in indicies[0]]

    scored = []

    for c in candidates:

        semantic_score = 1 / (1 + distances[0][candidates.index(c)])

        key_score = keyword_score(query, c["text"])

        final_score = semantic_score + (0.5 * key_score)

        scored.append((final_score, c))

    scored.sort(reverse=True, key=lambda x: x[0])

    return [c for _, c in scored[:k]]


def rerank(query, chunks, llm):

    scored = []

    for chunk in chunks:

        prompt = f"""
        Score relevence from 1 to 10.

        Query: {query}

        Chunk: {chunk['text']}
        """

        score = llm.generate_content(prompt).text

        try:
            score = float(score)
        except:
            score = 0

        scored.append((score, chunk))
    
    scored.sort(reverse=True)

    return [c for _, c in scored[:3]]