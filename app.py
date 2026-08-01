from google import generativeai as genai
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer

from loader import  load_multiple_pdfs
from rag import create_documents_embeddings, search, chunk_documents, create_faiss_index_doc
import os

load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

llm = genai.GenerativeModel("gemini-2.5-flash")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

files = ["untitled.pdf", "sample2.pdf"]
#text = load_pdf("untitled.pdf")
documents = load_multiple_pdfs(files)

#chunks = chunk_text(text)
chunks = chunk_documents(documents)

embeddings = create_documents_embeddings(chunks, embedding_model)

index = create_faiss_index_doc(embeddings)

def ask_llm(query, context):

    prompt = f"""
    Answer ONLY from context below:

    Context:
    {context}

    Question:
    {query}
    """

    response = llm.generate_content(prompt)

    return response.text



while True:
    query = input("\nAsk:")

    if query.lower() == "exit":
        break

    results = search(query, embedding_model, index, chunks)

    for r in results:
        print(f"\n[Source: {r['source']} | Page: {r['page']}]")
    
    context = "\n".join([r["text"] for r in results])

    answer = ask_llm(query, context)

    print("\nAnswer:", answer)