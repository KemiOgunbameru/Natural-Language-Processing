from sentence_transformers import SentenceTransformer
import chromadb
import config

_model = SentenceTransformer(config.EMBEDDING_MODEL)

def get_collection(chroma_path, collection_name):
    client = chromadb.PersistentClient(path=chroma_path)
    return client.get_or_create_collection(collection_name)

def retrieve(query, collection, top_k=config.TOP_K):
    embedding = _model.encode(query).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=top_k)
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text":  results["documents"][0][i],
            "title": results["metadatas"][0][i].get("title", ""),
            "url":   results["metadatas"][0][i].get("url", ""),
            "score": results["distances"][0][i],
        })
    return chunks