from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb
import csv
import config
from retriever import retrieve, get_collection
from generator import generate

QUERIES = [
    "What is photosynthesis?",
    "How does the immune system fight viruses?",
    "What causes earthquakes?",
    "Who invented the telephone?",
    "How do computers store data?",
    "What is the water cycle?",
    "How do vaccines work?",
    "What is the theory of evolution?",
    "How does a black hole form?",
    "What is the speed of light?",
]

def build_db():
    print("Loading dataset...")
    ds = load_dataset(
        "wikimedia/wikipedia", "20231101.simple",
        split="train", streaming=True
    )
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=config.CHROMA_PATH_STARTER)
    col = client.get_or_create_collection(config.COLLECTION_STARTER)

    items, texts = [], []
    for i, row in enumerate(ds):
        passage = row["text"][:400].strip()
        items.append({"title": row["title"], "url": row["url"], "text": passage})
        texts.append(passage)
        if len(items) >= 5000:
            break

    print(f"Embedding {len(items)} passages...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True).tolist()
    col.add(
        ids=[str(i) for i in range(len(items))],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"title": r["title"], "url": r["url"]} for r in items],
    )
    print("DB ready.")
    return col

if __name__ == "__main__":
    col = build_db()
    rows = []
    for qid, query in enumerate(QUERIES, 1):
        chunks = retrieve(query, col)
        answer = generate(query, chunks)
        sources = " | ".join(f"{c['title']} ({c['url']})" for c in chunks)
        scores  = " | ".join(f"{c['score']:.3f}" for c in chunks)
        grounded = input(f"\nQ{qid}: {query}\nAnswer: {answer[:200]}\nGrounded? [y/n] ").strip()
        rows.append([qid, query, sources, scores, answer[:150], grounded])

    with open("results_part1.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id","query","top_k_sources","scores","answer_preview","grounded"])
        w.writerows(rows)
    print("Saved results_part1.csv")