import os, csv, glob
from sentence_transformers import SentenceTransformer
import config
from retriever import retrieve, get_collection
from generator import generate

TARGETED_QUERIES = [
    "What is the main topic of your first file?",
    "Describe the key concept in your second file.",
    "What does your third file explain?",
    "Summarize the subject of your fourth file.",
    "What is discussed in your fifth file?",
]

CROSS_QUERIES = [
    "How does photosynthesis relate to the carbon cycle?",
    "What are the biological effects of radiation?",
    "Explain the relationship between DNA and proteins.",
    "How do neural networks mimic the brain?",
    "What connects climate change and ocean currents?",
]

def embed_new_files(col_new):
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    files = glob.glob("data/*.txt")
    print(f"Found {len(files)} new files.")
    for fpath in files:
        with open(fpath) as f:
            text = f.read().strip()
        fname = os.path.basename(fpath)
        emb = model.encode(text).tolist()
        col_new.add(
            ids=[fname],
            embeddings=[emb],
            documents=[text],
            metadatas=[{"title": fname, "url": f"local://{fname}"}],
        )
    print("New files embedded and inserted.")

def retrieve_both(query):
    col_starter = get_collection(config.CHROMA_PATH_STARTER, config.COLLECTION_STARTER)
    col_new     = get_collection(config.CHROMA_PATH_PART2,   config.COLLECTION_PART2)
    chunks_s = retrieve(query, col_starter)
    chunks_n = retrieve(query, col_new)
    for c in chunks_s: c["corpus"] = "starter"
    for c in chunks_n: c["corpus"] = "new"
    combined = sorted(chunks_s + chunks_n, key=lambda x: x["score"])[:config.TOP_K]
    return combined

if __name__ == "__main__":
    col_new = get_collection(config.CHROMA_PATH_PART2, config.COLLECTION_PART2)
    embed_new_files(col_new)

    rows = []
    all_queries = [("targeted", q) for q in TARGETED_QUERIES] + \
                  [("cross",    q) for q in CROSS_QUERIES]

    for qtype, query in all_queries:
        chunks  = retrieve_both(query)
        answer  = generate(query, chunks)
        sources = " | ".join(f"{c['title']} [{c['corpus']}]" for c in chunks)
        scores  = " | ".join(f"{c['score']:.3f}" for c in chunks)
        corpora = " | ".join(c["corpus"] for c in chunks)
        grounded = input(f"\n[{qtype}] {query}\n{answer[:200]}\nGrounded? [y/n] ").strip()
        rows.append([qtype, query, sources, scores, corpora, answer[:150], grounded])

    with open("results_part2.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type","query","sources","scores","corpora","answer_preview","grounded"])
        w.writerows(rows)
    print("Saved results_part2.csv")