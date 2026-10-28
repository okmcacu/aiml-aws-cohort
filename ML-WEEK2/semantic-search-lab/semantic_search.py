import pandas as pd
import chromadb


def load_faqs(path):
    """Load the raw FAQ CSV into a DataFrame."""
    return pd.read_csv(path)


def clean_faqs(df):
    """Strip whitespace, drop empty rows and duplicates."""
    df = df.copy()

    for col in ["question", "answer"]:
        df[col] = (
            df[col]
            .astype("string")
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

    df = df.dropna(subset=["question", "answer"])
    df = df[(df["question"] != "") & (df["answer"] != "")]
    df = df.drop_duplicates(subset=["question"]).reset_index(drop=True)
    df["faq_id"] = df.index
    return df


def chunk_text(text, chunk_size=40, overlap=10):
    """Split text into overlapping chunks of words."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_size])
        chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


def build_chunks(df, chunk_size=40, overlap=10):
    """Turn each FAQ into one or more chunks, keeping the question as context."""
    ids, documents, metadatas = [], [], []
    for _, row in df.iterrows():
        pieces = chunk_text(row["answer"], chunk_size, overlap)
        for i, piece in enumerate(pieces):
            ids.append(f"faq{row['faq_id']}_chunk{i}")
            documents.append(f"Q: {row['question']} A: {piece}")
            metadatas.append({
                "faq_id": int(row["faq_id"]),
                "question": str(row["question"]),
                "chunk_index": i,
            })
    return ids, documents, metadatas


def build_index(ids, documents, metadatas, path="chroma_db", name="faqs"):
    """Embed the chunks and store them in a fresh Chroma collection."""
    client = chromadb.PersistentClient(path=path)

    # Start clean so old chunks from a previous run (e.g. a different
    # chunk_size) do not stay in the index.
    try:
        client.delete_collection(name)
    except Exception:
        pass

    collection = client.get_or_create_collection(name=name)
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return collection


def search(collection, query, top_k=3):
    """Return the top_k closest chunks for a query."""
    results = collection.query(query_texts=[query], n_results=top_k)
    matches = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        matches.append({"document": doc, "question": meta["question"], "distance": dist})
    return matches


def print_matches(query, matches):
    print(f"\nQuery: {query}")
    for rank, m in enumerate(matches, start=1):
        print(f"  {rank}. (distance {m['distance']:.3f}) {m['document']}")


def main():
    raw = load_faqs("data/faqs.csv")
    clean = clean_faqs(raw)
    ids, documents, metadatas = build_chunks(clean)
    collection = build_index(ids, documents, metadatas)
    print(f"Raw rows: {len(raw)} | Clean rows: {len(clean)} | Chunks stored: {len(ids)}")


    queries = [
        "I twisted my foot playing football and it is swollen",
        "I keep vomiting and have stomach cramps after eating",
        "someone collapsed and is not breathing",
        "how do I avoid mosquito bites",
    ]
    
    for q in queries:
        print_matches(q, search(collection, q))


if __name__ == "__main__":
    main()
