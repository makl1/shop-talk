import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss

def main():
    ds = pd.read_json("/listings/metadata/listings_0.json.gz", lines=True)
    def find_us_tags(x):
        us_texts = [item["value"] for item in x if item["language_tag"]== "en-US"]
        return us_texts[0] if us_texts else None
    ds = ds.assign(item_name_in_en_us = ds.item_name.apply(find_us_tags))
    ds = ds[~ds.item_name_in_en_us.isna()][["item_id", "item_name_in_en_us", "main_image_id"]]
    print(f"#products with US English title: {len(meta)}")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    emb = np.array(emb,dtype="float32")

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    print("\nReady. Type a query (or 'exit'):\n")

    while True:
        q = input("Query>").strip()
        if q.lower() in {"exit","quit"}:
            break

        q_emb = model.encode([q], normalize_embeddings=True)
        q_emb = np.array(q_emb,dtype="float32")

        k=5
        scores,neighbors = index.search(q_emb,k)

        print("\nTop 5 results:")
        for rank, (idx, score) in enumerate(zip(neighbors[0], scores[0]), start=1):
            snippet = texts[idx].replace("\n", " ")
            snippet = (snippet[:160] + "...") if len(snippet) > 160 else snippet
            print(f"{rank}. id={ids[idx]} score={score:.3f} | {snippet}")
        print()

if __name__ == "__main__":
    main()
