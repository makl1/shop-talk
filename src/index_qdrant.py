import os
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("COLLECTION_NAME", "products")
MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

DATA = Path("data/processed/products_small.csv")
LIMIT = int(os.getenv("LIMIT", "5000"))       # how many products to index
BATCH = int(os.getenv("BATCH", "128"))        # upsert chunk size

def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Missing {DATA}. Build products_small.csv first.")

    df = pd.read_csv(DATA, dtype=str).head(LIMIT)
    if "item_id" not in df.columns or "doc" not in df.columns:
        raise ValueError("products_small.csv must contain columns: item_id, doc")

    df["doc"] = df["doc"].fillna("").astype(str)

    # 1) Embed
    model = SentenceTransformer(MODEL_NAME)
    vecs = model.encode(
        df["doc"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64
    )
    vecs = np.asarray(vecs, dtype=np.float32)
    dim = vecs.shape[1]

    # 2) Connect to Qdrant
    client = QdrantClient(url=QDRANT_URL)

    # 3) Create/Recreate collection (simple for now)
    #    If you want incremental updates later, we switch to create-if-not-exists + upsert.
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    )

    # 4) Upsert in batches
    points = []
    for i, row in df.iterrows():
        points.append(
            qm.PointStruct(
                id=int(i),                  # stable id
                vector=vecs[i].tolist(),
                payload={
                    "item_id": row["item_id"],
                    "title": row.get("title_en", ""),
                    "caption": row.get("caption", ""),
                    "image_path": row.get("image_local_path", ""),
                },
            )
        )

        if len(points) >= BATCH:
            client.upsert(collection_name=COLLECTION, points=points)
            points = []

    if points:
        client.upsert(collection_name=COLLECTION, points=points)

    print(f"   Upserted {len(df):,} products into Qdrant collection '{COLLECTION}'")
    print(f"   Qdrant: {QDRANT_URL}")
    print(f"   Model:  {MODEL_NAME}")
    print(f"   Dim:    {dim}")

if __name__ == "__main__":
    main()
