import os
import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

import boto3
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("COLLECTION_NAME", "products_clip")
S3_BUCKET = os.getenv("S3_BUCKET", "")
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")

CLIP_NAME = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
ALPHA = float(os.getenv("ALPHA", "0.5"))  # 0.5 = equal blend

DATA = Path("data/processed/products_small.csv")
LIMIT = int(os.getenv("LIMIT", "5000"))
BATCH = int(os.getenv("BATCH", "64"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True)

def s3_get_image_bytes(s3, bucket: str, key: str) -> bytes | None:
    if not bucket or not key:
        return None
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return obj["Body"].read()
    except Exception:
        return None

def clip_text_embed(model, processor, text: str) -> np.ndarray:
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
    feats = l2norm(feats)
    return feats.squeeze(0).cpu().numpy().astype(np.float32)

def clip_image_embed(model, processor, image_bytes: bytes) -> np.ndarray | None:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
        feats = l2norm(feats)
        return feats.squeeze(0).cpu().numpy().astype(np.float32)
    except Exception:
        return None

def combine(text_vec: np.ndarray | None, img_vec: np.ndarray | None, alpha: float) -> np.ndarray | None:
    if text_vec is None and img_vec is None:
        return None
    if img_vec is None:
        v = text_vec
    elif text_vec is None:
        v = img_vec
    else:
        v = alpha * text_vec + (1 - alpha) * img_vec
    # normalize after mixing
    v = v / (np.linalg.norm(v) + 1e-12)
    return v.astype(np.float32)

def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Missing {DATA}")

    df = pd.read_csv(DATA, dtype=str).head(LIMIT).fillna("").reset_index(drop=True)

    # Expect columns like: item_id,title_en,caption,image_key (or image_path)
    # If you still have image_path like data/raw/images/..., convert it to S3 key now.
    if "image_key" not in df.columns:
        # Try converting from image_path -> image_key
        # Example: data/raw/images/small/95/x.jpg -> products/images/small/95/x.jpg
        df["image_key"] = df.get("image_local_path", "").str.replace("\\", "/")
        df["image_key"] = df["image_key"].str.replace("data/raw/images/", "products/images/", regex=False)

    # Build the product text used for CLIP text embedding
    def build_text(r):
        parts = []
        if r.get("title_en"): parts.append(r["title_en"])
        if r.get("caption"): parts.append(r["caption"])  # keep if you want; CLIP can use it
        return " . ".join([p for p in parts if p.strip()])[:2000]

    df["clip_text"] = df.apply(build_text, axis=1)

    # Setup clients
    s3 = boto3.client("s3", region_name=AWS_REGION)
    qdrant = QdrantClient(url=QDRANT_URL)

    clip_model = CLIPModel.from_pretrained(CLIP_NAME).to(DEVICE)
    clip_proc = CLIPProcessor.from_pretrained(CLIP_NAME)

    # Create collection
    qdrant.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=qm.VectorParams(size=512, distance=qm.Distance.COSINE),
    )

    points = []
    for i, r in df.iterrows():
        text = r["clip_text"]
        text_vec = clip_text_embed(clip_model, clip_proc, text) if text.strip() else None

        key = r["image_key"].strip()
        img_bytes = s3_get_image_bytes(s3, S3_BUCKET, key) if key else None
        img_vec = clip_image_embed(clip_model, clip_proc, img_bytes) if img_bytes else None

        vec = combine(text_vec, img_vec, ALPHA)
        if vec is None:
            continue

        points.append(
            qm.PointStruct(
                id=int(i),
                vector=vec.tolist(),
                payload={
                    "item_id": r.get("item_id", ""),
                    "title": r.get("title_en", ""),
                    "caption": r.get("caption", ""),
                    "image_key": key,
                },
            )
        )

        if len(points) >= BATCH:
            qdrant.upsert(collection_name=COLLECTION, points=points)
            points = []

    if points:
        qdrant.upsert(collection_name=COLLECTION, points=points)

    print(f" Indexed {len(df):,} products into {COLLECTION} (CLIP multimodal)")

if __name__ == "__main__":
    main()
