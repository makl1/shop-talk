import os, io
import numpy as np
import torch
from PIL import Image

from fastapi import FastAPI, HTTPException, Query, File, UploadFile
from qdrant_client import QdrantClient

from transformers import CLIPModel, CLIPProcessor

import boto3
from botocore.config import Config

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("shop-talk-api")


# ------------------------
# Config
# ------------------------
QDRANT_URL  = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION  = os.getenv("COLLECTION_NAME", "products_clip")

CLIP_NAME   = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
ALPHA       = float(os.getenv("ALPHA", "0.5"))

S3_BUCKET   = os.getenv("S3_BUCKET", "shoptalk-assistant")
AWS_REGION  = os.getenv("AWS_REGION", "us-east-2")
SIGNED_URL_TTL = int(os.getenv("SIGNED_URL_TTL", "900"))  # seconds

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()

client = QdrantClient(url=QDRANT_URL)

clip_model = CLIPModel.from_pretrained(CLIP_NAME).to(DEVICE)
clip_proc  = CLIPProcessor.from_pretrained(CLIP_NAME)
clip_model.eval()

s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    config=Config(signature_version="s3v4")
)

REQS = Counter("http_requests_total", "Total HTTP requests", ["route"])
LAT  = Histogram("http_request_seconds", "Request latency seconds", ["route"])

# ------------------------
# Helpers
# ------------------------s
def _as_tensor(x):
    logger.info("In astensor flow")
    # HF model outputs often have pooler_output
    if hasattr(x, "pooler_output") and isinstance(x.pooler_output, torch.Tensor):
        return x.pooler_output
    # Some outputs allow dict-style access
    if isinstance(x, dict) and "pooler_output" in x:
        return x["pooler_output"]
    return x

def _l2norm(t: torch.Tensor) -> torch.Tensor:
    x = _as_tensor(t)
    logger.info("In l2 norm")
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"_l2norm expected torch.Tensor, got {type(x)}")
    try:
        result = x / x.norm(dim=-1, keepdim=True)
    except Exception as e :
        logger.error(f"Error in l2 norm - {e}")
    return result

def presign_image(image_key: str) -> str:
    logger.info(f"S3Bucket:{not S3_BUCKET}")
    logger.info(f"image_key:{not image_key}")

    if not image_key or not S3_BUCKET:
        return ""
    try:
        result = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": image_key},
            ExpiresIn=SIGNED_URL_TTL,
        )
        logger.info(f"image url {result}")
        return result
    except Exception:
        logger.error(f"image url error")
        return ""

def build_query_vector(q: str | None, image_bytes: bytes | None) -> list[float]:
    text_vec = None
    img_vec  = None

    logger.info(f"Building query vector | text={bool(q)} | image={bool(image_bytes)}")

    try:
        if q and q.strip():
            inputs = clip_proc(text=[q.strip()], return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            with torch.no_grad():
                output = clip_model.get_text_features(**inputs)
            if hasattr(output, 'text_embeds'):
                features = output.text_embeds # Access the tensor
            else:
                features = output

            text_vec = _l2norm(features).squeeze(0).cpu().numpy().astype(np.float32)

        if image_bytes:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            inputs = clip_proc(images=img, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                v = clip_model.get_image_features(**inputs)
                logger.info(f"Image embedding shape: {v.shape}")
            img_vec = _l2norm(v).squeeze(0).cpu().numpy().astype(np.float32)
    except Exception as e:
        logger.error(f"embedding error {e}")
        return ""

    if text_vec is None and img_vec is None:
        raise ValueError("Provide text query or an image.")

    if text_vec is None:
        vec = img_vec
    elif img_vec is None:
        vec = text_vec
    else:
        vec = ALPHA * text_vec + (1.0 - ALPHA) * img_vec

    vec = vec / (np.linalg.norm(vec) + 1e-12)
    return vec.astype(np.float32).tolist()

def format_hits(hits):
    out = []
    for r in hits:
        p = r.payload or {}
        image_key = p.get("image_key", "")
        out.append({
            "score": float(r.score),
            "item_id": p.get("item_id", ""),
            "title": p.get("title", ""),
            "caption": p.get("caption", ""),
            "image_key": image_key,
            "image_url": presign_image(image_key),
        })
    return out

# ------------------------
# Routes
# ------------------------
@app.get("/health")
def health():
    REQS.labels("/health").inc()
    return {
        "ok": True,
        "qdrant_url": QDRANT_URL,
        "collection": COLLECTION,
        "clip_model": CLIP_NAME,
        "alpha": ALPHA,
        "s3_bucket_set": bool(S3_BUCKET),
        "aws_region": AWS_REGION,
    }

@app.get("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

# Text-only (quick testing)
@app.get("/search")
def search_get(q: str = Query(..., min_length=2), k: int = 5):
    route = "/search(GET)"
    REQS.labels(route).inc()
    try:
        with LAT.labels(route).time():
            qv = build_query_vector(q, None)
            logger.info(f"{COLLECTION} , {QDRANT_URL}")
            res = client.query_points(
                collection_name="products_clip",
                query=qv,
                limit=k,
                with_payload=True
            )
            return {"mode": "text", "query": q, "k": k, "results": format_hits(res.points)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Multimodal (text + optional image upload)
@app.post("/search")
async def search_post(q: str = Query("", min_length=0), k: int = 5, image: UploadFile | None = File(default=None)):
    route = "/search(POST)"
    REQS.labels(route).inc()
    logger.info(f"Searching collection={COLLECTION} k={k}")

    try:
        with LAT.labels(route).time():
            image_bytes = await image.read() if image else None
            qv = build_query_vector(q, image_bytes)

            res = client.query_points(
                collection_name=COLLECTION,
                query=qv,
                limit=k,
                with_payload=True
            )
            mode = "text+image" if (q and q.strip() and image_bytes) else ("image" if image_bytes else "text")
            return {"mode": mode, "query": q, "k": k, "results": format_hits(res.points)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
