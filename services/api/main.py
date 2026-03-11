import os, io
import logging
import numpy as np
import torch
from PIL import Image

from fastapi import FastAPI, HTTPException, Query, File, UploadFile, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
from transformers import CLIPModel, CLIPProcessor

import boto3
from botocore.config import Config

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("shop-talk-api")

# ------------------------
# Config
# ------------------------
QDRANT_URL     = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION     = os.getenv("COLLECTION_NAME", "products_clip")
CLIP_NAME      = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
ALPHA          = float(os.getenv("ALPHA", "0.5"))
S3_BUCKET      = os.getenv("S3_BUCKET", "shoptalk-assistant")
AWS_REGION     = os.getenv("AWS_REGION", "us-east-2")
SIGNED_URL_TTL = int(os.getenv("SIGNED_URL_TTL", "900"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",")

MAX_IMAGE_BYTES   = 10 * 1024 * 1024  # 10 MB
MAX_QUERY_LENGTH  = 500
ALLOWED_IMG_TYPES = {"image/jpeg", "image/png", "image/webp"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# App / middleware
# ------------------------
limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ------------------------
# Clients / model
# ------------------------
client = QdrantClient(url=QDRANT_URL)

clip_model = CLIPModel.from_pretrained(CLIP_NAME).to(DEVICE)
clip_proc  = CLIPProcessor.from_pretrained(CLIP_NAME)
clip_model.eval()

s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    config=Config(signature_version="s3v4"),
)

REQS = Counter("http_requests_total", "Total HTTP requests", ["route"])
LAT  = Histogram("http_request_seconds", "Request latency seconds", ["route"])

# ------------------------
# Helpers
# ------------------------
def _as_tensor(x):
    if hasattr(x, "pooler_output") and isinstance(x.pooler_output, torch.Tensor):
        return x.pooler_output
    if isinstance(x, dict) and "pooler_output" in x:
        return x["pooler_output"]
    return x

def _l2norm(t: torch.Tensor) -> torch.Tensor:
    x = _as_tensor(t)
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"_l2norm expected torch.Tensor, got {type(x)}")
    return x / x.norm(dim=-1, keepdim=True)

def presign_image(image_key: str) -> str:
    if not image_key or not S3_BUCKET:
        return ""
    try:
        return s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": image_key},
            ExpiresIn=SIGNED_URL_TTL,
        )
    except Exception:
        logger.error("Failed to generate presigned URL for key: %s", image_key)
        return ""

def build_query_vector(q: str | None, image_bytes: bytes | None, alpha: float = ALPHA) -> list[float]:
    text_vec = None
    img_vec  = None

    if q and q.strip():
        inputs = clip_proc(
            text=[q.strip()], return_tensors="pt", padding=True, truncation=True
        ).to(DEVICE)
        with torch.no_grad():
            output = clip_model.get_text_features(**inputs)
        features = output.text_embeds if hasattr(output, "text_embeds") else output
        text_vec = _l2norm(features).squeeze(0).cpu().numpy().astype(np.float32)

    if image_bytes:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = clip_proc(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            v = clip_model.get_image_features(**inputs)
        img_vec = _l2norm(v).squeeze(0).cpu().numpy().astype(np.float32)

    if text_vec is None and img_vec is None:
        raise ValueError("Provide text query or an image.")

    if text_vec is None:
        vec = img_vec
    elif img_vec is None:
        vec = text_vec
    else:
        vec = alpha * text_vec + (1.0 - alpha) * img_vec

    vec = vec / (np.linalg.norm(vec) + 1e-12)
    return vec.astype(np.float32).tolist()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def _qdrant_query(query_vector, limit):
    return client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )

def format_hits(hits):
    out = []
    for r in hits:
        p = r.payload or {}
        image_key = p.get("image_key", "")
        out.append({
            "score":     float(r.score),
            "item_id":   p.get("item_id", ""),
            "title":     p.get("title", ""),
            "caption":   p.get("caption", ""),
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
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/search")
@limiter.limit("20/minute")
def search_get(
    request: Request,  # required by slowapi rate limiter
    q: str = Query(..., min_length=2, max_length=MAX_QUERY_LENGTH),
    k: int = Query(5, ge=1, le=20),
    alpha: float = Query(ALPHA, ge=0.0, le=1.0),
):
    route = "/search(GET)"
    REQS.labels(route).inc()
    try:
        with LAT.labels(route).time():
            qv = build_query_vector(q, None, alpha=alpha)
            res = _qdrant_query(qv, k)
            return {"mode": "text", "query": q, "k": k, "alpha": alpha, "results": format_hits(res.points)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.error("GET /search failed", exc_info=True)
        raise HTTPException(status_code=500, detail="Search failed. Please try again.")

@app.post("/search")
@limiter.limit("20/minute")
async def search_post(
    request: Request,  # required by slowapi rate limiter
    q: str = Query("", max_length=MAX_QUERY_LENGTH),
    k: int = Query(5, ge=1, le=20),
    alpha: float = Query(ALPHA, ge=0.0, le=1.0),
    image: UploadFile | None = File(default=None),
):
    route = "/search(POST)"
    REQS.labels(route).inc()

    image_bytes = None
    if image:
        if image.content_type not in ALLOWED_IMG_TYPES:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported image type '{image.content_type}'. Accepted: JPEG, PNG, WebP.",
            )
        image_bytes = await image.read()
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Image exceeds {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit.",
            )

    if not q.strip() and not image_bytes:
        raise HTTPException(status_code=400, detail="Provide a text query or an image.")

    mode = "text+image" if (q.strip() and image_bytes) else ("image" if image_bytes else "text")
    logger.info("POST /search collection=%s k=%d mode=%s", COLLECTION, k, mode)

    try:
        with LAT.labels(route).time():
            qv = build_query_vector(q, image_bytes, alpha=alpha)
            res = _qdrant_query(qv, k)
            return {"mode": mode, "query": q, "k": k, "alpha": alpha, "results": format_hits(res.points)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.error("POST /search failed", exc_info=True)
        raise HTTPException(status_code=500, detail="Search failed. Please try again.")
