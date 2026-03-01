# ShopTalk — Multimodal Product Search

> *Can a search engine understand what you're looking for, even when words aren't enough?*

**ShopTalk** is a semantic search application that lets you find products using natural language, images, or both at once. It combines OpenAI's CLIP vision-language model with the Qdrant vector database to match queries against a catalog of products indexed from the Amazon Berkeley Objects dataset.

---

## Overview

Traditional keyword search breaks when shoppers can't name what they're looking for. ShopTalk addresses this by encoding products and queries into a shared 512-dimensional embedding space, enabling searches like:

- **Text:** *"abstract blue orange geometric"* — finds products matching the description even without exact keyword matches
- **Image:** upload a photo — returns visually similar items
- **Multimodal:** text + image together — blends both signals for more precise results

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Pipeline (offline)                  │
│                                                                 │
│  Amazon Berkeley Objects Dataset                                │
│         │                                                       │
│         ▼                                                       │
│  build-manifest.py ──► image_manifest.csv                      │
│         │                                                       │
│         ▼                                                       │
│  caption_with_blip.py ──► captions.csv     (BLIP model)        │
│         │                                                       │
│         ▼                                                       │
│  build_products.py ──► products_small.csv                      │
│         │                                                       │
│         ▼                                                       │
│  index_qdrant_clip.py ──────────────────────────────────┐      │
│         │  (CLIP text + image → 512-dim fused vector)    │      │
│                                                          │      │
└──────────────────────────────────────────────────────────┼──────┘
                                                           │
                                                           ▼
┌──────────────────┐    HTTP/REST    ┌───────────────┐   ┌────────────────────┐
│  Streamlit UI    │◄──────────────►│  FastAPI      │◄──│  Qdrant            │
│  :8501           │  multipart     │  :8000        │   │  :6333             │
│                  │  form-data     │               │   │  products_clip     │
│  • text input    │                │  /search GET  │   │  512-dim COSINE    │
│  • image upload  │                │  /search POST │   │                    │
│  • result cards  │                │  /health      │   └────────────────────┘
└──────────────────┘                │  /metrics     │
                                    └───────┬───────┘
                                            │ presigned URLs
                                            ▼
                                     ┌─────────────┐
                                     │  AWS S3     │
                                     │  product    │
                                     │  images     │
                                     └─────────────┘
```

---

## Dataset

**[Amazon Berkeley Objects (ABO)](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)** — a large-scale multimodal dataset released by Amazon and UC Berkeley under CC-BY-4.0.

| Property | Details |
|----------|---------|
| Products | ~147,000 unique listings |
| Languages | Multilingual (English preferred) |
| Images | Small (max 256px) and original resolution |
| Metadata | Title, brand, color, material, dimensions |

The pipeline uses up to 5,000 products by default (configurable via `LIMIT`).

---

## How It Works

### 1. Embedding Strategy

Each product is embedded as a **fused CLIP vector** that blends text and image signals:

```
product_vector = α · clip_text(title + caption) + (1 - α) · clip_image(product_image)
```

`α = 0.5` by default (equal weight). The same fusion formula is applied at query time, enabling apples-to-apples comparison between queries and indexed products.

### 2. Image Captioning

Raw product images are captioned offline using **Salesforce BLIP** (`Salesforce/blip-image-captioning-base`). Captions augment sparse or missing product titles, improving text-side embedding quality.

### 3. Vector Search

Qdrant performs COSINE similarity search over the 512-dim vector collection. The API supports returning top-K results (default K=5).

### 4. Search Modes

| Mode | Endpoint | Input |
|------|----------|-------|
| Text | `GET /search?q=...&k=5` | query string |
| Image | `POST /search` | image file upload |
| Multimodal | `POST /search?q=...` | text + image |

---

## Getting Started

### Prerequisites

- Docker + Docker Compose
- AWS credentials in `~/.aws` (for S3 image access)
- Amazon Berkeley Objects raw data in `data/raw/`

### Run the Application

```bash
docker-compose up --build
```

| Service | URL |
|---------|-----|
| Search UI | http://localhost:8501 |
| API | http://localhost:8000 |
| API Health | http://localhost:8000/health |
| Qdrant Dashboard | http://localhost:6333/dashboard |

### Build the Search Index

Run these scripts once to process raw data and populate Qdrant:

```bash
# 1. Extract product metadata
python src/build-manifest.py

# 2. Generate image captions (GPU recommended)
python src/caption_with_blip.py

# 3. Merge into unified product records
python src/build_products.py

# 4. Embed and index into Qdrant (CLIP multimodal)
python src/index_qdrant_clip.py
```

---

## Configuration

Key environment variables (set in `docker-compose.yaml` or shell):

| Variable | Default | Description |
|----------|---------|-------------|
| `COLLECTION_NAME` | `products_clip` | Qdrant collection to query |
| `CLIP_MODEL` | `openai/clip-vit-base-patch32` | HuggingFace CLIP model |
| `ALPHA` | `0.5` | Text/image blend ratio (0=image only, 1=text only) |
| `S3_BUCKET` | `shoptalk-assistant` | S3 bucket for product images |
| `AWS_REGION` | `us-east-2` | AWS region |
| `SIGNED_URL_TTL` | `900` | Presigned URL lifetime (seconds) |
| `LIMIT` | `5000` | Max products to index |

---

## Repository Structure

```
shop-talk/
├── docker-compose.yaml          # Orchestrates all three services
├── services/
│   ├── api/
│   │   ├── main.py              # FastAPI app — search endpoints, CLIP inference
│   │   ├── dockerfile
│   │   └── requirements.txt
│   └── ui/
│       ├── app.py               # Streamlit frontend
│       ├── dockerfile
│       └── requirements.txt
├── src/                         # Offline data pipeline scripts
│   ├── build-manifest.py        # Step 1: extract & join metadata
│   ├── caption_with_blip.py     # Step 2: BLIP image captioning
│   ├── build_products.py        # Step 3: merge into products CSV
│   ├── index_qdrant.py          # Index with text-only embeddings
│   └── index_qdrant_clip.py     # Index with CLIP multimodal embeddings
└── data/
    ├── raw/                     # Amazon Berkeley Objects dataset (not committed)
    ├── processed/               # Pipeline outputs (CSV files)
    └── qdrant_storage/          # Qdrant persistence volume
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Embeddings | [OpenAI CLIP](https://huggingface.co/openai/clip-vit-base-patch32) (vision-language) |
| Captioning | [Salesforce BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) |
| Vector DB | [Qdrant](https://qdrant.tech/) |
| API | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn |
| UI | [Streamlit](https://streamlit.io/) |
| Image Storage | AWS S3 |
| Monitoring | Prometheus (`/metrics`) |
| Container | Docker Compose |

---

## Data Attribution

Amazon Berkeley Objects dataset is used under the **Creative Commons Attribution 4.0 (CC-BY-4.0)** license.
Attribution: Jasmine Collins et al., Amazon and UC Berkeley, 2022.
