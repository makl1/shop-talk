"""
Build a starter evaluation test set from the product catalog.

Strategy: for each sampled product, its own item_id must appear in the
top-K results when searching by title. This gives a weak but automatic
lower-bound signal — if a product can't find itself, something is wrong.

Usage:
    python eval/build_testset.py

Outputs:
    eval/testset.json   — edit this to add manual labels / extra queries
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

API_URL = "http://localhost:8000"
DATA    = Path("data/processed/products_small.csv")
OUT     = Path("eval/testset.json")
N       = 50   # number of products to sample
K       = 10   # results to fetch per query
SEED    = 42


def search(query: str, k: int) -> list[str]:
    resp = requests.get(f"{API_URL}/search", params={"q": query, "k": k}, timeout=30)
    resp.raise_for_status()
    return [r["item_id"] for r in resp.json().get("results", [])]


def main():
    if not DATA.exists():
        sys.exit(f"Missing {DATA} — run the data pipeline first.")

    df = pd.read_csv(DATA, dtype=str).fillna("")
    sample = df.sample(n=min(N, len(df)), random_state=SEED).reset_index(drop=True)

    entries = []
    for _, row in sample.iterrows():
        item_id = row["item_id"].strip()
        title   = row["title_en"].strip()
        if not title or not item_id:
            continue

        print(f"  querying: {title[:60]}")
        try:
            search(title, K)
        except Exception as e:
            print(f"    SKIP — API error: {e}")
            continue

        time.sleep(3.5)  # stay under the 20 req/min rate limit
        entries.append({
            "query":        title,
            "relevant_ids": [item_id],   # weak label: product must find itself
            "notes":        "auto-generated — review and add more relevant_ids manually",
        })

    OUT.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    print(f"\nWrote {len(entries)} entries to {OUT}")
    print("Review and extend relevant_ids before running evaluate.py.")


if __name__ == "__main__":
    main()
