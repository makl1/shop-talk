"""
Evaluate retrieval quality against a labeled test set.

Metrics computed per query, then averaged:
  Precision@K  — fraction of top-K results that are relevant
  Recall@K     — fraction of relevant items found in top-K
  MRR          — 1/rank of the first relevant result (0 if none in top-K)
  NDCG@K       — ranked relevance; penalises relevant items ranked lower

Usage:
    python eval/evaluate.py [--testset eval/testset.json] [--k 5] [--api http://localhost:8000]

Outputs a per-query table and an aggregate summary.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import requests


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top = retrieved[:k]
    return sum(1 for r in top if r in relevant) / k if k else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = retrieved[:k]
    return sum(1 for r in top if r in relevant) / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for rank, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    def dcg(ids):
        return sum(
            (1.0 if ids[i] in relevant else 0.0) / math.log2(i + 2)
            for i in range(min(k, len(ids)))
        )

    actual_dcg = dcg(retrieved)
    # ideal: put all relevant items first
    ideal = list(relevant) + [""] * k
    ideal_dcg = dcg(ideal)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def search(api_url: str, query: str, k: int, alpha: float | None = None) -> list[str]:
    params = {"q": query, "k": k}
    if alpha is not None:
        params["alpha"] = alpha
    resp = requests.get(f"{api_url}/search", params=params, timeout=30)
    resp.raise_for_status()
    return [r["item_id"] for r in resp.json().get("results", [])]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", default="eval/testset.json")
    parser.add_argument("--k",       type=int,   default=5)
    parser.add_argument("--alpha",   type=float, default=None, help="Override ALPHA (0=image-only, 1=text-only). Omit to use server default.")
    parser.add_argument("--api",     default="http://localhost:8000")
    args = parser.parse_args()

    testset_path = Path(args.testset)
    if not testset_path.exists():
        sys.exit(f"Test set not found: {testset_path}\nRun eval/build_testset.py first.")

    entries = json.loads(testset_path.read_text())
    if not entries:
        sys.exit("Test set is empty.")

    K = args.k
    col_w = 55

    header = f"{'Query':<{col_w}} {'P@K':>6} {'R@K':>6} {'MRR':>6} {'NDCG@K':>7}"
    alpha_label = f"alpha={args.alpha}" if args.alpha is not None else "alpha=server-default"
    print(f"\nEvaluation  |  k={K}  |  {alpha_label}  |  n={len(entries)} queries")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    results = []
    errors  = 0

    for entry in entries:
        query    = entry["query"]
        relevant = set(entry.get("relevant_ids", []))

        if not relevant:
            continue

        try:
            retrieved = search(args.api, query, K, alpha=args.alpha)
            time.sleep(3.5)  # stay under the 20 req/min rate limit
        except Exception as e:
            print(f"  ERROR on '{query[:40]}': {e}")
            errors += 1
            continue

        p  = precision_at_k(retrieved, relevant, K)
        r  = recall_at_k(retrieved, relevant, K)
        m  = mrr(retrieved, relevant)
        nd = ndcg_at_k(retrieved, relevant, K)

        results.append((p, r, m, nd))

        label = query[:col_w - 2] + ".." if len(query) > col_w else query
        print(f"{label:<{col_w}} {p:>6.3f} {r:>6.3f} {m:>6.3f} {nd:>7.3f}")

    if not results:
        print("\nNo results — check API is running and test set has relevant_ids.")
        return

    n = len(results)
    avg_p  = sum(r[0] for r in results) / n
    avg_r  = sum(r[1] for r in results) / n
    avg_m  = sum(r[2] for r in results) / n
    avg_nd = sum(r[3] for r in results) / n

    print("=" * len(header))
    print(f"{'AVERAGE':<{col_w}} {avg_p:>6.3f} {avg_r:>6.3f} {avg_m:>6.3f} {avg_nd:>7.3f}")
    print()
    print(f"Queries evaluated : {n}")
    print(f"API errors skipped: {errors}")
    print()
    print("Interpretation guide:")
    print(f"  Precision@{K}  {avg_p:.3f}  — of every {K} results, ~{avg_p*K:.1f} are relevant")
    print(f"  Recall@{K}     {avg_r:.3f}  — found {avg_r*100:.0f}% of known relevant items in top {K}")
    print(f"  MRR          {avg_m:.3f}  — first relevant result appears at avg rank ~{1/avg_m:.1f}" if avg_m > 0 else f"  MRR          {avg_m:.3f}  — no relevant item found in top {K}")
    print(f"  NDCG@{K}      {avg_nd:.3f}  — ranking quality (1.0 = perfect order)")


if __name__ == "__main__":
    main()
