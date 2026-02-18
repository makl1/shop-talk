import pandas as pd
from pathlib import Path

# ---- paths ----
DATA_DIR = Path("data")
CAPTIONS = DATA_DIR / "processed" / "captions.csv"
IMAGES   = DATA_DIR / "processed" / "image_manifest.csv"
OUT      = DATA_DIR / "processed" / "products_small.csv"

# ---- load ----
caps = pd.read_csv(CAPTIONS)
imgs = pd.read_csv(IMAGES)

# EXPECTED image_manifest columns:
# item_id,image_local_path,title_en (title optional)

# ---- merge ----
df = imgs.merge(caps, on="item_id", how="inner")

# ---- build doc text (THIS IS IMPORTANT) ----
def build_doc(row):
    parts = []
    if "title_en" in row and pd.notna(row["title_en"]):
        parts.append(f"TITLE: {row['title_en']}")
    parts.append(f"IMAGE_CAPTION: {row['caption']}")
    return "\n".join(parts)

df["doc"] = df.apply(build_doc, axis=1)

# ---- sanity checks ----
assert df["item_id"].isna().sum() == 0
assert df["doc"].str.len().min() > 10

# ---- keep only what we need ----
cols = ["item_id", "title_en", "caption", "image_local_path", "doc"]
cols = [c for c in cols if c in df.columns]

df = df[cols]

# ---- write ----
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)

print(f"✅ Wrote {len(df)} rows to {OUT}")
print(df.head(3))
