
import json
import gzip
import csv
from pathlib import Path
import pandas as pd

LISTINGS_GZ = Path("data/raw/listings/metadata/listings_0.json.gz")
IMAGES_CSV = Path("data/raw/images/metadata/images.csv")          # can also be images.csv.gz
IMAGES_CSV_GZ = Path("data/raw/images/metadata/images.csv.gz")    # optional
RAW_ROOT = Path("data/raw")                                       # where images were extracted
OUT_MANIFEST = Path("data/processed/image_manifest.csv")
MAX_ITEMS = 5000

def load_images_lookup():
    if IMAGES_CSV.exists():
        df = pd.read_csv(IMAGES_CSV)
    elif IMAGES_CSV_GZ.exists():
        df = pd.read_csv(IMAGES_CSV_GZ, compression="gzip")
    else:
        raise FileNotFoundError("Could not find images.csv or images.csv.gz")
    
    cols = {c.lower(): c for c in df.columns}
    if "image_id" not in cols:
        raise ValueError(f"images.csv columns: {list(df.columns)}. Expected an image_id column.")

    image_id_col = cols["image_id"]
    path_col = None
    for candidate in ["path", "file_path", "filename", "relative_path", "key"]:
        if candidate in cols:
            path_col = cols[candidate]
            break

    if not path_col:
        raise ValueError(
            f"images.csv columns: {list(df.columns)}. Could not find a path-like column."
        )

    lookup = dict(zip(df[image_id_col].astype(str), df[path_col].astype(str)))
    return lookup

def pick_english_value(multi_list, preferred=("en_US", "en_GB", "en")):
    if not multi_list:
        return ""
    if isinstance(multi_list, list):
    # exact preferred tags first
        for tag in preferred:
            for entry in multi_list:
                if isinstance(entry, dict) and entry.get("language_tag") == tag:
                    return entry.get("value", "") or ""
        for entry in multi_list:
            if isinstance(entry, dict):
                lang = (entry.get("language_tag") or "").lower()
                if lang.startswith("en"):
                    return entry.get("value", "") or ""
        if isinstance(multi_list, str):
            return multi_list
        return ""

def find_local_image_file(rel_path: str):
    rel_path = rel_path.strip().lstrip("/\\")
    direct = RAW_ROOT / rel_path
    if direct.exists():
        return str(direct)
    
    fname = Path(rel_path).name
    if not fname:
        return ""
    
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        for p in RAW_ROOT.rglob(fname):
             if p.exists():
                return str(p)
    return ""

def iter_listings(path_gz: Path):
    with gzip.open(path_gz, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def main():
    print("Loading images.csv lookup...")
    image_lookup = load_images_lookup()
    print(f"Loaded {len(image_lookup):,} image_id mappings")

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    kept =0
    scanned =0

    with open(OUT_MANIFEST, "w",newline ="",encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f,
                                fieldnames=["item_id", "title_en", "main_image_id", "image_rel_path", "image_local_path"])

        writer.writeheader()

        print(f"Reading listings from: {LISTINGS_GZ}")
        for rec in iter_listings(LISTINGS_GZ):
            scanned +=1
            item_id = str(rec.get("item_id","")).strip()
            main_image_id = str(rec.get("main_image_id", "")).strip()
            if not item_id:
                item_id = str(rec.get("product_id", "")).strip()
            title_en = pick_english_value(rec.get("item_name"))
            if not item_id or not main_image_id or not title_en:
                continue
            rel_path = image_lookup.get(main_image_id, "")
            if not rel_path:
                continue

            local_path = find_local_image_file(rel_path)
            if not local_path:
                continue
            writer.writerow({
            "item_id": item_id,
            "title_en": title_en,
            "main_image_id": main_image_id,
            "image_rel_path": rel_path,
            "image_local_path": local_path
             })
            kept += 1
            if kept >= MAX_ITEMS:
                break

    print(f"\nDone.")
    print(f"Scanned listings: {scanned:,}")
    print(f"Kept (has English title + image exists locally): {kept:,}")
    print(f"Manifest written to: {OUT_MANIFEST}")

if __name__ == "__main__":
    main()