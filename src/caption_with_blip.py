from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

MANIFEST = Path("data/processed/image_manifest.csv")
OUT_CAPTIONS = Path("data/processed/captions.csv")

MODEL_NAME = "Salesforce/blip-image-captioning-base"

def main():
    df = pd.read_csv(MANIFEST)
    df =df.dropna(subset=["image_local_path", "item_id"])
    df["image_local_path"] = df["image_local_path"].astype(str)

    done = set()
    if OUT_CAPTIONS.exists():
        prev = pd.read_csv(OUT_CAPTIONS)
        done = set(prev["item_id"].astype(str).tolist())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    OUT_CAPTIONS.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    with OUT_CAPTIONS.open("a", encoding="utf-8", newline="") as f:
        if OUT_CAPTIONS.stat().st_size == 0:
            f.write("item_id,caption\n")
        for _,r in tqdm(df.iterrows(), total = len(df)):
            item_id = str(r["item_id"])
            if item_id in done:
                continue
            img_path = r["image_local_path"]
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                print("uable to open file")
                continue
            inputs = processor(images=image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens = 30)
            caption = processor.decode(out[0],skip_special_tokens = True)
            safe_caption = caption.replace('"', '""')
            f.write(f'{item_id},"{safe_caption}"\n')
            rows_written += 1
            
        print(f"Done. Added {rows_written} captions -> {OUT_CAPTIONS}")
if __name__ == "__main__":
    main()
