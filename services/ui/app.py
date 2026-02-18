import os
from pathlib import Path
import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("API_URL", "http://localhost:8000")  # local; in Docker set to http://api:8000

st.set_page_config(page_title="ShopTalk", layout="wide")

st.title("ShopTalk")
st.caption("Search over product text + image captions stored in Qdrant.")
st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 2rem; }
      .stTextInput > div > div > input { padding: 0.65rem; border-radius: 0.7rem; }
      .pill {
        display:inline-block; padding:0.18rem 0.55rem; border-radius:999px;
        border:1px solid rgba(120,120,120,.25); font-size:0.85rem;
        margin-right: .35rem;
      }
      .muted { opacity: 0.75; }
      .card {
        border:1px solid rgba(120,120,120,.22);
        border-radius: 1.1rem;
        padding: 1rem 1rem 0.9rem 1rem;
        background: rgba(255,255,255,.02);
      }
      .title { font-size: 1.12rem; font-weight: 700; line-height: 1.2; margin-bottom: .2rem; }
      .caption { font-size: 0.92rem; opacity: .85; margin-top: .35rem; }
      .imgwrap img { border-radius: 0.9rem; }
      hr { margin: 0.8rem 0; opacity: .25; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Settings")
    k = st.slider("Top K", 1, 10, 5)
    st.text_input("API URL", value=API_URL, disabled=True)
    show_images = st.toggle("Show images (if paths are accessible)", value=True)

q = st.text_input(
    "Search",
    placeholder="Try: cat phone case • abstract blue orange • samsung s8 flowers",
)
uploaded = st.file_uploader("Search by image (optional)", type=["jpg", "jpeg", "png"])


col1, col2 = st.columns([1, 5])
with col1:
    do_search = st.button("Search", use_container_width=True)
with col2:
    st.write("")

def to_container_path(p: str) -> str:
    if not p:
        return ""
    p = p.replace("\\", "/")  # windows -> linux separators
    if p.lower().startswith("data/"):
        return "/data/" + p[5:]  # replace leading "data/" with "/data/"
    return p


def try_load_image(path_str: str):
    """Loads local images if possible. Returns PIL Image or None."""
    if not path_str:
        return None
    p = Path(path_str)
    if p.exists() and p.is_file():
        try:
            return Image.open(p)
        except Exception:
            return None
    return None

if do_search and (q.strip() or uploaded):
    try:
        files = None
        if uploaded:
            files = {"image": (uploaded.name, uploaded.getvalue(), uploaded.type)}
        with st.spinner("Searching…"):
            resp = requests.post(
                f"{API_URL}/search",
                params={"q": q, "k": k},
                files=files,
                timeout=30,
            )
        if resp.status_code != 200:
            st.error(f"API error {resp.status_code}: {resp.text}")
        else:
            data = resp.json()
            results = data.get("results", [])

            st.write(f"**Results for:** `{data.get('query','')}`  |  **Top K:** {data.get('k', k)}")

            if not results:
                st.warning("No results returned. (Is the Qdrant collection empty?)")
            else:
                for r in results:
                    with st.container(border=True):
                        left, right = st.columns([1, 3])

                        title = r.get("title") or "(no title)"
                        item_id = r.get("item_id", "")
                        caption = r.get("caption", "")
                        score = r.get("score", 0.0)
                        image_url = r.get("image_url", "")
                        image_key = r.get("image_key", "")

                        with left:
                            if show_images and image_url:                               
                                st.image(image_url, use_container_width=True)
                            else:
                                st.caption("No image preview")
                           

                        with right:
                            st.subheader(title)
                            st.write(f"**ID:** `{item_id}`")
                            st.write(f"**Score:** `{float(score):.3f}`")
                            if caption:
                                st.caption(caption)

                            if image_key:
                                st.code(image_key, language="text")
                                st.code(image_url,  language="text")

    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach API at {API_URL}. Is FastAPI running?")
    except requests.exceptions.Timeout:
        st.error("API request timed out. Try lowering K or check server load.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
