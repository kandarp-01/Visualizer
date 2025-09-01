"""
Precompute embeddings for all products in products.json using Gemini.
Requires: GEMINI_API_KEY env var
Usage: GEMINI_API_KEY=... python precompute_embeddings.py
"""

import os
import json
import requests
import numpy as np
from time import sleep
from pathlib import Path
import google.generativeai as genai

PROD_FILE = Path(__file__).parent / "products.json"
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY environment variable before running.")

genai.configure(api_key=API_KEY)

def fetch_image_bytes(url):
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.content

def get_image_embedding_bytes(img_bytes, model="embedding-001"):
    model_client = genai.GenerativeModel(model)
    res = model_client.embed(content=img_bytes)
    return np.array(res["embedding"], dtype=float)

def main():
    with open(PROD_FILE, "r", encoding="utf-8") as f:
        products = json.load(f)

    for i, p in enumerate(products):
        if p.get("embedding"):
            print(f"[{i}] {p['id']} already has embedding — skipping.")
            continue

        print(f"[{i}] Fetching image for {p['id']} from {p['image_url']}")
        try:
            b = fetch_image_bytes(p["image_url"])
            emb = get_image_embedding_bytes(b)
            p["embedding"] = emb.tolist()
            print(f" -> embedding length: {len(emb)}")
        except Exception as e:
            print(f"Failed for {p['id']}: {e}")
            p["embedding"] = []
        sleep(0.5)

    with open(PROD_FILE, "w", encoding="utf-8") as f:
        json.dump(products, f, indent=2)
    print("Done: products.json updated with embeddings.")

if __name__ == "__main__":
    main()
