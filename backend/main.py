import os
import json
import io
import numpy as np
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import google.generativeai as genai


# config
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is required.")
genai.configure(api_key=API_KEY)

BASE_DIR = os.path.dirname(__file__)
PRODUCTS_PATH = os.path.join(BASE_DIR, "products.json")

app = FastAPI(title="Visual Product Matcher - Backend")

# allow CORS for local dev / deployed frontend domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for production, lock this down to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load products
with open(PRODUCTS_PATH, "r", encoding="utf-8") as f:
    PRODUCTS = json.load(f)

# convert embeddings to numpy arrays for fast compute (if present)
for p in PRODUCTS:
    emb = p.get("embedding")
    p["_np_embedding"] = np.array(emb) if emb else None

class ImageUrlRequest(BaseModel):
    image_url: str
    top_k: Optional[int] = 5
    min_score: Optional[float] = 0.0

def get_image_bytes_from_url(url: str) -> bytes:
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return r.content

def embed_image_bytes(img_bytes: bytes, model="embedding-001"):
    # call Gemini to get embedding for the image bytes
    m = genai.GenerativeModel(model)
    res = m.embed(content=img_bytes)
    emb = res["embedding"]
    return np.array(emb, dtype=float)

def compute_similarities(query_emb: np.ndarray):
    candidates = []
    for p in PRODUCTS:
        emb = p.get("_np_embedding")
        if emb is None or emb.size == 0:
            continue
        score = cosine_similarity_manual(query_emb, emb)
        candidates.append({
            "id": p["id"],
            "name": p.get("name"),
            "category": p.get("category"),
            "image_url": p.get("image_url"),
            "score": score
        })
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


@app.get("/products")
def list_products():
    # return metadata (omit internal np arrays)
    return [
        {k: v for k, v in p.items() if k not in ("_np_embedding", "embedding")}
        for p in PRODUCTS
    ]

@app.post("/search")
async def search_image(
    image_url: Optional[str] = Form(None),
    top_k: Optional[int] = Form(5),
    min_score: Optional[float] = Form(0.0),
    file: Optional[UploadFile] = File(None)
):
    # Accept file OR image_url. If both provided, file takes precedence.
    if file is None and not image_url:
        raise HTTPException(status_code=400, detail="Provide a file or image_url.")

    try:
        if file:
            body = await file.read()
        else:
            body = get_image_bytes_from_url(image_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch image: {e}")

    try:
        q_emb = embed_image_bytes(body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    results = compute_similarities(q_emb)
    # apply min_score filter and top_k
    filtered = [r for r in results if r["score"] >= float(min_score)]
    return {"query_image_preview": "received", "results": filtered[:int(top_k)]}
