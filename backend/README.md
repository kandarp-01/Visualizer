# Backend (FastAPI) for Visual Product Matcher

1. Create a virtualenv and install requirements:
   pip install -r requirements.txt

2. Set your Gemini API key:
   export GEMINI_API_KEY="your_key_here"   # Linux/Mac
   set GEMINI_API_KEY=your_key_here        # Windows (cmd)

3. Precompute product embeddings:
   python precompute_embeddings.py

4. Run the server:
   uvicorn main:app --host 0.0.0.0 --port 8000

Endpoints:
- GET /products
- POST /search (multipart form: file OR form field image_url, plus top_k, min_score)
