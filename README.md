Visual Product Matcher

Visual Product Matcher is a web application that allows users to search for visually similar products using images. It leverages Google Gemini’s image embedding API to power visual search without requiring training a custom model.

The system is designed for speed, scalability, and easy deployment, combining a React frontend with a FastAPI backend.

Table of Contents

Features

Architecture

Getting Started

Usage

Deployment

Future Enhancements

Tech Stack

License

Features

Upload an image or provide an image URL to find visually similar products.

Fast search powered by precomputed Gemini embeddings.

Top-N results with cosine similarity scores and product metadata.

Basic filtering by minimum similarity threshold.

User-friendly UX with loading indicators and error handling.

Modular and well-documented code for easy extension.

Architecture
Frontend (React)
 ├─ Upload image / input URL
 └─ Display query image & results

Backend (FastAPI)
 ├─ Receive search request
 ├─ Obtain query embedding via Google Gemini API
 ├─ Compute cosine similarity with precomputed product embeddings
 ├─ Return top-N matches with metadata & similarity scores

Product Database
 └─ Metadata + precomputed Gemini embeddings for catalog images


Optimization: Precomputing product embeddings keeps queries fast and cost-effective. For larger catalogs, an ANN index like FAISS or Pinecone can be integrated for sub-linear retrieval.

Getting Started
Prerequisites

Python 3.9+

Node.js 18+

Gemini API credentials

Backend Setup
cd backend
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
pip install -r requirements.txt
uvicorn main:app --reload

Frontend Setup
cd frontend
npm install
npm start


The frontend will run at http://localhost:3000 by default.

Usage

Open the frontend in your browser.

Upload an image or enter an image URL.

Wait for the loading indicator.

View the top-N visually similar products with similarity scores and metadata.

Use the minimum similarity filter to refine results.

Deployment

Frontend: Deploy to Vercel or Netlify.

Backend: Deploy to Render, Railway, or any free-tier hosting.

For production, make sure to set environment variables for Gemini API keys and configure CORS correctly.

Future Enhancements

Integrate FAISS / Pinecone for faster search in larger catalogs.

Add category-based filtering and advanced search options.

Enable user authentication and save search history.

Improve UI/UX with infinite scroll and pagination for search results.

Tech Stack

Frontend: React, Tailwind CSS

Backend: FastAPI, Python

Embedding: Google Gemini API

Database: JSON / SQLite (small catalogs)
