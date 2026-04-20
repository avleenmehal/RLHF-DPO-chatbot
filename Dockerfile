# ── Serving container — no training deps ─────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# System deps for psycopg2 + FAISS
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK stop words at build time so containers start instantly
RUN python -c "import nltk; nltk.download('stopwords', quiet=True)"

# Copy application code
COPY . .

# Create local data dir (used when VECTOR_STORE_PATH is a local path)
RUN mkdir -p data

# Cloud Run injects PORT — default to 8080
ENV PORT=8080
EXPOSE 8080

CMD uvicorn server:app --host 0.0.0.0 --port ${PORT}
