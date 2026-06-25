# Build from this repo's own root:
#   cd backend-rag-context-pipeline && docker compose up --build
# (compose sets context: . and dockerfile: Dockerfile)
#
# The query engine is installed as the `rag-engine` package straight from its
# Git repo (see requirements.txt) — no sibling source is copied in, so the build
# context is just this backend repo.
FROM python:3.12-slim

# git is needed to install the rag-engine dependency from its git+https URL.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Install torch from the CPU-only index first; PyPI's default wheels (amd64 AND
# aarch64) bundle the multi-GB CUDA/nvidia libraries, which this API never uses.
# Keep this line identical to the engine Dockerfile's so the layer is shared.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Bake the embedding model into the image (~130 MB) so container start needs no
# HuggingFace download. Must match EMBEDDING_MODEL in the indexing repo's
# build_index.py — rebuild with --build-arg if that changes.
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('$EMBEDDING_MODEL')"

COPY api/ /app/api/

WORKDIR /app
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
