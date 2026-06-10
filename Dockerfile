# Build from the umbrella root so the sibling engine repo is in scope:
#   cd backend-rag-context-pipeline && docker compose up --build
# (compose sets context: .. and dockerfile: backend-rag-context-pipeline/Dockerfile)
#
# The image mirrors the umbrella layout (/app/backend-… + /app/engine-…) so the
# ENGINE_DIR sys.path bridge in api/main.py works unchanged.
FROM python:3.12-slim

# Install torch from the CPU-only index first; PyPI's default wheels (amd64 AND
# aarch64) bundle the multi-GB CUDA/nvidia libraries, which this API never uses.
# The requirements install below then sees torch as already satisfied.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY backend-rag-context-pipeline/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Bake the embedding model into the image (~130 MB) so container start needs no
# HuggingFace download. Must match EMBEDDING_MODEL in the indexing repo's
# build_index.py — rebuild with --build-arg if that changes.
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('$EMBEDDING_MODEL')"

COPY engine-rag-context-pipeline/retriever.py \
     engine-rag-context-pipeline/chain.py \
     engine-rag-context-pipeline/load_index.py \
     /app/engine-rag-context-pipeline/
COPY backend-rag-context-pipeline/api/ /app/backend-rag-context-pipeline/api/

WORKDIR /app/backend-rag-context-pipeline
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
