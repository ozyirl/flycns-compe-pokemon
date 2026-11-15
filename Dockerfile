# Serves serve_fly:app, the Fly CNS battle policy behind POST /v1/choose.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # A decision takes about a millisecond; extra intra-op threads would only contend.
    OMP_NUM_THREADS=1 \
    # serve_fly.py listens on both IPv6 and IPv4 for "::"; Railway sets PORT.
    HOST=::

WORKDIR /app

# The CPU-only wheel is a fraction of the default CUDA build; requirements.txt then
# accepts it as torch==2.14.0.
COPY requirements.txt .
RUN pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY encoding.py serve_fly.py ./
COPY flycns/ flycns/
COPY artifacts/fly_connectome.npz artifacts/
COPY deploy/fly_matchup.pt deploy/

RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 8000
CMD ["python", "serve_fly.py"]
