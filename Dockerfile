# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — builder : compile et installe les dependances dans un venv isole
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Outils de compilation (necessaires a certaines wheels) — restent dans ce stage
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Environnement virtuel dedie que l'on copiera tel quel dans le runtime
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — runtime : image finale, minimale et non-root
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# ── Variables d'environnement ────────────────────────────────────────────────
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Flags PaddlePaddle (miroir de main.py) : evite les crashs oneDNN/PIR
    FLAGS_use_mkldnn=0 \
    FLAGS_enable_pir_api=0 \
    PADDLE_DISABLE_MKLDNN=1 \
    FLAGS_minloglevel=3 \
    DNNL_MAX_CPU_ISA=VANILLA \
    ONEDNN_MAX_CPU_ISA=VANILLA

# ── Librairies systeme requises au runtime par PaddleOCR / OpenCV ───────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Utilisateur non-root ─────────────────────────────────────────────────────
RUN useradd --create-home --uid 10001 appuser

# ── Dependances Python (venv issu du builder) ───────────────────────────────
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Dossiers de travail ecrivables par l'utilisateur non-root
RUN mkdir -p /app/data/uploads /app/data/output \
    && chown -R appuser:appuser /app/data

USER appuser

# ── Pre-telechargement des modeles PaddleOCR (baked dans l'image) ───────────
# Avant le COPY app/ pour que les changements de code n'invalident pas ce layer (~97 MB)
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='fr', use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)"

# ── Code applicatif (change souvent — layers legers) ─────────────────────────
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser main.py ./main.py

EXPOSE 8000

# ── Healthcheck : verifie que l'API repond ──────────────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0) if urllib.request.urlopen('http://localhost:8000/health').status==200 else sys.exit(1)"

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
