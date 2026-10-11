FROM node:22-bookworm-slim AS frontend
WORKDIR /build
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 AEROFOLD_DATA_DIR=/app/runtime HF_HOME=/app/runtime/huggingface
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt
# The upstream package stores weights beside its code. Fetch at build time so
# the unprivileged runtime never needs to write to site-packages.
RUN python -c "import ablang2; ablang2.pretrained(model_to_use='ablang2-paired', random_init=False)"
COPY backend/ ./backend/
COPY data/ ./data/
COPY --from=frontend /build/dist ./frontend/dist/
RUN useradd --create-home --uid 10001 aerofold && mkdir -p /app/runtime && chown -R aerofold:aerofold /app/runtime
USER aerofold
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
