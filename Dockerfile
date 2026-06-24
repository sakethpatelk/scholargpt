# Multi-stage build: keeps the final image lean by separating dep install
# from the app copy.

# ── Stage 1: dependency install ────────────────────────────────────────────
FROM python:3.11-slim AS deps

WORKDIR /install

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime image ─────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from the deps stage
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy application source
COPY . .

# Persist ChromaDB and uploaded files outside the container layer via volumes.
# docker run -v ./data:/app/data ...
VOLUME ["/app/data"]

# Streamlit listens on 8501 by default.
EXPOSE 8501

# Disable Streamlit's browser-open-on-start behaviour in a container.
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501

CMD ["streamlit", "run", "app.py"]
