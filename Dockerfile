# ---------- Base image using Python 3.12 ----------
FROM python:3.12-slim

# Do not create .pyc files, always flush logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ---------- System dependencies ----------
# Needed for numpy, pydantic, fastapi, faiss-cpu, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---------- Working directory ----------
WORKDIR /code

# ---------- Install Python dependencies ----------
COPY requirements.txt .

# Upgrade pip and install all packages from requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------- Copy the entire project ----------
COPY . .

# ---------- HuggingFace uses PORT env automatically ----------
ENV PORT=7860

# ---------- Run FastAPI using Uvicorn ----------
CMD ["bash", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
