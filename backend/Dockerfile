# PUMS backend image.
#
# Deliberately installs only `requirements.txt` — the classical inference path
# needs no PyTorch, which keeps the image around 400 MB instead of ~2.5 GB.
# Add `requirements-torch.txt` here if the torchvision adapter is needed.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libgomp1 is required by scikit-learn/scipy's OpenMP kernels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first so dependency layers cache across code changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY configs ./configs
COPY scripts ./scripts

RUN mkdir -p /app/var/analyses

ENV PUMS_STORAGE_DIR=/app/var/analyses

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
