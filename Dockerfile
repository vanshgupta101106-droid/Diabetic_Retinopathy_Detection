# Dockerfile for Diabetic Retinopathy Detection API

# 1. Base Image with Python 3.10 (Slim for smaller size)
FROM python:3.10-slim

# 2. Set Environment Variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# 3. Install System Dependencies
# libgl1 is often needed for opencv even in headless mode on some slim distros,
# but mostly for 'opencv-python'. We keep it for safety.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 4. Set Workspace
WORKDIR /app

# 5. Install Dependencies (Optimized for Cache)
# 5a. Install PyTorch CPU-only first (Large download, changes rarely)
# We do this explicitly before pyproject.toml to avoid pulling the huge CUDA version
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 5b. Install Project Dependencies from pyproject.toml
# We copy the definition files and the source code (required for pip install .)
COPY pyproject.toml README.md /app/
COPY src/ /app/src/
COPY configs/ /app/configs/

# Install the package in editable mode or standard mode
# This installs all dependencies listed in pyproject.toml (fastapi, opencv, etc.)
RUN pip install --no-cache-dir .

# 6. Create directory for models
RUN mkdir -p /app/models

# 7. Expose Port
EXPOSE 8000

# 8. Startup Command
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
