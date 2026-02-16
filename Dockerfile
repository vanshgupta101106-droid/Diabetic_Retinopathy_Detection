# Dockerfile for Diabetic Retinopathy Detection API

# 1. Base Image with Python 3.9 (Slim for smaller size)
FROM python:3.9-slim

# 2. Set Environment Variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# 3. Install System Dependencies (OpenCV requires libgl1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 4. Set Workspace
WORKDIR /app

# 5. Install Python Dependencies
# Use a constraint-based installation if possible, or copy requirements
COPY environment.yaml /app/environment.yaml
# Since environment.yaml is for conda, let's look for requirements.txt or create one via pip
# For this Dockerfile, we'll install key packages explicitly based on usage
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    fastapi uvicorn python-multipart \
    opencv-python-headless \
    pandas numpy matplotlib scikit-learn \
    pillow pydantic requests

# 6. Copy Source Code
COPY src/ /app/src/
COPY configs/ /app/configs/

# 7. Create directory for models (and optional: copy local weights if available)
RUN mkdir -p /app/models
# COPY models/*.pth /app/models/ # Uncomment if you want to bake weights into image

# 8. Expose Port
EXPOSE 8000

# 9. Startup Command
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
