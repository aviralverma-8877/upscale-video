# ── Base: PyTorch 2.1.2 + CUDA 12.1 + Python 3.10 ──────────────────────────
# Python 3.10 is intentional: basicsr==1.4.2 has a setup.py bug on 3.12+
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /app

# System libraries required by OpenCV (headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install app dependencies
# torch & torchvision are already present in the base image — skip them here
RUN pip install --no-cache-dir \
        opencv-python-headless>=4.8.0 \
        basicsr==1.4.2 \
        realesrgan==0.3.0 \
        "flask>=3.0.0" \
        "imageio-ffmpeg>=0.5.0" \
        "numpy>=1.24.0,<2" \
        "pillow>=10.0.0"

# Copy application source
COPY app.py .
COPY templates/ templates/

# Ensure data directories exist inside the image
# (they will be overridden by the volume mounts at runtime)
RUN mkdir -p input output models

EXPOSE 7000

ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
