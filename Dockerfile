FROM python:3.12-slim

WORKDIR /app

# Enable unbuffered stdout for logging
ENV PYTHONUNBUFFERED=1

# Install required system dependencies (libmcp-dev for faiss, others for pillow/requests)
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .

# Install Torch CPU-only specifically to save space/RAM during build
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary application directories
RUN mkdir -p /app/models /app/data/originals /app/data/queries /app/db

# Copy the entire workspace into the container
COPY . .

# Hugging Face Spaces exposes port 7860 by default
EXPOSE 7860

# Start the FastAPI engine using Uvicorn
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
