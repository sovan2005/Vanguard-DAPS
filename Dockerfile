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
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary application directories
RUN mkdir -p /app/models /app/data/originals /app/data/queries /app/db

# Copy the entire workspace into the container
COPY . .

# Ensure the setup scripts are run during the Docker image build phase
# 1. Download PyTorch models and generate synthetic base images
RUN python scripts/setup_ml_environment.py

# 2. Build the FAISS database and populate SQLite with testing variants
RUN python scripts/build_test_dataset.py

# Hugging Face Spaces exposes port 7860 by default
EXPOSE 7860

# Start the FastAPI engine using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
