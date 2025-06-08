# Use NVIDIA PyTorch base image with CUDA support
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the project files
COPY . .

# Copy .env (if needed at runtime, else load during build)
# ENV variables can also be baked in here if necessary
ENV HUGGINGFACEHUB_API_TOKEN=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

# Expose FastAPI port
EXPOSE 8000

# Start the app with Ray Serve
CMD ["python", "app.py"]
