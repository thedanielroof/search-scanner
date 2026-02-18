FROM python:3.11-slim

# System dependencies for OpenCV and audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY image_finder.py .

# Create data directory
RUN mkdir -p /root/.searchscanner/history

# Default port (overridden by PORT env var on cloud platforms)
ENV PORT=8457
EXPOSE ${PORT}

CMD ["python", "image_finder.py", "--no-browser"]
