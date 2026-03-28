# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create upload folder
RUN mkdir -p static/uploads

# Expose port
EXPOSE 5000

# Run with Gunicorn (production)
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]