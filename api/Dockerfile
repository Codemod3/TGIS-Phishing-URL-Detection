# Use official Python 3.10 slim image for a lightweight footprint
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /app

# Set working directory
WORKDIR /app

# Install system dependencies
# - gcc, build-essential: Needed for some Python packages (e.g., compiling extensions)
# - whois: Required for domain WHOIS lookups
# - libpq-dev: Required for psycopg2 (PostgreSQL adapter)
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    whois \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application codebase
COPY src/ src/
COPY api/ api/
COPY data/ data/
COPY config/ config/
COPY scripts/ scripts/

# Expose the API port
EXPOSE 8000

# Run the application with Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
