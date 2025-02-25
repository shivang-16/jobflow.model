# Stage 1: Build Stage
FROM python:3.9-slim AS builder

# Set environment variables to prevent .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /jobflow-model

# Copy dependency file and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Stage 2: Final Image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/jobflow-model \
    PATH="/usr/local/bin:$PATH"

# Install required packages for runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /jobflow-model

# Copy only the installed packages and application from the builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /jobflow-model /jobflow-model
COPY --from=builder /usr/local/bin/gunicorn /usr/local/bin/

# Install dependencies in the final image
COPY requirements.txt .
RUN pip install --upgrade pip && pip install gunicorn

# Expose the port your app runs on
EXPOSE 5001

# Use Gunicorn to run the app
CMD ["gunicorn", "--workers", "4", "--timeout", "300", "--bind", "0.0.0.0:5001", "app:app"]

