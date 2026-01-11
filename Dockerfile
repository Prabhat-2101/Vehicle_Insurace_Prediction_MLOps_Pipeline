FROM python:3.13.7-slim

WORKDIR /app

# Install system dependencies (optional but recommended)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose both ports
EXPOSE 8000
EXPOSE 8501

# Start both services
CMD ["bash", "start.sh"]
