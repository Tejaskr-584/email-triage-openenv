FROM python:3.11-slim

WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies (standardized deploy step)
RUN pip install --no-cache-dir -r requirements.txt

# Ensure the project root is importable (so `env.*` and `tasks.*` work)
ENV PYTHONPATH=/app

# Runtime defaults
ENV PYTHONUNBUFFERED=1

# Run baseline inference to validate the environment end-to-end
CMD ["python", "inference.py"]
