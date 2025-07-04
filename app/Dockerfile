# app/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY app.py .
COPY ../app/models/ ./models/
COPY templates/ ./templates/
COPY static/ ./static/

EXPOSE 5000

# Run Flask app
CMD ["python", "app.py"]
